import argparse
import datetime
import gc
from importlib.util import find_spec
import json
import random
import os
import re
import time
import copy
from types import SimpleNamespace
from typing import Tuple, Optional, List, Any, Dict, Union

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, PngImagePlugin

from library import (
    anima_er_sde_sampling,
    anima_models,
    anima_train_utils,
    anima_utils,
    hunyuan_image_utils,
    qwen_image_autoencoder_kl,
    strategy_anima,
    strategy_base,
)
from library.device_utils import clean_memory_on_device, synchronize_device

lycoris_available = find_spec("lycoris") is not None
if lycoris_available:
    from lycoris.kohya import create_network_from_weights

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

# Valid choices for the --sampler / --scheduler options. Used both to define the argparse choices and
# to decide whether a sampler/scheduler pulled from PNG metadata maps to something this script supports.
SAMPLER_OPTION_CHOICES = ("euler", "er_sde", "euler_ancestral")
SCHEDULER_OPTION_CHOICES = ("default", "beta57", "simple")


def map_metadata_value_to_script_option(raw_value: Optional[str], valid_options: Tuple[str, ...]) -> Optional[str]:
    """Map a raw metadata value (e.g. an A1111 'Sampler'/'Schedule type') to one of valid_options.

    Normalizes case and spaces/hyphens to underscores, then returns the match if it is one of
    valid_options, otherwise None (so the caller falls back to the provided/default value).
    """
    if not raw_value:
        return None
    normalized = raw_value.strip().lower().replace(" ", "_").replace("-", "_")
    return normalized if normalized in valid_options else None


class GenerationSettings:
    def __init__(self, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None):
        self.device = device
        self.dit_weight_dtype = dit_weight_dtype  # not used currently because model may be optimized


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="HunyuanImage inference script")

    parser.add_argument(
        "--dit",
        type=str,
        default=None,
        help="DiT path. Also accepts an all-in-one checkpoint (civitai/ComfyUI CheckpointSave) with the "
        "DiT, VAE, and text encoder baked in; the components are auto-detected and extracted once to a "
        "sibling folder named for the model (reused on later runs), and used instead of --vae/--text_encoder.",
    )
    parser.add_argument("--vae", type=str, default=None, help="VAE directory or path")
    parser.add_argument(
        "--vae_chunk_size",
        type=int,
        default=None,
        help="Spatial chunk size for VAE encoding/decoding to reduce memory usage. Must be even number. If not specified, chunking is disabled (official behavior)."
        + " / メモリ使用量を減らすためのVAEエンコード/デコードの空間チャンクサイズ。偶数である必要があります。未指定の場合、チャンク処理は無効になります（公式の動作）。",
    )
    parser.add_argument(
        "--vae_disable_cache",
        action="store_true",
        help="Disable internal VAE caching mechanism to reduce memory usage. Encoding / decoding will also be faster, but this differs from official behavior."
        + " / VAEのメモリ使用量を減らすために内部のキャッシュ機構を無効にします。エンコード/デコードも速くなりますが、公式の動作とは異なります。",
    )
    parser.add_argument(
        "--qwen_image_vae_2d",
        action="store_true",
        help="Use the image-only 2D Qwen-Image VAE implementation. Official Qwen-Image VAE weights are converted on load."
        + " / 画像専用の2D Qwen-Image VAE実装を使用します。公式Qwen-Image VAEの重みはロード時に変換されます。",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="Text Encoder (Qwen3) path. Optional when --dit is an all-in-one checkpoint with the text "
        "encoder baked in (it is extracted and used automatically).",
    )

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument(
        "--lora_list",
        type=str,
        nargs="*",
        default=None,
        help="Inline list of LoRAs as a flat sequence of '<path> <multiplier>' tokens (multiplier optional, "
        "default 1.0). Use shell line-continuation to put one 'path multiplier' per line. Populates "
        "--lora_weight/--lora_multiplier.",
    )
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")
    parser.add_argument(
        "--lora_test_folder",
        type=str,
        nargs="+",
        default=None,
        metavar="FOLDER [MULTIPLIER]",
        help="LoRA A/B tester: run the entire otherwise-configured generation once PER top-level .safetensors "
        "in FOLDER (subfolders ignored), each time adding that one test LoRA on top of the fixed "
        "--lora_list/--lora_weight LoRAs at MULTIPLIER (default 1.0; models reload per test LoRA). If a "
        "'<loraname>.txt' sits next to a test LoRA, its text is injected after --pre_prompt and before the "
        "main prompt (the LoRA's trigger words). Total images = (#test LoRAs) x (images otherwise). The "
        "settings sidecar records the source image path and the test LoRA.",
    )

    # inference
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5, help="Guidance scale for classifier free guidance. Default is 3.5."
    )
    parser.add_argument("--prompt", type=str, default=None, help="prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt for generation, default is empty string")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="image size, height and width")
    parser.add_argument("--infer_steps", type=int, default=50, help="number of inference steps, default is 50")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for evaluation. Omit or pass -1 for a random seed."
    )
    parser.add_argument(
        "--images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt (currently applies to --from_image_embed). The seed "
        "increments by one per image, so each prompt yields N seed-varied iterations.",
    )

    # Flow Matching
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=5.0,
        help="Shift factor for flow matching schedulers. Default is 5.0.",
    )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")

    parser.add_argument("--text_encoder_cpu", action="store_true", help="Inference on CPU for Text Encoders")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "torch", "sageattn", "xformers", "sdpa"],  #  "sdpa" for backward compatibility
        help="attention mode",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="images",
        choices=["images", "latent", "latent_images"],
        help="output type",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="er_sde",
        choices=list(SAMPLER_OPTION_CHOICES),
        help="sampler: euler (deterministic flow Euler), er_sde (stochastic ER-SDE-Solver-3, Anima's "
        "recommended sampler and the default here), or euler_ancestral (Euler a; ancestral renoise each "
        "step, stochastic)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="beta57",
        choices=list(SCHEDULER_OPTION_CHOICES),
        help="sigma scheduler: default (flow-shifted linspace), beta57 (RES4LYF beta alpha=0.5/beta=0.7, "
        "more low-noise emphasis; the default here for Anima), or simple (ComfyUI simple_scheduler, even "
        "stride over the flow-shifted sigma table)",
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    parser.add_argument(
        "--lycoris", action="store_true", help=f"use lycoris for inference{'' if lycoris_available else ' (not available)'}"
    )

    # arguments for batch and interactive modes
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file")
    parser.add_argument(
        "--from_folder",
        type=str,
        default=None,
        help="Read prompts from every top-level .txt file in this folder (one prompt per file; subfolders ignored). "
        "Each file's text is combined with --pre_prompt (prefix) and --from_folder_settings (appended flags).",
    )
    parser.add_argument(
        "--from_folder_settings",
        type=str,
        default="",
        help="Flag string appended to every --from_folder prompt, e.g. '--w 832 --h 1216 --s 50 --l 3.5 --fs 1.0 --d 42'",
    )
    parser.add_argument(
        "--pre_prompt",
        type=str,
        default="",
        help="Text prepended to every --from_folder prompt (e.g. quality/style tags). Used verbatim.",
    )
    parser.add_argument(
        "--pre_prompt_neg",
        type=str,
        default="",
        help="Global negative prompt applied to every --from_folder prompt (used when guidance_scale > 1).",
    )
    parser.add_argument(
        "--from_image_embed",
        type=str,
        nargs="+",
        default=None,
        metavar="[prompts_only] [ignore_negative_prompt] [prompt_only_and_all_settings] FOLDER",
        help="Read prompts from the A1111 'parameters' metadata embedded in every top-level .png in FOLDER "
        "(one image per prompt; subfolders ignored). Positive and negative prompts are pulled from the "
        "metadata, and Steps/CFG scale/Seed/Size/Sampler/Scheduler are applied per-image when present "
        "(sampler/scheduler only when they map to a script choice). Prefix the folder with the literal "
        "keyword 'prompts_only' to pull ONLY the positive/negative prompts and keep all settings from the "
        "CLI args; 'ignore_negative_prompt' to discard the metadata negative (falls back to CLI / "
        "--pre_prompt_neg); or 'prompt_only_and_all_settings' to render a comparison PAIR per prompt (one "
        "prompt-only image and one all-metadata-settings image, both at the same --seed). Keywords may be "
        "combined in any order before FOLDER.",
    )
    parser.add_argument(
        "--prompt_count",
        type=int,
        default=None,
        help="Limit --from_folder / --from_file / --from_image_embed to the first N usable prompts "
        "(files sorted by name for --from_folder / --from_image_embed); default all",
    )
    parser.add_argument(
        "--prompt_count_skip_first",
        type=int,
        default=0,
        help="Skip the first N usable prompts before applying --prompt_count (for --from_folder / "
        "--from_file / --from_image_embed). Usable-counted, so it paginates cleanly: e.g. run once with "
        "--prompt_count 4, then again with --prompt_count_skip_first 4 --prompt_count 4 for the next 4.",
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: read prompts from console")

    args = parser.parse_args()

    normalize_from_image_embed_arg(args)

    # Validate arguments
    if sum(bool(x) for x in (args.from_file, args.from_folder, args.from_image_embed, args.interactive)) > 1:
        raise ValueError("Use only one of --from_file, --from_folder, --from_image_embed, or --interactive at the same time")

    if args.latent_path is None or len(args.latent_path) == 0:
        if (
            args.prompt is None
            and not args.from_file
            and not args.from_folder
            and not args.from_image_embed
            and not args.interactive
        ):
            raise ValueError("Either --prompt, --from_file, --from_folder, --from_image_embed or --interactive must be specified")

    if args.lora_test_folder and args.interactive:
        raise ValueError("--lora_test_folder cannot be combined with --interactive")

    if args.lora_test_folder and args.latent_path:
        raise ValueError("--lora_test_folder cannot be combined with --latent_path (latent decode does not use LoRAs)")

    if args.lycoris and not lycoris_available:
        raise ValueError("install lycoris: https://github.com/KohakuBlueleaf/LyCORIS")

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    return args


FROM_IMAGE_EMBED_PROMPTS_ONLY_KEYWORD = "prompts_only"
FROM_IMAGE_EMBED_IGNORE_NEGATIVE_KEYWORD = "ignore_negative_prompt"
FROM_IMAGE_EMBED_PROMPT_ONLY_AND_ALL_SETTINGS_KEYWORD = "prompt_only_and_all_settings"
FROM_IMAGE_EMBED_KEYWORDS = (
    FROM_IMAGE_EMBED_PROMPTS_ONLY_KEYWORD,
    FROM_IMAGE_EMBED_IGNORE_NEGATIVE_KEYWORD,
    FROM_IMAGE_EMBED_PROMPT_ONLY_AND_ALL_SETTINGS_KEYWORD,
)


def normalize_from_image_embed_arg(args: argparse.Namespace) -> None:
    """Split the raw --from_image_embed tokens into a folder path plus mode flags.

    The user may pass '<folder>', optionally preceded (in any order) by the literal keywords
    'prompts_only', 'ignore_negative_prompt', and/or 'prompt_only_and_all_settings'. After this runs,
    args.from_image_embed is the folder path string (or None), and args.from_image_embed_prompts_only /
    args.from_image_embed_ignore_negative_prompt / args.from_image_embed_prompt_only_and_all_settings
    are bools.
    """
    tokens = getattr(args, "from_image_embed", None)
    args.from_image_embed_prompts_only = False
    args.from_image_embed_ignore_negative_prompt = False
    args.from_image_embed_prompt_only_and_all_settings = False

    if not tokens:
        args.from_image_embed = None
        return

    remaining_tokens = list(tokens)
    while remaining_tokens and remaining_tokens[0] in FROM_IMAGE_EMBED_KEYWORDS:
        keyword = remaining_tokens.pop(0)
        if keyword == FROM_IMAGE_EMBED_PROMPTS_ONLY_KEYWORD:
            args.from_image_embed_prompts_only = True
        elif keyword == FROM_IMAGE_EMBED_IGNORE_NEGATIVE_KEYWORD:
            args.from_image_embed_ignore_negative_prompt = True
        else:
            args.from_image_embed_prompt_only_and_all_settings = True

    if len(remaining_tokens) != 1:
        raise ValueError(
            "--from_image_embed expects a single folder path, optionally preceded by the keyword(s) "
            f"{FROM_IMAGE_EMBED_KEYWORDS} (got: {getattr(args, 'from_image_embed')})"
        )

    args.from_image_embed = remaining_tokens[0]


def parse_a1111_png_prompt_metadata(parameters_text: str) -> Dict[str, Any]:
    """Parse an Automatic1111-style 'parameters' metadata string into prompt + settings overrides.

    The A1111 format is:
        <positive prompt, may span multiple lines>
        Negative prompt: <negative prompt, may span multiple lines>
        Steps: 20, Sampler: Euler a, CFG scale: 5.0, Seed: 12345, Size: 1024x1536, ...

    Returns a dict using the same override keys as parse_prompt_line (prompt, negative_prompt,
    infer_steps, guidance_scale, seed, image_size_width, image_size_height). Keys are only present
    when found in the metadata.
    """
    overrides: Dict[str, Any] = {}
    if not parameters_text or not parameters_text.strip():
        return overrides

    lines = parameters_text.split("\n")

    # The trailing settings line (if any) always starts with "Steps:" in A1111 output.
    settings_line = ""
    if lines and lines[-1].strip().startswith("Steps:"):
        settings_line = lines[-1].strip()
        body = "\n".join(lines[:-1])
    else:
        body = parameters_text

    negative_marker = "Negative prompt:"
    if negative_marker in body:
        marker_index = body.index(negative_marker)
        overrides["prompt"] = body[:marker_index].strip()
        overrides["negative_prompt"] = body[marker_index + len(negative_marker):].strip()
    else:
        overrides["prompt"] = body.strip()

    if settings_line:
        steps_match = re.search(r"Steps:\s*(\d+)", settings_line)
        if steps_match:
            overrides["infer_steps"] = int(steps_match.group(1))

        cfg_match = re.search(r"CFG scale:\s*([\d.]+)", settings_line)
        if cfg_match:
            overrides["guidance_scale"] = float(cfg_match.group(1))

        seed_match = re.search(r"Seed:\s*(\d+)", settings_line)
        if seed_match:
            overrides["seed"] = int(seed_match.group(1))

        # A1111 records Size as WIDTHxHEIGHT; this script's image_size is [height, width].
        size_match = re.search(r"Size:\s*(\d+)x(\d+)", settings_line)
        if size_match:
            overrides["image_size_width"] = int(size_match.group(1))
            overrides["image_size_height"] = int(size_match.group(2))

        # Sampler / scheduler are only carried over when the metadata value maps to a script option;
        # otherwise the key is omitted so the provided/default --sampler/--scheduler is used.
        sampler_match = re.search(r"Sampler:\s*([^,]+)", settings_line)
        if sampler_match:
            sampler_option = map_metadata_value_to_script_option(sampler_match.group(1), SAMPLER_OPTION_CHOICES)
            if sampler_option:
                overrides["sampler"] = sampler_option

        scheduler_match = re.search(r"(?:Schedule type|Scheduler):\s*([^,]+)", settings_line)
        if scheduler_match:
            scheduler_option = map_metadata_value_to_script_option(scheduler_match.group(1), SCHEDULER_OPTION_CHOICES)
            if scheduler_option:
                overrides["scheduler"] = scheduler_option

    return overrides


IMAGE_EMBED_PROMPT_KEYS = ("prompt", "negative_prompt")


def apply_image_embed_settings_gate(
    overrides: Dict[str, Any], prompts_only: bool, ignore_negative_prompt: bool = False
) -> Dict[str, Any]:
    """Decide which parsed metadata overrides to keep for a PNG.

    - prompts_only: keep only the positive/negative prompts; steps/guidance/seed/size come from CLI.
    - Otherwise settings mode requires BOTH Steps and CFG scale in the metadata. If either is missing,
      all settings are dropped and only the prompts are kept (revert to prompt-only). When settings
      mode applies, a missing Seed defaults to 0, while a missing Size is simply not overridden (the
      CLI-provided image size is used).
    - ignore_negative_prompt: drop the metadata negative prompt regardless of the above, so the
      negative falls back to the CLI value (blank or --pre_prompt_neg).
    """
    if prompts_only:
        gated_overrides = {key: value for key, value in overrides.items() if key in IMAGE_EMBED_PROMPT_KEYS}
    elif "infer_steps" in overrides and "guidance_scale" in overrides:
        gated_overrides = dict(overrides)
        if "seed" not in gated_overrides:
            gated_overrides["seed"] = 0
    else:
        gated_overrides = {key: value for key, value in overrides.items() if key in IMAGE_EMBED_PROMPT_KEYS}

    if ignore_negative_prompt:
        gated_overrides.pop("negative_prompt", None)

    return gated_overrides


def parse_comfyui_prompt_metadata(prompt_json_text: str) -> Dict[str, Any]:
    """Parse a ComfyUI 'prompt' node-graph JSON into prompt + settings overrides.

    Finds the sampler node (the one whose inputs carry both 'positive' and 'negative' links), follows
    those links to the CLIPTextEncode 'text', and reads seed/steps/cfg/sampler/scheduler from the
    sampler node and width/height from the linked latent image. Sampler/scheduler are only kept when
    they map to a script choice. Returns {} when no usable positive prompt is found.
    """
    try:
        graph = json.loads(prompt_json_text)
    except (ValueError, TypeError):
        return {}
    if not isinstance(graph, dict):
        return {}

    sampler_inputs = None
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        node_inputs = node.get("inputs")
        if isinstance(node_inputs, dict) and "positive" in node_inputs and "negative" in node_inputs:
            sampler_inputs = node_inputs
            break
    if sampler_inputs is None:
        return {}

    def linked_text(link_ref: Any) -> Optional[str]:
        if isinstance(link_ref, list) and link_ref:
            target_node = graph.get(str(link_ref[0]))
            if isinstance(target_node, dict):
                text_value = target_node.get("inputs", {}).get("text")
                if isinstance(text_value, str):
                    return text_value.strip()
        return None

    positive_prompt = linked_text(sampler_inputs.get("positive"))
    if not positive_prompt:
        return {}

    overrides: Dict[str, Any] = {"prompt": positive_prompt}

    negative_prompt = linked_text(sampler_inputs.get("negative"))
    if negative_prompt is not None:
        overrides["negative_prompt"] = negative_prompt

    steps_value = sampler_inputs.get("steps")
    if isinstance(steps_value, int):
        overrides["infer_steps"] = steps_value

    cfg_value = sampler_inputs.get("cfg")
    if isinstance(cfg_value, (int, float)):
        overrides["guidance_scale"] = float(cfg_value)

    seed_value = sampler_inputs.get("seed")
    if isinstance(seed_value, int):
        overrides["seed"] = seed_value

    sampler_option = map_metadata_value_to_script_option(sampler_inputs.get("sampler_name"), SAMPLER_OPTION_CHOICES)
    if sampler_option:
        overrides["sampler"] = sampler_option

    scheduler_option = map_metadata_value_to_script_option(sampler_inputs.get("scheduler"), SCHEDULER_OPTION_CHOICES)
    if scheduler_option:
        overrides["scheduler"] = scheduler_option

    latent_ref = sampler_inputs.get("latent_image")
    if isinstance(latent_ref, list) and latent_ref:
        latent_node = graph.get(str(latent_ref[0]))
        if isinstance(latent_node, dict):
            latent_inputs = latent_node.get("inputs", {})
            latent_width = latent_inputs.get("width")
            latent_height = latent_inputs.get("height")
            if isinstance(latent_width, int) and isinstance(latent_height, int):
                overrides["image_size_width"] = latent_width
                overrides["image_size_height"] = latent_height

    return overrides


def decode_exif_user_comment_bytes(raw_user_comment: Any) -> str:
    """Decode an EXIF UserComment value (per-spec 8-byte charset prefix) into text.

    Handles UNICODE (UTF-16, byte order guessed by which decoding yields more printable ASCII), ASCII,
    already-decoded str, and unknown/other as UTF-8. Returns '' when nothing decodable.
    """
    if isinstance(raw_user_comment, str):
        return raw_user_comment.strip()
    if not isinstance(raw_user_comment, (bytes, bytearray)):
        return ""

    charset_prefix = bytes(raw_user_comment[:8])
    comment_body = bytes(raw_user_comment[8:])

    if charset_prefix.startswith(b"UNICODE"):
        def printable_score(text: str) -> int:
            return sum(1 for character in text if 32 <= ord(character) < 127)

        candidates = [comment_body.decode(encoding, errors="replace") for encoding in ("utf-16-le", "utf-16-be")]
        return max(candidates, key=printable_score).strip()
    if charset_prefix.startswith(b"ASCII"):
        return comment_body.decode("ascii", errors="replace").strip()
    return bytes(raw_user_comment).decode("utf-8", errors="replace").strip()


def extract_exif_user_comment_text(exif) -> str:
    """Return the decoded EXIF UserComment text from a PIL Exif object, or '' if absent.

    UserComment (tag 0x9286) lives in the Exif sub-IFD (0x8769), so it is looked up there as well.
    """
    if not exif:
        return ""
    user_comment = exif.get(0x9286)
    if user_comment is None:
        try:
            exif_sub_ifd = exif.get_ifd(0x8769)
        except Exception:
            exif_sub_ifd = None
        if exif_sub_ifd:
            user_comment = exif_sub_ifd.get(0x9286)
    if not user_comment:
        return ""
    return decode_exif_user_comment_bytes(user_comment)


def read_png_parsed_metadata(png_path: str) -> Optional[Dict[str, Any]]:
    """Read a PNG's embedded generation metadata and return the raw (ungated) parsed overrides.

    Tries, in order: Automatic1111 'parameters' text, ComfyUI 'prompt' node graph, then the EXIF
    UserComment (which may itself be an A1111 parameters string). Returns the first parse that yields a
    positive prompt, or None if none do. Callers apply gating per render variant.
    """
    with Image.open(png_path) as image:
        image_info = dict(image.info)
        user_comment_text = extract_exif_user_comment_text(image.getexif())

    parameters_text = image_info.get("parameters")
    if parameters_text:
        a1111_overrides = parse_a1111_png_prompt_metadata(parameters_text)
        if a1111_overrides.get("prompt"):
            return a1111_overrides

    comfyui_prompt_text = image_info.get("prompt")
    if comfyui_prompt_text:
        comfyui_overrides = parse_comfyui_prompt_metadata(comfyui_prompt_text)
        if comfyui_overrides.get("prompt"):
            return comfyui_overrides

    if user_comment_text:
        exif_overrides = parse_a1111_png_prompt_metadata(user_comment_text)
        if exif_overrides.get("prompt"):
            return exif_overrides

    return None


def read_png_prompt_overrides(
    png_path: str, prompts_only: bool, ignore_negative_prompt: bool = False
) -> Optional[Dict[str, Any]]:
    """Read a PNG's A1111 'parameters' metadata and return gated prompt/settings overrides.

    Returns None when the PNG has no 'parameters' metadata or no positive prompt in it (the caller
    logs an error and skips). See apply_image_embed_settings_gate for how settings are gated.
    """
    parsed = read_png_parsed_metadata(png_path)
    if parsed is None:
        return None

    return apply_image_embed_settings_gate(parsed, prompts_only, ignore_negative_prompt)


def parse_prompt_line(line: str) -> Dict[str, Any]:
    """Parse a prompt line into a dictionary of argument overrides

    Args:
        line: Prompt line with options

    Returns:
        Dict[str, Any]: Dictionary of argument overrides
    """
    parts = line.split(" --")
    prompt = parts[0].strip()

    # Create dictionary of overrides
    overrides = {"prompt": prompt}

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        # Map options to argument names
        if option == "w":
            overrides["image_size_width"] = int(value)
        elif option == "h":
            overrides["image_size_height"] = int(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        elif option == "fs":
            overrides["flow_shift"] = float(value)
        elif option == "n":
            overrides["negative_prompt"] = value

    return overrides


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    """Apply overrides to args

    Args:
        args: Original arguments
        overrides: Dictionary of overrides

    Returns:
        argparse.Namespace: New arguments with overrides applied
    """
    args_copy = copy.deepcopy(args)

    for key, value in overrides.items():
        if key == "image_size_width":
            args_copy.image_size[1] = value
        elif key == "image_size_height":
            args_copy.image_size[0] = value
        else:
            setattr(args_copy, key, value)

    return args_copy


def check_inputs(args: argparse.Namespace) -> Tuple[int, int]:
    """Validate video size and length

    Args:
        args: command line arguments

    Returns:
        Tuple[int, int]: (height, width)
    """
    height = args.image_size[0]
    width = args.image_size[1]

    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

    return height, width


# region Model


def convert_peft_diffusion_model_lora_keys(lora_sd: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ComfyUI/PEFT LoRA keys to the down/up/alpha form the DiT merge hook expects.

    ComfyUI/PEFT LoRAs name keys 'diffusion_model.<module>.lora_A.weight' / '.lora_B.weight' / '.alpha'.
    The merge (library.lora_utils) matches a bare model weight key by replacing '.' with '_' and looking
    for '<name>.lora_down.weight' / '.lora_up.weight' / '.alpha' (empty prefix). So we strip
    'diffusion_model.', underscore the module path, and map lora_A->lora_down, lora_B->lora_up.
    Keys not matching these suffixes (or lacking the prefix) are skipped.
    """
    converted: Dict[str, Any] = {}
    for key, value in lora_sd.items():
        if not key.startswith("diffusion_model."):
            continue
        module_and_suffix = key[len("diffusion_model.") :]
        for source_suffix, target_suffix in ((".lora_A.weight", ".lora_down.weight"), (".lora_B.weight", ".lora_up.weight"), (".alpha", ".alpha")):
            if module_and_suffix.endswith(source_suffix):
                module_path = module_and_suffix[: -len(source_suffix)]
                converted[module_path.replace(".", "_") + target_suffix] = value
                break
    return converted


def select_dit_lora_state_dict(lora_sd: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-detect the LoRA format and return DiT LoRA weights in the merge's expected form.

    Prefers kohya 'lora_unet_' keys when present; otherwise converts ComfyUI/PEFT 'diffusion_model.'
    LoRAs. Returns an empty dict if neither format is found.
    """
    kohya_unet_lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
    if kohya_unet_lora_sd:
        return kohya_unet_lora_sd
    return convert_peft_diffusion_model_lora_keys(lora_sd)


def load_dit_model(
    args: argparse.Namespace, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None
) -> anima_models.Anima:
    """load DiT model

    Args:
        args: command line arguments
        device: device to use
        dit_weight_dtype: data type for the model weights. None for as-is

    Returns:
        anima_models.Anima: DiT model instance
    """
    # If LyCORIS is enabled, we will load the model to CPU and then merge LoRA weights (static method)

    loading_device = "cpu"
    if not args.lycoris:
        loading_device = device

    # load LoRA weights
    if not args.lycoris and args.lora_weight is not None and len(args.lora_weight) > 0:
        lora_weights_list = []
        for lora_weight in args.lora_weight:
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = load_file(lora_weight)  # load on CPU, dtype is as is
            # Keep kohya 'lora_unet_' keys, or auto-convert ComfyUI/PEFT 'diffusion_model.' LoRAs.
            lora_sd = select_dit_lora_state_dict(lora_sd)
            lora_weights_list.append(lora_sd)
    else:
        lora_weights_list = None

    loading_weight_dtype = dit_weight_dtype
    if args.fp8_scaled and not args.lycoris:
        loading_weight_dtype = None  # we will load weights as-is and then optimize to fp8

    model = anima_utils.load_anima_model(
        device,
        args.dit,
        args.attn_mode,
        True,  # enable split_attn to trim masked tokens
        loading_device,
        loading_weight_dtype,
        args.fp8_scaled and not args.lycoris,
        lora_weights_list=lora_weights_list,
        lora_multipliers=args.lora_multiplier,
    )
    if not args.fp8_scaled:
        # simple cast to dit_weight_dtype
        target_dtype = None  # load as-is (dit_weight_dtype == dtype of the weights in state_dict)
        if dit_weight_dtype is not None:  # in case of args.fp8 and not args.fp8_scaled
            logger.info(f"Convert model to {dit_weight_dtype}")
            target_dtype = dit_weight_dtype

        logger.info(f"Move model to device: {device}")
        target_device = device

        model.to(target_device, target_dtype)  # move and cast  at the same time. this reduces redundant copy operations

    # model.to(device)
    model.to(device, dtype=torch.bfloat16)  # ensure model is in bfloat16 for inference

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)

    return model


def load_text_encoder(
    args: argparse.Namespace, dtype: torch.dtype = torch.bfloat16, device: torch.device = torch.device("cpu")
) -> torch.nn.Module:
    lora_weights_list = None
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        lora_weights_list = []
        for lora_weight in args.lora_weight:
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = load_file(lora_weight)  # load on CPU, dtype is as is
            # lora_sd = filter_lora_state_dict(lora_sd, args.include_patterns, args.exclude_patterns)
            lora_sd = {
                "model_" + k[len("lora_te_") :]: v for k, v in lora_sd.items() if k.startswith("lora_te_")
            }  # only keep Text Encoder lora weights, remove prefix "lora_te_" and add "model_" prefix
            lora_weights_list.append(lora_sd)

    text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        args.text_encoder, dtype=dtype, device=device, lora_weights=lora_weights_list, lora_multipliers=args.lora_multiplier
    )
    text_encoder.eval()
    return text_encoder


# endregion


def decode_latent(
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage, latent: torch.Tensor, device: torch.device
) -> torch.Tensor:
    logger.info(f"Decoding image. Latent shape {latent.shape}, device {device}")

    vae.to(device)
    with torch.no_grad():
        pixels = vae.decode_to_pixels(latent.to(device, dtype=vae.dtype))
        # pixels = vae.decode(latent.to(device, dtype=torch.bfloat16), scale=vae_scale)
    if pixels.ndim == 5:  # remove frame dimension if exists, [B, C, F, H, W] -> [B, C, H, W]
        pixels = pixels.squeeze(2)

    pixels = pixels.to("cpu", dtype=torch.float32)  # move to CPU and convert to float32 (bfloat16 is not supported by numpy)
    vae.to("cpu")

    logger.info(f"Decoded. Pixel shape {pixels.shape}")
    return pixels[0]  # remove batch dimension


def process_escape(text: str) -> str:
    """Process escape sequences in text

    Args:
        text: Input text with escape sequences

    Returns:
        str: Processed text
    """
    return text.encode("utf-8").decode("unicode_escape")


def prepare_text_inputs(
    args: argparse.Namespace, device: torch.device, anima: anima_models.Anima, shared_models: Optional[Dict] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare text-related inputs for T2I: LLM encoding. Anima model is also needed for preprocessing"""

    # load text encoder: conds_cache holds cached encodings for prompts without padding
    conds_cache = {}
    text_encoder_device = torch.device("cpu") if args.text_encoder_cpu else device
    if shared_models is not None:
        text_encoder = shared_models.get("text_encoder")

        if "conds_cache" in shared_models:  # Use shared cache if available
            conds_cache = shared_models["conds_cache"]

        # text_encoder is on device (batched inference) or CPU (interactive inference)
    else:  # Load if not in shared_models
        text_encoder_dtype = torch.bfloat16  # Default dtype for Text Encoder
        text_encoder = load_text_encoder(args, dtype=text_encoder_dtype, device=text_encoder_device)
        text_encoder.eval()
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        # Store references so load_target_model can reuse them

    # Store original devices to move back later if they were shared. This does nothing if shared_models is None
    text_encoder_original_device = text_encoder.device if text_encoder else None

    # Ensure text_encoder is not None before proceeding
    if not text_encoder:
        raise ValueError("Text encoder is not loaded properly.")

    # Define a function to move models to device if needed
    # This is to avoid moving models if not needed, especially in interactive mode
    model_is_moved = False

    def move_models_to_device_if_needed():
        nonlocal model_is_moved
        nonlocal shared_models

        if model_is_moved:
            return
        model_is_moved = True

        logger.info(f"Moving Text Encoder to appropriate device: {text_encoder_device}")
        text_encoder.to(text_encoder_device)  # If text_encoder_cpu is True, this will be CPU

    logger.info("Encoding prompt with Text Encoder")

    prompt = process_escape(args.prompt)
    cache_key = prompt
    if cache_key in conds_cache:
        embed = conds_cache[cache_key]
    else:
        move_models_to_device_if_needed()

        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

        with torch.no_grad():
            # embed = anima_text_encoder.get_text_embeds(anima, tokenizer, text_encoder, t5xxl_tokenizer, prompt)
            tokens = tokenize_strategy.tokenize(prompt)
            embed = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
            crossattn_emb = anima._preprocess_text_embeds(
                source_hidden_states=embed[0].to(anima.device),
                target_input_ids=embed[2].to(anima.device),
                target_attention_mask=embed[3].to(anima.device),
                source_attention_mask=embed[1].to(anima.device),
            )
            crossattn_emb[~embed[3].bool()] = 0
            embed[0] = crossattn_emb
        embed[0] = embed[0].cpu()

        conds_cache[cache_key] = embed

    negative_prompt = process_escape(args.negative_prompt)
    cache_key = negative_prompt
    if cache_key in conds_cache:
        negative_embed = conds_cache[cache_key]
    else:
        move_models_to_device_if_needed()

        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

        with torch.no_grad():
            # negative_embed = anima_text_encoder.get_text_embeds(anima, tokenizer, text_encoder, t5xxl_tokenizer, negative_prompt)
            tokens = tokenize_strategy.tokenize(negative_prompt)
            negative_embed = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
            crossattn_emb = anima._preprocess_text_embeds(
                source_hidden_states=negative_embed[0].to(anima.device),
                target_input_ids=negative_embed[2].to(anima.device),
                target_attention_mask=negative_embed[3].to(anima.device),
                source_attention_mask=negative_embed[1].to(anima.device),
            )
            crossattn_emb[~negative_embed[3].bool()] = 0
            negative_embed[0] = crossattn_emb
        negative_embed[0] = negative_embed[0].cpu()

        conds_cache[cache_key] = negative_embed

    if not (shared_models and "text_encoder" in shared_models):  # if loaded locally
        # There is a bug text_encoder is not freed from GPU memory when text encoder is fp8
        del text_encoder
        gc.collect()  # This may force Text Encoder to be freed from GPU memory
    else:  # if shared, move back to original device (likely CPU)
        if text_encoder:
            text_encoder.to(text_encoder_original_device)

    clean_memory_on_device(device)

    arg_c = {"embed": embed, "prompt": prompt}
    arg_null = {"embed": negative_embed, "prompt": negative_prompt}

    return arg_c, arg_null


def generate(
    args: argparse.Namespace,
    gen_settings: GenerationSettings,
    shared_models: Optional[Dict] = None,
    precomputed_text_data: Optional[Dict] = None,
) -> torch.Tensor:
    """main function for generation

    Args:
        args: command line arguments
        shared_models: dictionary containing pre-loaded models (mainly for DiT)
        precomputed_image_data: Optional dictionary with precomputed image data
        precomputed_text_data: Optional dictionary with precomputed text data

    Returns:
        tuple: (HunyuanVAE2D model (vae) or None, torch.Tensor generated latent)
    """
    device, dit_weight_dtype = (gen_settings.device, gen_settings.dit_weight_dtype)

    # prepare seed
    seed = resolve_random_seed(args.seed)
    args.seed = seed  # set seed to args for saving

    if shared_models is None or "model" not in shared_models:
        # load DiT model
        anima = load_dit_model(args, device, dit_weight_dtype)

        if shared_models is not None:
            shared_models["model"] = anima
    else:
        # use shared model
        logger.info("Using shared DiT model.")
        anima: anima_models.Anima = shared_models["model"]

    if precomputed_text_data is not None:
        logger.info("Using precomputed text data.")
        context = precomputed_text_data["context"]
        context_null = precomputed_text_data["context_null"]

    else:
        logger.info("No precomputed data. Preparing image and text inputs.")
        context, context_null = prepare_text_inputs(args, device, anima, shared_models)

    return generate_body(args, anima, context, context_null, device, seed)


def generate_body(
    args: Union[argparse.Namespace, SimpleNamespace],
    anima: anima_models.Anima,
    context: Dict[str, Any],
    context_null: Optional[Dict[str, Any]],
    device: torch.device,
    seed: int,
) -> torch.Tensor:

    # set random generator
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)

    height, width = check_inputs(args)
    logger.info(f"Image size: {height}x{width} (HxW), infer_steps: {args.infer_steps}")

    # image generation ######

    logger.info(f"Prompt: {context['prompt']}")

    embed = context["embed"][0].to(device, dtype=torch.bfloat16)
    if context_null is None:
        context_null = context  # dummy for unconditional
    negative_embed = context_null["embed"][0].to(device, dtype=torch.bfloat16)

    # Prepare latent variables
    num_channels_latents = anima_models.Anima.LATENT_CHANNELS
    shape = (
        1,
        num_channels_latents,
        1,  # Frame dimension
        height // 8,  # qwen_image_autoencoder_kl.SCALE_FACTOR,
        width // 8,  # qwen_image_autoencoder_kl.SCALE_FACTOR,
    )
    latents = randn_tensor(shape, generator=seed_g, device=device, dtype=torch.bfloat16)

    # Create padding mask
    bs = latents.shape[0]
    h_latent = latents.shape[-2]
    w_latent = latents.shape[-1]
    padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=torch.bfloat16, device=device)

    logger.info(f"Embed: {embed.shape}, negative_embed: {negative_embed.shape}, latents: {latents.shape}")
    embed = embed.to(torch.bfloat16)
    negative_embed = negative_embed.to(torch.bfloat16)

    # Prepare sigmas according to the selected scheduler (flow sigmas in [0,1], descending to 0)
    if args.scheduler == "beta57":
        sigmas = anima_er_sde_sampling.build_beta57_sigmas(args.infer_steps, args.flow_shift, device)
    elif args.scheduler == "simple":
        sigmas = anima_er_sde_sampling.build_simple_sigmas(args.infer_steps, args.flow_shift, device)
    else:
        _timesteps, sigmas = hunyuan_image_utils.get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)

    do_cfg = args.guidance_scale != 1.0
    autocast_enabled = args.fp8

    def run_velocity_with_cfg(current_latents, sigma_scalar):
        # The Anima DiT consumes the flow sigma (in [0,1]) directly as its time input.
        time_input = sigma_scalar.to(device=device, dtype=torch.bfloat16).expand(current_latents.shape[0])
        model_input = current_latents.to(torch.bfloat16)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            velocity = anima(model_input, time_input, embed, padding_mask=padding_mask)
        if do_cfg:
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                uncond_velocity = anima(model_input, time_input, negative_embed, padding_mask=padding_mask)
            velocity = uncond_velocity + args.guidance_scale * (velocity - uncond_velocity)
        return velocity

    def predict_denoised_x0(current_latents, sigma_scalar):
        velocity = run_velocity_with_cfg(current_latents, sigma_scalar).to(torch.float32)
        return current_latents.to(torch.float32) - sigma_scalar.to(torch.float32) * velocity

    if args.sampler == "er_sde":
        latents = anima_er_sde_sampling.sample_er_sde_rectified_flow(
            predict_denoised_x0, latents, sigmas, seed=args.seed
        ).to(latents.dtype)
    elif args.sampler == "euler_ancestral":
        latents = anima_er_sde_sampling.sample_euler_ancestral_rectified_flow(
            predict_denoised_x0, latents, sigmas, seed=args.seed
        ).to(latents.dtype)
    else:
        with tqdm(total=len(sigmas) - 1, desc="Denoising steps") as pbar:
            for i in range(len(sigmas) - 1):
                noise_pred = run_velocity_with_cfg(latents, sigmas[i])
                latents = hunyuan_image_utils.step(latents, noise_pred, sigmas, i).to(latents.dtype)
                pbar.update()

    return latents


def stream_usable_prompt_overrides(items, load_overrides_for_item, prompt_count, skip_first=0):
    """Yield (index, item, overrides) in order, counting only usable items toward the window.

    load_overrides_for_item(item) returns the override dict for an item, or None when the item has no
    usable prompt (e.g. a PNG with no positive prompt, or an empty caption).

    skip_first: skip (do not yield) the first skip_first USABLE items; unusable items encountered while
    still skipping are passed over silently. This is usable-counted, so it paginates cleanly regardless
    of interspersed skips (e.g. run 1 with prompt_count=4, run 2 with skip_first=4 prompt_count=4).

    After the skip window, unusable items are yielded (with overrides=None) so the caller can log and
    skip them, but they do NOT count toward prompt_count. Iteration stops once prompt_count usable
    items have been yielded; prompt_count=None means no limit.
    """
    usable_skipped = 0
    usable_yielded = 0
    for index, item in enumerate(items):
        if prompt_count is not None and usable_yielded >= prompt_count:
            return
        overrides = load_overrides_for_item(item)
        if usable_skipped < skip_first:
            if overrides is not None:
                usable_skipped += 1
            continue
        yield index, item, overrides
        if overrides is not None:
            usable_yielded += 1


def resolve_random_seed(seed_value: Optional[int]) -> int:
    """Return seed_value, or a fresh random seed when randomization is requested.

    Randomization is requested when seed_value is None (no --seed given) or -1 (explicit random).
    """
    if seed_value is None or seed_value == -1:
        return random.randint(0, 2**32 - 1)
    return seed_value


def get_time_flag():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S-%f")[:-3]


def save_latent(latent: torch.Tensor, args: argparse.Namespace, height: int, width: int) -> str:
    """Save latent to file

    Args:
        latent: Latent tensor
        args: command line arguments
        height: height of frame
        width: width of frame

    Returns:
        str: Path to saved latent file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed

    latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"

    if args.no_metadata:
        metadata = None
    else:
        metadata = {
            "seeds": f"{seed}",
            "prompt": f"{args.prompt}",
            "height": f"{height}",
            "width": f"{width}",
            "infer_steps": f"{args.infer_steps}",
            # "embedded_cfg_scale": f"{args.embedded_cfg_scale}",
            "guidance_scale": f"{args.guidance_scale}",
        }
        if args.negative_prompt is not None:
            metadata["negative_prompt"] = f"{args.negative_prompt}"

    sd = {"latent": latent.contiguous()}
    save_file(sd, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")

    return latent_path


def save_images(
    sample: torch.Tensor,
    args: argparse.Namespace,
    original_base_name: Optional[str] = None,
    precomputed_image_name: Optional[str] = None,
) -> str:
    """Save images to directory

    Args:
        sample: Video tensor
        args: command line arguments
        original_base_name: Original base name (if latents are loaded from files)
        precomputed_image_name: If provided, use this exact base name for the PNG (its settings
            sidecar was already written before generation, so it is not re-written here).

    Returns:
        str: Path to saved images directory
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    if precomputed_image_name is not None:
        image_name = precomputed_image_name
    else:
        time_flag = get_time_flag()
        seed = args.seed
        original_name = "" if original_base_name is None else f"_{original_base_name}"
        image_name = f"{time_flag}_{seed}{original_name}"
        # Non-batch modes write the sidecar here (batch mode writes it before generation).
        write_generation_settings_sidecar(save_path, image_name, args)

    x = torch.clamp(sample, -1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    x = x.transpose(1, 2, 0)  # C, H, W -> H, W, C

    image = Image.fromarray(x)

    png_info = None
    if not args.no_metadata:
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("parameters", build_png_generation_metadata_text(args))
    image.save(os.path.join(save_path, f"{image_name}.png"), pnginfo=png_info)

    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


def save_output(
    args: argparse.Namespace,
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    latent: torch.Tensor,
    device: torch.device,
    original_base_name: Optional[str] = None,
    precomputed_image_name: Optional[str] = None,
) -> None:
    """save output

    Args:
        args: command line arguments
        vae: VAE model
        latent: latent tensor
        device: device to use
        original_base_name: original base name (if latents are loaded from files)
    """
    height, width = latent.shape[-2], latent.shape[-1]  # BCTHW
    height *= 8  # qwen_image_autoencoder_kl.SCALE_FACTOR
    width *= 8  # qwen_image_autoencoder_kl.SCALE_FACTOR
    # print(f"Saving output. Latent shape {latent.shape}; pixel shape {height}x{width}")
    if args.output_type == "latent" or args.output_type == "latent_images":
        # save latent
        save_latent(latent, args, height, width)
    if args.output_type == "latent":
        return

    if vae is None:
        logger.error("VAE is None, cannot decode latents for saving video/images.")
        return

    if latent.ndim == 2:  # S,C. For packed latents from other inference scripts
        latent = latent.unsqueeze(0)
        height, width = check_inputs(args)  # Get height/width from args
        latent = latent.view(
            1,
            vae.latent_channels,
            1,  # Frame dimension
            height // 8,  # qwen_image_autoencoder_kl.SCALE_FACTOR,
            width // 8,  # qwen_image_autoencoder_kl.SCALE_FACTOR,
        )

    image = decode_latent(vae, latent, device)

    if args.output_type == "images" or args.output_type == "latent_images":
        # save images
        if original_base_name is None:
            original_name = ""
        else:
            original_name = f"_{original_base_name}"
        save_images(image, args, original_name, precomputed_image_name=precomputed_image_name)


def preprocess_prompts_for_batch(prompt_lines: List[str], base_args: argparse.Namespace) -> List[Dict]:
    """Process multiple prompts for batch mode

    Args:
        prompt_lines: List of prompt lines
        base_args: Base command line arguments

    Returns:
        List[Dict]: List of prompt data dictionaries
    """
    prompts_data = []

    for line in prompt_lines:
        line = line.strip()
        if not line or line.startswith("#"):  # Skip empty lines and comments
            continue

        # Parse prompt line and create override dictionary
        prompt_data = parse_prompt_line(line)
        logger.info(f"Parsed prompt data: {prompt_data}")
        prompts_data.append(prompt_data)

    return prompts_data


def build_repeated_single_prompt_data(prompt: str, base_seed: int, images_per_prompt: int) -> List[Dict]:
    """Return prompts_data (for process_batch_prompts) that renders images_per_prompt copies of one
    prompt with consecutive seeds base_seed, base_seed+1, ... So single-prompt mode can produce N
    seed-incremented images in a single run (one model load), matching --images_per_prompt elsewhere.
    """
    count = max(1, images_per_prompt)
    return [{"prompt": prompt, "seed": base_seed + iteration} for iteration in range(count)]


def apply_pre_prompt_to_batch_prompts(
    prompts_data: List[Dict], pre_prompt_prefix: str, pre_prompt_negative: str
) -> List[Dict]:
    """Apply --pre_prompt / --pre_prompt_neg to a --from_file batch (mutates and returns prompts_data).

    pre_prompt_prefix is prepended to each prompt; pre_prompt_negative is used as the negative for any
    line that does not already specify one (a per-line '--n' negative is preserved). Both are no-ops
    when empty.
    """
    prefix = (pre_prompt_prefix or "").strip()
    negative = (pre_prompt_negative or "").strip()
    for prompt_data in prompts_data:
        if prefix:
            prompt_data["prompt"] = f"{prefix} {prompt_data['prompt']}".strip()
        if negative and not prompt_data.get("negative_prompt"):
            prompt_data["negative_prompt"] = negative
    return prompts_data


def load_shared_models(args: argparse.Namespace) -> Dict:
    """Load shared models for batch processing or interactive mode.
    Models are loaded to CPU to save memory. VAE is NOT loaded here.
    DiT model is also NOT loaded here, handled by process_batch_prompts or generate.

    Args:
        args: Base command line arguments

    Returns:
        Dict: Dictionary of shared models (text/image encoders)
    """
    shared_models = {}
    # Load text encoders to CPU
    text_encoder_dtype = torch.bfloat16  # Default dtype for Text Encoder
    text_encoder = load_text_encoder(args, dtype=text_encoder_dtype, device=torch.device("cpu"))
    shared_models["text_encoder"] = text_encoder
    return shared_models


def process_batch_prompts(prompts_data: List[Dict], args: argparse.Namespace) -> None:
    """Process multiple prompts with model reuse and batched precomputation

    Args:
        prompts_data: List of prompt data dictionaries
        args: Base command line arguments
    """
    if not prompts_data:
        logger.warning("No valid prompts found")
        return

    gen_settings = get_generation_settings(args)
    dit_weight_dtype = gen_settings.dit_weight_dtype
    device = gen_settings.device

    # 1. Prepare VAE
    logger.info("Loading VAE for batch generation...")
    vae_for_batch = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
    vae_for_batch.to(torch.bfloat16)
    vae_for_batch.eval()

    all_prompt_args_list = [apply_overrides(args, pd) for pd in prompts_data]  # Create all arg instances first
    for prompt_args in all_prompt_args_list:
        check_inputs(prompt_args)  # Validate each prompt's height/width

    # 2. Load DiT Model once
    logger.info("Loading DiT model for batch generation...")
    # Use args from the first prompt for DiT loading (LoRA etc. should be consistent for a batch)
    first_prompt_args = all_prompt_args_list[0]
    anima = load_dit_model(first_prompt_args, device, dit_weight_dtype)  # Load directly to target device if possible

    shared_models_for_generate = {"model": anima}  # Pass DiT via shared_models

    # 3. Precompute Text Data (Text Encoder)
    logger.info("Loading Text Encoder for batch text preprocessing...")

    # Text Encoder loaded to CPU by load_text_encoder
    text_encoder_dtype = torch.bfloat16  # Default dtype for Text Encoder
    text_encoder_batch = load_text_encoder(args, dtype=text_encoder_dtype, device=torch.device("cpu"))

    # Text Encoder to device for this phase
    text_encoder_device = torch.device("cpu") if args.text_encoder_cpu else device
    text_encoder_batch.to(text_encoder_device)  # Moved into prepare_text_inputs logic

    all_precomputed_text_data = []
    conds_cache_batch = {}

    logger.info("Preprocessing text and LLM/TextEncoder encoding for all prompts...")
    temp_shared_models_txt = {
        "text_encoder": text_encoder_batch,  # on GPU if not text_encoder_cpu
        "conds_cache": conds_cache_batch,
    }

    for i, prompt_args_item in enumerate(all_prompt_args_list):
        logger.info(f"Text preprocessing for prompt {i+1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")

        # prepare_text_inputs will move text_encoders to device temporarily
        context, context_null = prepare_text_inputs(prompt_args_item, device, anima, temp_shared_models_txt)
        text_data = {"context": context, "context_null": context_null}
        all_precomputed_text_data.append(text_data)

    # Models should be removed from device after prepare_text_inputs
    del text_encoder_batch, temp_shared_models_txt, conds_cache_batch
    gc.collect()  # Force cleanup of Text Encoder from GPU memory
    clean_memory_on_device(device)

    # Keep the DiT and VAE both resident so each prompt is generated, decoded, and saved before
    # moving to the next prompt, rather than saving everything at the end. The settings .txt is
    # written BEFORE generation (using a base name reused by the PNG) so you can read what is being
    # generated while it renders. This is the inference path (no gradients), so both models fit; if it
    # ever OOMs, decode can be moved back to a separate post-generation phase.
    if args.output_type != "latent":
        vae_for_batch.to(device)

    os.makedirs(args.save_path, exist_ok=True)

    logger.info("Generating and saving each prompt's output before moving to the next...")
    with torch.no_grad():
        for i, prompt_args_item in enumerate(all_prompt_args_list):
            current_text_data = all_precomputed_text_data[i]
            height, width = check_inputs(prompt_args_item)  # Get height/width for each prompt

            # Reserve the base name and write the settings sidecar before generation starts.
            image_base_name = f"{get_time_flag()}_{prompt_args_item.seed}"
            if prompt_args_item.output_type != "latent":
                write_generation_settings_sidecar(args.save_path, image_base_name, prompt_args_item)

            logger.info(f"Generating {i+1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")
            try:
                # generate uses precomputed text data and the resident DiT (shared_models_for_generate).
                latent = generate(prompt_args_item, gen_settings, shared_models_for_generate, current_text_data)

                if latent is None:
                    continue

                if prompt_args_item.output_type in ["latent", "latent_images"]:
                    save_latent(latent, prompt_args_item, height, width)

                if prompt_args_item.output_type != "latent":
                    # latent_images already saved the latent above; decode + save the image now.
                    if prompt_args_item.output_type == "latent_images":
                        prompt_args_item.output_type = "images"
                    save_output(prompt_args_item, vae_for_batch, latent, device, precomputed_image_name=image_base_name)

                del latent
            except Exception as e:
                logger.error(f"Error generating/saving prompt: {prompt_args_item.prompt}. Error: {e}", exc_info=True)
                continue

    # Free DiT and VAE
    logger.info("Releasing DiT and VAE from memory...")
    del shared_models_for_generate["model"]
    del anima
    if args.output_type != "latent":
        vae_for_batch.to("cpu")
    del vae_for_batch
    clean_memory_on_device(device)
    synchronize_device(device)


def process_interactive(args: argparse.Namespace) -> None:
    """Process prompts in interactive mode

    Args:
        args: Base command line arguments
    """
    gen_settings = get_generation_settings(args)
    device = gen_settings.device
    shared_models = load_shared_models(args)
    shared_models["conds_cache"] = {}  # Initialize empty cache for interactive mode

    vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
    vae.to(torch.bfloat16)
    vae.eval()

    print("Interactive mode. Enter prompts (Ctrl+D or Ctrl+Z (Windows) to exit):")

    try:
        import prompt_toolkit
    except ImportError:
        logger.warning("prompt_toolkit not found. Using basic input instead.")
        prompt_toolkit = None

    if prompt_toolkit:
        session = prompt_toolkit.PromptSession()

        def input_line(prompt: str) -> str:
            return session.prompt(prompt)

    else:

        def input_line(prompt: str) -> str:
            return input(prompt)

    try:
        while True:
            try:
                line = input_line("> ")
                if not line.strip():
                    continue
                if len(line.strip()) == 1 and line.strip() in ["\x04", "\x1a"]:  # Ctrl+D or Ctrl+Z with prompt_toolkit
                    raise EOFError  # Exit on Ctrl+D or Ctrl+Z

                # Parse prompt
                prompt_data = parse_prompt_line(line)
                prompt_args = apply_overrides(args, prompt_data)

                # Generate latent
                # For interactive, precomputed data is None. shared_models contains text encoders.
                latent = generate(prompt_args, gen_settings, shared_models)

                # # If not one_frame_inference, move DiT model to CPU after generation
                # model = shared_models.get("model")
                # model.to("cpu")  # Move DiT model to CPU after generation

                # Save latent and video
                # returned_vae from generate will be used for decoding here.
                save_output(prompt_args, vae, latent, device)

            except KeyboardInterrupt:
                print("\nInterrupted. Continue (Ctrl+D or Ctrl+Z (Windows) to exit)")
                continue

    except EOFError:
        print("\nExiting interactive mode")


def process_folder_streaming(args: argparse.Namespace) -> None:
    """Stream prompts from a folder of caption/tag .txt files, one file at a time.

    For each top-level .txt file (sorted by name; --prompt_count limits the total), build the prompt
    (--pre_prompt + caption + --from_folder_settings), write the settings sidecar, generate, then save
    the image before moving on. Models are loaded once and reused; captions are read and encoded one
    file at a time, so a folder with thousands of files is not processed up front.
    """
    gen_settings = get_generation_settings(args)
    device = gen_settings.device

    # Text encoder in shared_models; the DiT is loaded lazily by generate() and cached here for reuse.
    shared_models = load_shared_models(args)
    shared_models["conds_cache"] = {}

    vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
    vae.to(torch.bfloat16)
    vae.eval()
    if args.output_type != "latent":
        vae.to(device)

    folder = args.from_folder
    file_names = sorted(
        name
        for name in os.listdir(folder)
        if name.lower().endswith(".txt") and os.path.isfile(os.path.join(folder, name))
    )

    pre_prompt = args.pre_prompt.strip()
    pre_prompt_neg = args.pre_prompt_neg.strip()
    settings = args.from_folder_settings.strip()
    os.makedirs(args.save_path, exist_ok=True)

    def load_caption_overrides(file_name):
        # Return None (unusable skip, not counted toward --prompt_count) on read error or empty caption.
        try:
            with open(os.path.join(folder, file_name), "r", encoding="utf-8") as caption_file:
                caption = caption_file.read().strip()
        except Exception as exc:
            logger.error(f"Error reading caption file {file_name}: {exc}", exc_info=True)
            return None
        prompt_body = f"{pre_prompt} {caption}".strip() if pre_prompt else caption
        if not prompt_body:
            return None
        prompt_line = f"{prompt_body} {settings}".strip() if settings else prompt_body
        return parse_prompt_line(prompt_line)

    logger.info(
        f"Streaming up to {args.prompt_count if args.prompt_count is not None else 'all'} usable prompt(s) "
        f"from {len(file_names)} caption file(s) in {folder}"
    )
    for index, file_name, overrides in stream_usable_prompt_overrides(
        file_names, load_caption_overrides, args.prompt_count, skip_first=args.prompt_count_skip_first
    ):
        try:
            if overrides is None:
                logger.warning(f"[{index + 1}/{len(file_names)}] {file_name}: empty caption, skipping")
                continue

            prompt_args = apply_overrides(args, overrides)
            if prompt_args.seed is None:
                prompt_args.seed = random.randint(0, 2**32 - 1)
            if pre_prompt_neg:
                prompt_args.negative_prompt = pre_prompt_neg
            prompt_args.current_source_image_path = os.path.join(folder, file_name)

            logger.info(f"[{index + 1}/{len(file_names)}] {file_name}: {prompt_args.prompt}")

            # Write the settings sidecar before generation so it is readable while the image renders.
            image_base_name = f"{get_time_flag()}_{prompt_args.seed}"
            if prompt_args.output_type != "latent":
                write_generation_settings_sidecar(args.save_path, image_base_name, prompt_args)

            latent = generate(prompt_args, gen_settings, shared_models)

            if prompt_args.output_type in ["latent", "latent_images"]:
                height, width = check_inputs(prompt_args)
                save_latent(latent, prompt_args, height, width)
            if prompt_args.output_type != "latent":
                if prompt_args.output_type == "latent_images":
                    prompt_args.output_type = "images"
                save_output(prompt_args, vae, latent, device, precomputed_image_name=image_base_name)

            del latent
        except Exception as exc:
            logger.error(f"Error on caption file {file_name}: {exc}", exc_info=True)
            continue

    if args.output_type != "latent":
        vae.to("cpu")
    clean_memory_on_device(device)


def process_image_embed_streaming(args: argparse.Namespace) -> None:
    """Stream prompts from a folder of PNGs, reading each image's A1111 'parameters' metadata.

    For each top-level .png (sorted by name; --prompt_count limits usable prompts), pull the
    positive/negative prompt (and Steps/CFG/Seed/Size/Sampler/Scheduler, subject to mode) from the
    embedded metadata, write the settings sidecar, generate, then save the image before moving on.
    Modes (from the --from_image_embed keywords):
      - default: full settings mode (metadata settings applied, gated by apply_image_embed_settings_gate).
      - prompts_only: only prompts from metadata; all settings from CLI.
      - prompt_only_and_all_settings: render a comparison PAIR per prompt (one prompt-only image and
        one all-metadata-settings image) at the same provided --seed so only the settings differ.
    Models are loaded once and reused; metadata is read one file at a time. PNGs without a usable
    positive prompt are skipped (they do not count toward --prompt_count).
    """
    gen_settings = get_generation_settings(args)
    device = gen_settings.device

    # Text encoder in shared_models; the DiT is loaded lazily by generate() and cached here for reuse.
    shared_models = load_shared_models(args)
    shared_models["conds_cache"] = {}

    vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
    vae.to(torch.bfloat16)
    vae.eval()
    if args.output_type != "latent":
        vae.to(device)

    folder = args.from_image_embed
    file_names = sorted(
        name
        for name in os.listdir(folder)
        if name.lower().endswith(".png") and os.path.isfile(os.path.join(folder, name))
    )

    pre_prompt = args.pre_prompt.strip()
    pre_prompt_neg = args.pre_prompt_neg.strip()
    images_per_prompt = max(1, args.images_per_prompt)
    prompts_only = args.from_image_embed_prompts_only
    ignore_negative_prompt = args.from_image_embed_ignore_negative_prompt
    comparison_mode = args.from_image_embed_prompt_only_and_all_settings
    os.makedirs(args.save_path, exist_ok=True)

    def load_png_parsed(file_name):
        # Return None (unusable skip, not counted toward --prompt_count) on read/parse error or when
        # the PNG has no positive prompt, so one bad PNG does not abort the streaming run.
        try:
            return read_png_parsed_metadata(os.path.join(folder, file_name))
        except Exception as exc:
            logger.error(f"Error reading PNG metadata from {file_name}: {exc}", exc_info=True)
            return None

    def render_one_image(gated_overrides, seed_value, variant_label, source_image_path):
        """Apply overrides, force seed_value, and generate + save one image. variant_label (or None)
        is appended to the file base name so a comparison pair is distinguishable on disk.
        source_image_path is recorded in the settings sidecar."""
        local_overrides = gated_overrides
        if pre_prompt:
            local_overrides = dict(local_overrides)
            local_overrides["prompt"] = f"{pre_prompt} {local_overrides['prompt']}".strip()

        prompt_args = apply_overrides(args, local_overrides)
        if pre_prompt_neg:
            prompt_args.negative_prompt = f"{pre_prompt_neg} {prompt_args.negative_prompt}".strip()
        prompt_args.seed = seed_value
        prompt_args.current_source_image_path = source_image_path

        variant_suffix = f"_{variant_label}" if variant_label else ""
        # Write the settings sidecar before generation so it is readable while the image renders.
        image_base_name = f"{get_time_flag()}_{seed_value}{variant_suffix}"
        if prompt_args.output_type != "latent":
            write_generation_settings_sidecar(args.save_path, image_base_name, prompt_args)

        latent = generate(prompt_args, gen_settings, shared_models)

        if prompt_args.output_type in ["latent", "latent_images"]:
            height, width = check_inputs(prompt_args)
            save_latent(latent, prompt_args, height, width)
        if prompt_args.output_type != "latent":
            if prompt_args.output_type == "latent_images":
                prompt_args.output_type = "images"
            save_output(prompt_args, vae, latent, device, precomputed_image_name=image_base_name)

        del latent

    logger.info(
        f"Streaming up to {args.prompt_count if args.prompt_count is not None else 'all'} usable prompt(s) "
        f"from {len(file_names)} PNG(s) in {folder} "
        f"(prompts_only={prompts_only}, ignore_negative_prompt={ignore_negative_prompt}, "
        f"prompt_only_and_all_settings={comparison_mode}, images_per_prompt={images_per_prompt})"
    )
    for index, file_name, parsed_overrides in stream_usable_prompt_overrides(
        file_names, load_png_parsed, args.prompt_count, skip_first=args.prompt_count_skip_first
    ):
        try:
            if parsed_overrides is None:
                logger.error(f"[{index + 1}/{len(file_names)}] {file_name}: no positive prompt in PNG metadata, skipping")
                continue

            source_image_path = os.path.join(folder, file_name)
            logger.info(f"[{index + 1}/{len(file_names)}] {file_name}: {parsed_overrides['prompt']}")

            if comparison_mode:
                # One prompt-only image and one all-metadata-settings image per iteration, both at the
                # same provided --seed (seed + iteration) so only the settings differ between them.
                prompt_only_variant = apply_image_embed_settings_gate(
                    parsed_overrides, prompts_only=True, ignore_negative_prompt=ignore_negative_prompt
                )
                all_settings_variant = apply_image_embed_settings_gate(
                    parsed_overrides, prompts_only=False, ignore_negative_prompt=ignore_negative_prompt
                )
                base_seed = resolve_random_seed(args.seed)
                for iteration in range(images_per_prompt):
                    seed_value = base_seed + iteration
                    render_one_image(prompt_only_variant, seed_value, "promptonly", source_image_path)
                    render_one_image(all_settings_variant, seed_value, "allsettings", source_image_path)
            else:
                gated_overrides = apply_image_embed_settings_gate(
                    parsed_overrides, prompts_only, ignore_negative_prompt
                )
                # Seed source: metadata Seed in settings mode (or 0), else the provided --seed.
                base_seed = resolve_random_seed(gated_overrides.get("seed", args.seed))
                for iteration in range(images_per_prompt):
                    render_one_image(gated_overrides, base_seed + iteration, None, source_image_path)
        except Exception as exc:
            logger.error(f"Error on PNG {file_name}: {exc}", exc_info=True)
            continue

    if args.output_type != "latent":
        vae.to("cpu")
    clean_memory_on_device(device)


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    device = torch.device(args.device)

    dit_weight_dtype = torch.bfloat16  # default
    if args.fp8_scaled:
        dit_weight_dtype = None  # various precision weights, so don't cast to specific dtype
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn

    logger.info(f"Using device: {device}, DiT weight weight precision: {dit_weight_dtype}")

    gen_settings = GenerationSettings(device=device, dit_weight_dtype=dit_weight_dtype)
    return gen_settings


def expand_lora_list_tokens_into_lora_args(args) -> None:
    """Populate args.lora_weight / args.lora_multiplier from the inline --lora_list token list.

    Tokens are a flat sequence like '<path> <multiplier> <path> <multiplier> ...'. A token that
    parses as a float is treated as the multiplier for the preceding path; any other token starts a
    new LoRA with a default multiplier of 1.0 (so '<path>' alone also works).
    """
    tokens = getattr(args, "lora_list", None)
    if not tokens:
        return

    lora_paths = []
    lora_multipliers = []
    for token in tokens:
        try:
            parsed_multiplier = float(token)
            is_multiplier = True
        except ValueError:
            is_multiplier = False

        if is_multiplier and lora_paths:
            lora_multipliers[-1] = parsed_multiplier
        elif is_multiplier and not lora_paths:
            logger.warning(f"Ignoring --lora_list multiplier '{token}' with no preceding LoRA path")
        else:
            lora_paths.append(token)
            lora_multipliers.append(1.0)

    if lora_paths:
        args.lora_weight = lora_paths
        args.lora_multiplier = lora_multipliers
        logger.info(f"Using {len(lora_paths)} LoRA(s) from --lora_list: {list(zip(lora_paths, lora_multipliers))}")


# An all-in-one civitai/ComfyUI "CheckpointSave" bundles the DiT, VAE, and text encoder into one
# safetensors under these prefixes. Each entry maps that bundled component to the split-file key layout
# the script's loaders expect (verified against the official split files: DiT keys carry a 'net.' prefix,
# VAE/text-encoder keys are bare). The text-encoder source prefix includes '.transformer.' so the extra
# 'cond_stage_model.qwen3_06b.logit_scale' key is naturally excluded.
COMBINED_CHECKPOINT_COMPONENTS = [
    {"name": "dit", "arg": "dit", "source_prefix": "model.diffusion_model.", "target_prefix": "net.", "filename": "dit.safetensors"},
    {"name": "vae", "arg": "vae", "source_prefix": "first_stage_model.", "target_prefix": "", "filename": "vae.safetensors"},
    {
        "name": "text_encoder",
        "arg": "text_encoder",
        "source_prefix": "cond_stage_model.qwen3_06b.transformer.",
        "target_prefix": "",
        "filename": "text_encoder.safetensors",
    },
]

COMBINED_CHECKPOINT_DETECT_PREFIXES = ("model.diffusion_model.", "first_stage_model.", "cond_stage_model.")


def detect_combined_checkpoint(keys) -> bool:
    """True when the given safetensors keys include all three combined-checkpoint component prefixes."""
    key_list = list(keys)
    return all(any(key.startswith(prefix) for key in key_list) for prefix in COMBINED_CHECKPOINT_DETECT_PREFIXES)


def derive_extracted_models_folder(dit_path: str) -> str:
    """Return the sibling folder (named for the model, next to it) where extracted components live."""
    return os.path.splitext(dit_path)[0]


def rename_combined_component_keys(all_keys, source_prefix: str, target_prefix: str) -> Dict[str, str]:
    """Return {new_key: original_key} for keys under source_prefix, remapped to target_prefix."""
    key_rename_map = {}
    for key in all_keys:
        if key.startswith(source_prefix):
            key_rename_map[target_prefix + key[len(source_prefix):]] = key
    return key_rename_map


def extract_combined_checkpoint_to_folder(combined_checkpoint_path: str, output_folder: str) -> Dict[str, str]:
    """Extract the bundled DiT/VAE/text-encoder from a combined checkpoint into split files in
    output_folder (one component at a time to limit memory). Returns {component_name: file_path}."""
    os.makedirs(output_folder, exist_ok=True)
    extracted_paths = {}
    with safe_open(combined_checkpoint_path, framework="pt") as checkpoint:
        all_keys = list(checkpoint.keys())
        for component in COMBINED_CHECKPOINT_COMPONENTS:
            key_rename_map = rename_combined_component_keys(
                all_keys, component["source_prefix"], component["target_prefix"]
            )
            component_state_dict = {
                new_key: checkpoint.get_tensor(original_key) for new_key, original_key in key_rename_map.items()
            }
            output_path = os.path.join(output_folder, component["filename"])
            logger.info(f"Extracting {component['name']} ({len(component_state_dict)} tensors) -> {output_path}")
            save_file(component_state_dict, output_path)
            del component_state_dict
            extracted_paths[component["name"]] = output_path
    return extracted_paths


def prepare_split_models_from_combined_checkpoint(args: argparse.Namespace) -> None:
    """If --dit is an all-in-one checkpoint (DiT+VAE+text encoder baked in), extract the three
    components once to a sibling folder named for the model (reused on later runs) and point
    args.dit/vae/text_encoder at them. Baked-in components take precedence over any explicitly
    provided --vae/--text_encoder (per user preference: easier to just pass the one checkpoint)."""
    if not args.dit or not os.path.isfile(args.dit):
        return

    with safe_open(args.dit, framework="pt") as checkpoint:
        checkpoint_keys = list(checkpoint.keys())
    if not detect_combined_checkpoint(checkpoint_keys):
        return

    output_folder = derive_extracted_models_folder(args.dit)
    expected_paths = {
        component["name"]: os.path.join(output_folder, component["filename"])
        for component in COMBINED_CHECKPOINT_COMPONENTS
    }
    if all(os.path.isfile(path) for path in expected_paths.values()):
        logger.info(f"Using previously extracted models from combined checkpoint in: {output_folder}")
        extracted_paths = expected_paths
    else:
        logger.info(f"Combined checkpoint detected. Extracting embedded DiT/VAE/text encoder to: {output_folder}")
        extracted_paths = extract_combined_checkpoint_to_folder(args.dit, output_folder)

    if (args.vae and args.vae != extracted_paths["vae"]) or (
        args.text_encoder and args.text_encoder != extracted_paths["text_encoder"]
    ):
        logger.info("Using the checkpoint's baked-in VAE/text encoder (overriding explicitly provided --vae/--text_encoder).")

    args.dit = extracted_paths["dit"]
    args.vae = extracted_paths["vae"]
    args.text_encoder = extracted_paths["text_encoder"]


def normalize_lora_test_folder_arg(args: argparse.Namespace) -> None:
    """Split the raw --lora_test_folder tokens into a folder path and a test-LoRA multiplier.

    Accepts '<folder>' or '<folder> <multiplier>'. After this runs, args.lora_test_folder is the folder
    path string (or None) and args.lora_test_multiplier is a float (default 1.0).
    """
    tokens = getattr(args, "lora_test_folder", None)
    args.lora_test_multiplier = 1.0

    if not tokens:
        args.lora_test_folder = None
        return

    if len(tokens) > 2:
        raise ValueError(f"--lora_test_folder expects '<folder> [multiplier]' (got: {tokens})")

    args.lora_test_folder = tokens[0]
    if len(tokens) == 2:
        try:
            args.lora_test_multiplier = float(tokens[1])
        except ValueError:
            raise ValueError(f"--lora_test_folder multiplier must be a number (got: {tokens[1]!r})")


def list_test_lora_paths(folder: str) -> List[str]:
    """Return sorted full paths of top-level .safetensors LoRA files in folder (subfolders ignored)."""
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(".safetensors") and os.path.isfile(os.path.join(folder, name))
    )


def read_lora_trigger_prompt_text(lora_path: str) -> str:
    """Return the text from the '<lora_basename>.txt' sidecar next to lora_path, or '' if absent."""
    trigger_text_path = os.path.splitext(lora_path)[0] + ".txt"
    if not os.path.isfile(trigger_text_path):
        return ""
    with open(trigger_text_path, "r", encoding="utf-8") as trigger_file:
        return trigger_file.read().strip()


def compose_pre_prompt_with_lora_injection(user_pre_prompt: str, lora_injection: str) -> str:
    """Combine the user's --pre_prompt with a test LoRA's trigger text (user first, then injection)."""
    return f"{user_pre_prompt.strip()} {lora_injection.strip()}".strip()


def build_args_for_test_lora(base_args: argparse.Namespace, test_lora_path: str) -> argparse.Namespace:
    """Return a deep-copied args with the test LoRA appended to the fixed LoRAs and its trigger text
    injected into the pre-prompt. base_args is not mutated."""
    test_args = copy.deepcopy(base_args)
    test_args.lora_test_folder = None  # prevent the sweep from recursing

    fixed_lora_weights = list(base_args.lora_weight) if base_args.lora_weight else []
    if isinstance(base_args.lora_multiplier, list):
        fixed_lora_multipliers = list(base_args.lora_multiplier)
    elif base_args.lora_multiplier is None:
        fixed_lora_multipliers = []
    else:
        fixed_lora_multipliers = [base_args.lora_multiplier]
    # Align multipliers to the fixed weights: pad short (default 1.0) and drop any extras (e.g. the
    # scalar default multiplier carried when there are no fixed LoRA weights).
    fixed_lora_multipliers = (fixed_lora_multipliers + [1.0] * len(fixed_lora_weights))[: len(fixed_lora_weights)]

    test_args.lora_weight = fixed_lora_weights + [test_lora_path]
    test_args.lora_multiplier = fixed_lora_multipliers + [base_args.lora_test_multiplier]

    lora_injection = read_lora_trigger_prompt_text(test_lora_path)
    composed_pre_prompt = compose_pre_prompt_with_lora_injection(base_args.pre_prompt, lora_injection)
    test_args.pre_prompt = composed_pre_prompt  # used by from_folder / from_image_embed
    test_args.lora_test_prompt_prefix = composed_pre_prompt  # used by from_file / single --prompt
    test_args.current_test_lora = f"{test_lora_path} {base_args.lora_test_multiplier}"
    return test_args


def build_png_generation_metadata_text(args) -> str:
    """Build the A1111-style 'parameters' text embedded in output PNGs: image generation data only.

    Deliberately excludes file paths (model/VAE/text-encoder/LoRA paths, source image, test LoRA) so
    the embedded metadata is portable. The format round-trips with this script's own --from_image_embed
    reader and standard A1111 tools.
    """
    image_height, image_width = args.image_size[0], args.image_size[1]
    settings_line = (
        f"Steps: {args.infer_steps}, "
        f"Sampler: {args.sampler}, "
        f"CFG scale: {args.guidance_scale}, "
        f"Seed: {args.seed}, "
        f"Size: {image_width}x{image_height}, "  # A1111 Size is WIDTHxHEIGHT
        f"Schedule type: {args.scheduler}, "
        f"Flow shift: {args.flow_shift}"
    )
    return f"{args.prompt}\nNegative prompt: {args.negative_prompt}\n{settings_line}"


def build_generation_settings_dict(args) -> dict:
    """Return the generation settings for one image as a structured dict, for the JSON sidecar and for
    reloading into the GUI. Optional keys (loras/source_image/test_lora) are included only when present."""
    image_height, image_width = args.image_size[0], args.image_size[1]
    settings = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": image_width,
        "height": image_height,
        "steps": args.infer_steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "seed": args.seed,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "dit": args.dit,
        "vae": args.vae,
        "text_encoder": args.text_encoder,
    }

    if getattr(args, "lora_weight", None):
        multipliers = args.lora_multiplier if isinstance(args.lora_multiplier, list) else [args.lora_multiplier]
        settings["loras"] = [
            {
                "path": lora_path,
                "multiplier": multipliers[lora_index] if lora_index < len(multipliers) else 1.0,
            }
            for lora_index, lora_path in enumerate(args.lora_weight)
        ]

    source_image_path = getattr(args, "current_source_image_path", None)
    if source_image_path:
        settings["source_image"] = source_image_path

    test_lora = getattr(args, "current_test_lora", None)
    if test_lora:
        settings["test_lora"] = test_lora

    return settings


def write_generation_settings_sidecar(save_path: str, image_name: str, args) -> None:
    """Write '<image_name>.json' next to the PNG recording the generation settings for reproducibility
    and for reloading into the GUI."""
    settings_path = os.path.join(save_path, f"{image_name}.json")
    with open(settings_path, "w", encoding="utf-8") as settings_file:
        json.dump(build_generation_settings_dict(args), settings_file, indent=2, ensure_ascii=False)
        settings_file.write("\n")
    logger.info(f"Settings saved to: {settings_path}")


def dispatch_generation(args: argparse.Namespace) -> None:
    """Run one full generation for the configured mode (from_folder / from_image_embed / from_file /
    interactive / single --prompt). Applies the test-LoRA prompt prefix to from_file / single --prompt
    (the modes that have no --pre_prompt of their own) when set by the LoRA test sweep.
    """
    # In the LoRA test sweep the composed prefix (which already folds in --pre_prompt) is set on
    # lora_test_prompt_prefix; otherwise fall back to the user's --pre_prompt for --from_file.
    prompt_prefix = getattr(args, "lora_test_prompt_prefix", "") or args.pre_prompt.strip()

    if args.from_folder:
        # Streaming mode: one caption file at a time (settings txt -> generate -> save -> next)
        process_folder_streaming(args)

    elif args.from_image_embed:
        # Streaming mode: one PNG at a time, prompts pulled from embedded A1111 metadata
        process_image_embed_streaming(args)

    elif args.from_file:
        # Batch mode from file
        with open(args.from_file, "r", encoding="utf-8") as f:
            prompt_lines = f.readlines()

        # prompts_data is already the usable list (blank/comment lines removed), so skip_first +
        # prompt_count paginate it directly by index.
        prompts_data = preprocess_prompts_for_batch(prompt_lines, args)
        apply_pre_prompt_to_batch_prompts(prompts_data, prompt_prefix, args.pre_prompt_neg)
        skip_first = max(0, args.prompt_count_skip_first)
        if args.prompt_count is not None:
            prompts_data = prompts_data[skip_first : skip_first + max(0, args.prompt_count)]
        else:
            prompts_data = prompts_data[skip_first:]
        process_batch_prompts(prompts_data, args)

    elif args.interactive:
        # Interactive mode
        process_interactive(args)

    else:
        # Single prompt mode. Route through the batch path so --images_per_prompt N renders N
        # seed-incremented images in ONE run (single model load), and the settings sidecar is written
        # before each generation.
        if prompt_prefix and args.prompt is not None:
            args.prompt = f"{prompt_prefix} {args.prompt}".strip()

        base_seed = resolve_random_seed(args.seed)
        prompts_data = build_repeated_single_prompt_data(args.prompt, base_seed, args.images_per_prompt)
        process_batch_prompts(prompts_data, args)


def run_lora_test_sweep(args: argparse.Namespace) -> None:
    """Run dispatch_generation once per top-level .safetensors in --lora_test_folder, each time with
    that test LoRA added on top of the fixed LoRAs. Models are reloaded for each test LoRA because
    LoRAs are merged into the DiT/text encoder at load time."""
    test_lora_paths = list_test_lora_paths(args.lora_test_folder)
    if not test_lora_paths:
        logger.warning(f"No .safetensors test LoRAs found in {args.lora_test_folder}")
        return

    logger.info(f"LoRA test sweep: {len(test_lora_paths)} test LoRA(s) from {args.lora_test_folder}")
    for index, test_lora_path in enumerate(test_lora_paths):
        logger.info(
            f"[test LoRA {index + 1}/{len(test_lora_paths)}] {test_lora_path} "
            f"(multiplier {args.lora_test_multiplier})"
        )
        test_args = build_args_for_test_lora(args, test_lora_path)
        dispatch_generation(test_args)


def main():
    # Parse arguments
    args = parse_args()
    expand_lora_list_tokens_into_lora_args(args)
    normalize_lora_test_folder_arg(args)

    # If --dit is an all-in-one checkpoint, extract/reuse its baked-in VAE + text encoder and repoint args.
    prepare_split_models_from_combined_checkpoint(args)

    if not args.text_encoder:
        raise ValueError(
            "No text encoder available: pass --text_encoder, or a --dit that is an all-in-one checkpoint "
            "with the text encoder baked in."
        )
    if not args.vae:
        raise ValueError(
            "No VAE available: pass --vae, or a --dit that is an all-in-one checkpoint with the VAE baked in."
        )

    # Check if latents are provided
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # Set device
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    args.device = device

    if latents_mode:
        # Original latent decode mode
        original_base_names = []
        latents_list = []
        seeds = []

        # assert len(args.latent_path) == 1, "Only one latent path is supported for now"

        for latent_path in args.latent_path:
            original_base_names.append(os.path.splitext(os.path.basename(latent_path))[0])
            seed = 0

            if os.path.splitext(latent_path)[1] != ".safetensors":
                latents = torch.load(latent_path, map_location="cpu")
            else:
                latents = load_file(latent_path)["latent"]
                with safe_open(latent_path, framework="pt") as f:
                    metadata = f.metadata()
                if metadata is None:
                    metadata = {}
                logger.info(f"Loaded metadata: {metadata}")

                if "seeds" in metadata:
                    seed = int(metadata["seeds"])
                if "height" in metadata and "width" in metadata:
                    height = int(metadata["height"])
                    width = int(metadata["width"])
                    args.image_size = [height, width]

            seeds.append(seed)
            logger.info(f"Loaded latent from {latent_path}. Shape: {latents.shape}")

            if latents.ndim == 5:  # [BCTHW]
                latents = latents.squeeze(0)  # [CTHW]

            latents_list.append(latents)

        vae = anima_train_utils.load_qwen_image_vae(args, device=device, disable_mmap=True)
        vae.to(torch.bfloat16)
        vae.eval()

        for i, latent in enumerate(latents_list):
            args.seed = seeds[i]
            save_output(args, vae, latent, device, original_base_names[i])

    else:
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.text_encoder, t5_tokenizer_path=None, qwen3_max_length=512, t5_max_length=512
        )
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

        encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
        strategy_base.TextEncodingStrategy.set_strategy(encoding_strategy)

        if args.lora_test_folder:
            # Run the whole configured generation once per test LoRA (models reload per test LoRA).
            run_lora_test_sweep(args)
        else:
            dispatch_generation(args)

    logger.info("Done!")


if __name__ == "__main__":
    main()
