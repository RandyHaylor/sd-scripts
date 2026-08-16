import argparse
import datetime
import gc
from importlib.util import find_spec
import random
import os
import time
import copy
from types import SimpleNamespace
from typing import Tuple, Optional, List, Any, Dict, Union

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

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


class GenerationSettings:
    def __init__(self, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None):
        self.device = device
        self.dit_weight_dtype = dit_weight_dtype  # not used currently because model may be optimized


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="HunyuanImage inference script")

    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path")
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
    parser.add_argument("--text_encoder", type=str, required=True, help="Text Encoder 1 (Qwen2.5-VL) directory or path")

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

    # inference
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5, help="Guidance scale for classifier free guidance. Default is 3.5."
    )
    parser.add_argument("--prompt", type=str, default=None, help="prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt for generation, default is empty string")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="image size, height and width")
    parser.add_argument("--infer_steps", type=int, default=50, help="number of inference steps, default is 50")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")

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
        default="euler",
        choices=["euler", "er_sde"],
        help="sampler: euler (deterministic flow Euler, default) or er_sde (stochastic ER-SDE-Solver-3, Anima's recommended sampler)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="default",
        choices=["default", "beta57"],
        help="sigma scheduler: default (flow-shifted linspace) or beta57 (RES4LYF beta alpha=0.5/beta=0.7, more low-noise emphasis)",
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
        "--count",
        type=int,
        default=None,
        help="Limit --from_folder / --from_file to the first N prompts (files sorted by name for --from_folder); default all",
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: read prompts from console")

    args = parser.parse_args()

    # Validate arguments
    if sum(bool(x) for x in (args.from_file, args.from_folder, args.interactive)) > 1:
        raise ValueError("Use only one of --from_file, --from_folder, or --interactive at the same time")

    if args.latent_path is None or len(args.latent_path) == 0:
        if args.prompt is None and not args.from_file and not args.from_folder and not args.interactive:
            raise ValueError("Either --prompt, --from_file, --from_folder or --interactive must be specified")

    if args.lycoris and not lycoris_available:
        raise ValueError("install lycoris: https://github.com/KohakuBlueleaf/LyCORIS")

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    return args


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
            # lora_sd = filter_lora_state_dict(lora_sd, args.include_patterns, args.exclude_patterns)
            lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}  # only keep unet lora weights
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
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
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

    if args.sampler == "er_sde":
        def predict_denoised_x0(current_latents, sigma_scalar):
            velocity = run_velocity_with_cfg(current_latents, sigma_scalar).to(torch.float32)
            return current_latents.to(torch.float32) - sigma_scalar.to(torch.float32) * velocity

        latents = anima_er_sde_sampling.sample_er_sde_rectified_flow(
            predict_denoised_x0, latents, sigmas, seed=args.seed
        ).to(latents.dtype)
    else:
        with tqdm(total=len(sigmas) - 1, desc="Denoising steps") as pbar:
            for i in range(len(sigmas) - 1):
                noise_pred = run_velocity_with_cfg(latents, sigmas[i])
                latents = hunyuan_image_utils.step(latents, noise_pred, sigmas, i).to(latents.dtype)
                pbar.update()

    return latents


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
    image.save(os.path.join(save_path, f"{image_name}.png"))

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

    For each top-level .txt file (sorted by name; --count limits the total), build the prompt
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
    if args.count is not None:
        file_names = file_names[: max(0, args.count)]

    pre_prompt = args.pre_prompt.strip()
    pre_prompt_neg = args.pre_prompt_neg.strip()
    settings = args.from_folder_settings.strip()
    os.makedirs(args.save_path, exist_ok=True)

    logger.info(f"Streaming {len(file_names)} caption file(s) from {folder}")
    for index, file_name in enumerate(file_names):
        try:
            with open(os.path.join(folder, file_name), "r", encoding="utf-8") as caption_file:
                caption = caption_file.read().strip()
            prompt_body = f"{pre_prompt} {caption}".strip() if pre_prompt else caption
            prompt_line = f"{prompt_body} {settings}".strip() if settings else prompt_body

            prompt_args = apply_overrides(args, parse_prompt_line(prompt_line))
            if prompt_args.seed is None:
                prompt_args.seed = random.randint(0, 2**32 - 1)
            if pre_prompt_neg:
                prompt_args.negative_prompt = pre_prompt_neg

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


def write_generation_settings_sidecar(save_path: str, image_name: str, args) -> None:
    """Write '<image_name>.txt' next to the PNG recording the generation settings for reproducibility."""
    image_height, image_width = args.image_size[0], args.image_size[1]
    settings_lines = [
        f"prompt: {args.prompt}",
        f"negative_prompt: {args.negative_prompt}",
        f"width: {image_width}",
        f"height: {image_height}",
        f"steps: {args.infer_steps}",
        f"guidance_scale: {args.guidance_scale}",
        f"flow_shift: {args.flow_shift}",
        f"seed: {args.seed}",
        f"sampler: {args.sampler}",
        f"scheduler: {args.scheduler}",
        f"dit: {args.dit}",
        f"vae: {args.vae}",
        f"text_encoder: {args.text_encoder}",
    ]
    if getattr(args, "lora_weight", None):
        multipliers = args.lora_multiplier if isinstance(args.lora_multiplier, list) else [args.lora_multiplier]
        for lora_index, lora_path in enumerate(args.lora_weight):
            multiplier = multipliers[lora_index] if lora_index < len(multipliers) else 1.0
            settings_lines.append(f"lora: {lora_path} {multiplier}")

    settings_path = os.path.join(save_path, f"{image_name}.txt")
    with open(settings_path, "w", encoding="utf-8") as settings_file:
        settings_file.write("\n".join(settings_lines) + "\n")
    logger.info(f"Settings saved to: {settings_path}")


def main():
    # Parse arguments
    args = parse_args()
    expand_lora_list_tokens_into_lora_args(args)

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

        if args.from_folder:
            # Streaming mode: one caption file at a time (settings txt -> generate -> save -> next)
            process_folder_streaming(args)

        elif args.from_file:
            # Batch mode from file

            # Read prompts from file
            with open(args.from_file, "r", encoding="utf-8") as f:
                prompt_lines = f.readlines()

            # Process prompts
            prompts_data = preprocess_prompts_for_batch(prompt_lines, args)
            if args.count is not None:
                prompts_data = prompts_data[: max(0, args.count)]
            process_batch_prompts(prompts_data, args)

        elif args.interactive:
            # Interactive mode
            process_interactive(args)

        else:
            # Single prompt mode (original behavior)

            # Generate latent
            gen_settings = get_generation_settings(args)

            # For single mode, precomputed data is None, shared_models is None.
            # generate will load all necessary models (Text Encoders, DiT).
            latent = generate(args, gen_settings)

            clean_memory_on_device(device)

            # Save latent and video
            vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
            vae.to(torch.bfloat16)
            vae.eval()
            save_output(args, vae, latent, device)

    logger.info("Done!")


if __name__ == "__main__":
    main()
