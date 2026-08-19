import argparse
import json
import os

import torch
from PIL import Image, PngImagePlugin

from library import anima_er_sde_sampling

from anima_minimal_inference import (
    SAMPLER_OPTION_CHOICES,
    SCHEDULER_OPTION_CHOICES,
    apply_image_embed_settings_gate,
    apply_pre_prompt_to_batch_prompts,
    build_args_for_test_lora,
    build_generation_settings_dict,
    build_repeated_single_prompt_data,
    convert_peft_diffusion_model_lora_keys,
    select_dit_lora_state_dict,
    build_png_generation_metadata_text,
    COMBINED_CHECKPOINT_COMPONENTS,
    compose_pre_prompt_with_lora_injection,
    decode_exif_user_comment_bytes,
    derive_extracted_models_folder,
    detect_combined_checkpoint,
    list_test_lora_paths,
    rename_combined_component_keys,
    map_metadata_value_to_script_option,
    normalize_from_image_embed_arg,
    normalize_lora_test_folder_arg,
    parse_a1111_png_prompt_metadata,
    parse_comfyui_prompt_metadata,
    read_lora_trigger_prompt_text,
    read_png_parsed_metadata,
    read_png_prompt_overrides,
    resolve_random_seed,
    hold_exclusive_cross_process_file_lock,
    serialize_model_file_disk_reads,
    serialize_model_loading_phase,
    serialize_gpu_compute,
    gpu_phase_is_covered_by_scope,
    GPU_LOCK_SCOPE_DENOISE_ONLY,
    GPU_LOCK_SCOPE_ALL_COMPUTE,
    GPU_PHASE_TEXT_ENCODE,
    GPU_PHASE_DENOISE,
    GPU_PHASE_VAE_DECODE,
    stream_usable_prompt_overrides,
)

A1111_PARAMETERS_WITH_NEGATIVE_AND_SETTINGS = (
    "1girl, solo, sorceress, hat, black hair,\n"
    "Negative prompt: lazyneg, patreon logo, bad hands,\n"
    "Steps: 20, Sampler: Euler a, CFG scale: 5.0, Seed: 614115846627341, Size: 1664x2432, Clip skip: 2"
)


def test_parse_a1111_metadata_extracts_prompts_and_settings():
    overrides = parse_a1111_png_prompt_metadata(A1111_PARAMETERS_WITH_NEGATIVE_AND_SETTINGS)

    assert overrides["prompt"] == "1girl, solo, sorceress, hat, black hair,"
    assert overrides["negative_prompt"] == "lazyneg, patreon logo, bad hands,"
    assert overrides["infer_steps"] == 20
    assert overrides["guidance_scale"] == 5.0
    assert overrides["seed"] == 614115846627341
    # A1111 Size is WIDTHxHEIGHT; this script stores height/width separately.
    assert overrides["image_size_width"] == 1664
    assert overrides["image_size_height"] == 2432


def test_parse_a1111_metadata_without_negative_or_settings():
    overrides = parse_a1111_png_prompt_metadata("just a positive prompt")

    assert overrides == {"prompt": "just a positive prompt"}


def test_parse_a1111_metadata_multiline_negative():
    parameters = (
        "positive line one, positive line two\n"
        "Negative prompt: neg line one,\n"
        "neg line two\n"
        "Steps: 30, CFG scale: 7.5"
    )
    overrides = parse_a1111_png_prompt_metadata(parameters)

    assert overrides["prompt"] == "positive line one, positive line two"
    assert overrides["negative_prompt"] == "neg line one,\nneg line two"
    assert overrides["infer_steps"] == 30
    assert overrides["guidance_scale"] == 7.5


def test_parse_empty_metadata_returns_empty():
    assert parse_a1111_png_prompt_metadata("") == {}
    assert parse_a1111_png_prompt_metadata("   ") == {}


def _write_png_with_parameters(path, parameters_text):
    image = Image.new("RGB", (8, 8), color=(10, 20, 30))
    png_info = PngImagePlugin.PngInfo()
    if parameters_text is not None:
        png_info.add_text("parameters", parameters_text)
    image.save(path, pnginfo=png_info)


def test_read_png_prompt_overrides_full(tmp_path):
    png_path = os.path.join(tmp_path, "sample.png")
    _write_png_with_parameters(png_path, A1111_PARAMETERS_WITH_NEGATIVE_AND_SETTINGS)

    overrides = read_png_prompt_overrides(png_path, prompts_only=False)

    assert overrides["prompt"] == "1girl, solo, sorceress, hat, black hair,"
    assert overrides["negative_prompt"] == "lazyneg, patreon logo, bad hands,"
    assert overrides["infer_steps"] == 20
    assert overrides["image_size_width"] == 1664


def test_read_png_prompt_overrides_prompts_only_drops_settings(tmp_path):
    png_path = os.path.join(tmp_path, "sample.png")
    _write_png_with_parameters(png_path, A1111_PARAMETERS_WITH_NEGATIVE_AND_SETTINGS)

    overrides = read_png_prompt_overrides(png_path, prompts_only=True)

    assert set(overrides.keys()) == {"prompt", "negative_prompt"}
    assert overrides["prompt"] == "1girl, solo, sorceress, hat, black hair,"
    assert overrides["negative_prompt"] == "lazyneg, patreon logo, bad hands,"


def test_read_png_prompt_overrides_without_metadata_returns_none(tmp_path):
    png_path = os.path.join(tmp_path, "no_meta.png")
    _write_png_with_parameters(png_path, None)

    assert read_png_prompt_overrides(png_path, prompts_only=False) is None


def test_read_png_prompt_overrides_without_positive_returns_none(tmp_path):
    png_path = os.path.join(tmp_path, "no_positive.png")
    _write_png_with_parameters(png_path, "Negative prompt: only a negative\nSteps: 10, CFG scale: 4.0")

    assert read_png_prompt_overrides(png_path, prompts_only=False) is None


def test_settings_gate_requires_steps_and_cfg():
    # Both Steps and CFG present -> settings mode: all found settings kept.
    overrides = {
        "prompt": "p",
        "negative_prompt": "n",
        "infer_steps": 20,
        "guidance_scale": 5.0,
        "seed": 123,
        "image_size_width": 1024,
        "image_size_height": 1536,
    }
    gated = apply_image_embed_settings_gate(overrides, prompts_only=False)
    assert gated == overrides


def test_settings_gate_missing_cfg_reverts_to_prompts_only():
    overrides = {"prompt": "p", "negative_prompt": "n", "infer_steps": 20}
    gated = apply_image_embed_settings_gate(overrides, prompts_only=False)
    assert gated == {"prompt": "p", "negative_prompt": "n"}


def test_settings_gate_missing_seed_defaults_to_zero():
    # Steps + CFG present, no Seed and no Size: seed defaults to 0, size left to CLI (absent).
    overrides = {"prompt": "p", "infer_steps": 20, "guidance_scale": 5.0}
    gated = apply_image_embed_settings_gate(overrides, prompts_only=False)
    assert gated["infer_steps"] == 20
    assert gated["guidance_scale"] == 5.0
    assert gated["seed"] == 0
    assert "image_size_width" not in gated


def test_settings_gate_prompts_only_drops_settings():
    overrides = {"prompt": "p", "negative_prompt": "n", "infer_steps": 20, "guidance_scale": 5.0, "seed": 9}
    gated = apply_image_embed_settings_gate(overrides, prompts_only=True)
    assert gated == {"prompt": "p", "negative_prompt": "n"}


def test_settings_gate_ignore_negative_drops_negative_but_keeps_settings():
    overrides = {"prompt": "p", "negative_prompt": "n", "infer_steps": 20, "guidance_scale": 5.0, "seed": 9}
    gated = apply_image_embed_settings_gate(overrides, prompts_only=False, ignore_negative_prompt=True)
    assert "negative_prompt" not in gated
    assert gated["prompt"] == "p"
    assert gated["infer_steps"] == 20
    assert gated["seed"] == 9


def test_settings_gate_ignore_negative_with_prompts_only():
    overrides = {"prompt": "p", "negative_prompt": "n", "infer_steps": 20, "guidance_scale": 5.0}
    gated = apply_image_embed_settings_gate(overrides, prompts_only=True, ignore_negative_prompt=True)
    assert gated == {"prompt": "p"}


def test_read_png_prompt_overrides_ignore_negative(tmp_path):
    png_path = os.path.join(tmp_path, "sample.png")
    _write_png_with_parameters(png_path, A1111_PARAMETERS_WITH_NEGATIVE_AND_SETTINGS)

    overrides = read_png_prompt_overrides(png_path, prompts_only=False, ignore_negative_prompt=True)

    assert "negative_prompt" not in overrides
    assert overrides["prompt"] == "1girl, solo, sorceress, hat, black hair,"


def test_resolve_random_seed_fixed_value_unchanged():
    assert resolve_random_seed(42) == 42
    assert resolve_random_seed(0) == 0


def test_resolve_random_seed_none_is_random_in_range():
    seed = resolve_random_seed(None)
    assert 0 <= seed <= 2**32 - 1


def test_resolve_random_seed_minus_one_is_random_in_range():
    seed = resolve_random_seed(-1)
    assert 0 <= seed <= 2**32 - 1


def _collect_streamed(items, usable_map, prompt_count, skip_first=0):
    """Drive stream_usable_prompt_overrides with a dict-backed loader (None => unusable)."""
    return list(
        stream_usable_prompt_overrides(items, lambda name: usable_map[name], prompt_count, skip_first=skip_first)
    )


def test_stream_usable_no_limit_yields_all_including_skips():
    items = ["a", "b", "c"]
    usable_map = {"a": {"prompt": "pa"}, "b": None, "c": {"prompt": "pc"}}
    streamed = _collect_streamed(items, usable_map, prompt_count=None)
    assert streamed == [
        (0, "a", {"prompt": "pa"}),
        (1, "b", None),
        (2, "c", {"prompt": "pc"}),
    ]


def test_stream_usable_limit_counts_only_usable_skips_do_not_count():
    # Two unusable items sit between usable ones; a limit of 2 must still reach 2 usable outputs.
    items = ["a", "skip1", "b", "skip2", "c"]
    usable_map = {
        "a": {"prompt": "pa"},
        "skip1": None,
        "b": {"prompt": "pb"},
        "skip2": None,
        "c": {"prompt": "pc"},
    }
    streamed = _collect_streamed(items, usable_map, prompt_count=2)
    # 'a' (usable #1), 'skip1' (skipped, visited), 'b' (usable #2) -> stops before skip2/c.
    assert streamed == [
        (0, "a", {"prompt": "pa"}),
        (1, "skip1", None),
        (2, "b", {"prompt": "pb"}),
    ]


def test_stream_usable_leading_skips_still_reach_limit():
    items = ["skip1", "skip2", "a", "b"]
    usable_map = {"skip1": None, "skip2": None, "a": {"prompt": "pa"}, "b": {"prompt": "pb"}}
    streamed = _collect_streamed(items, usable_map, prompt_count=1)
    assert streamed == [
        (0, "skip1", None),
        (1, "skip2", None),
        (2, "a", {"prompt": "pa"}),
    ]


def test_stream_usable_zero_limit_yields_nothing():
    items = ["a", "b"]
    usable_map = {"a": {"prompt": "pa"}, "b": {"prompt": "pb"}}
    assert _collect_streamed(items, usable_map, prompt_count=0) == []


def test_stream_usable_skip_first_skips_usable_only():
    items = ["a", "b", "c", "d"]
    usable_map = {"a": {"prompt": "pa"}, "b": {"prompt": "pb"}, "c": {"prompt": "pc"}, "d": {"prompt": "pd"}}
    streamed = _collect_streamed(items, usable_map, prompt_count=None, skip_first=2)
    # First two usable ('a','b') skipped and NOT yielded; remaining yielded.
    assert streamed == [(2, "c", {"prompt": "pc"}), (3, "d", {"prompt": "pd"})]


def test_stream_usable_skip_first_ignores_unusable_in_skip_region():
    items = ["skip1", "a", "skip2", "b", "c"]
    usable_map = {
        "skip1": None,
        "a": {"prompt": "pa"},
        "skip2": None,
        "b": {"prompt": "pb"},
        "c": {"prompt": "pc"},
    }
    # skip_first=2 must skip the first TWO usable ('a','b'); unusable items in the skip region are
    # passed over silently (not yielded). Yielding resumes at 'c'.
    streamed = _collect_streamed(items, usable_map, prompt_count=None, skip_first=2)
    assert streamed == [(4, "c", {"prompt": "pc"})]


def test_stream_usable_skip_first_then_limit_paginates():
    # "do 4, then next 4": skip_first=2 + prompt_count=2 yields the 3rd and 4th usable.
    items = ["a", "b", "c", "d", "e"]
    usable_map = {k: {"prompt": k} for k in items}
    streamed = _collect_streamed(items, usable_map, prompt_count=2, skip_first=2)
    assert streamed == [(2, "c", {"prompt": "c"}), (3, "d", {"prompt": "d"})]


def test_convert_peft_diffusion_model_lora_keys():
    lora_sd = {
        "diffusion_model.blocks.0.adaln_modulation_cross_attn.1.lora_A.weight": "A",
        "diffusion_model.blocks.0.adaln_modulation_cross_attn.1.lora_B.weight": "B",
        "diffusion_model.blocks.0.adaln_modulation_cross_attn.1.alpha": "alpha",
    }
    converted = convert_peft_diffusion_model_lora_keys(lora_sd)
    assert converted == {
        "blocks_0_adaln_modulation_cross_attn_1.lora_down.weight": "A",  # lora_A -> lora_down
        "blocks_0_adaln_modulation_cross_attn_1.lora_up.weight": "B",  # lora_B -> lora_up
        "blocks_0_adaln_modulation_cross_attn_1.alpha": "alpha",
    }


def test_convert_peft_ignores_non_diffusion_model_keys():
    assert convert_peft_diffusion_model_lora_keys({"lora_unet_blocks_0.lora_down.weight": "x"}) == {}


def test_select_dit_lora_prefers_kohya_unet_keys():
    lora_sd = {
        "lora_unet_blocks_0_cross_attn_k_proj.lora_down.weight": "d",
        "lora_unet_blocks_0_cross_attn_k_proj.lora_up.weight": "u",
        "lora_te_something.lora_down.weight": "te",  # text-encoder keys are not kept for the DiT
    }
    selected = select_dit_lora_state_dict(lora_sd)
    assert selected == {
        "lora_unet_blocks_0_cross_attn_k_proj.lora_down.weight": "d",
        "lora_unet_blocks_0_cross_attn_k_proj.lora_up.weight": "u",
    }


def test_select_dit_lora_converts_peft_when_no_unet_keys():
    lora_sd = {
        "diffusion_model.blocks.0.cross_attn.q_proj.lora_A.weight": "A",
        "diffusion_model.blocks.0.cross_attn.q_proj.lora_B.weight": "B",
    }
    selected = select_dit_lora_state_dict(lora_sd)
    assert selected == {
        "blocks_0_cross_attn_q_proj.lora_down.weight": "A",
        "blocks_0_cross_attn_q_proj.lora_up.weight": "B",
    }


def test_build_repeated_single_prompt_data_increments_seed():
    data = build_repeated_single_prompt_data("a cat", 42, 3)
    assert data == [
        {"prompt": "a cat", "seed": 42},
        {"prompt": "a cat", "seed": 43},
        {"prompt": "a cat", "seed": 44},
    ]


def test_build_repeated_single_prompt_data_minimum_one():
    assert build_repeated_single_prompt_data("p", 7, 0) == [{"prompt": "p", "seed": 7}]


def test_apply_pre_prompt_to_batch_prompts():
    prompts_data = [
        {"prompt": "a cat"},
        {"prompt": "a dog", "negative_prompt": "leash"},  # keeps its own negative
    ]
    apply_pre_prompt_to_batch_prompts(prompts_data, "masterpiece", "worst quality")
    assert prompts_data[0]["prompt"] == "masterpiece a cat"
    assert prompts_data[0]["negative_prompt"] == "worst quality"  # line had none -> pre_prompt_neg used
    assert prompts_data[1]["prompt"] == "masterpiece a dog"
    assert prompts_data[1]["negative_prompt"] == "leash"  # per-line negative preserved


def test_apply_pre_prompt_empty_is_noop():
    prompts_data = [{"prompt": "a cat"}]
    apply_pre_prompt_to_batch_prompts(prompts_data, "", "")
    assert prompts_data[0] == {"prompt": "a cat"}


def test_sampler_and_scheduler_choices_include_new_options():
    assert "euler_ancestral" in SAMPLER_OPTION_CHOICES
    assert "simple" in SCHEDULER_OPTION_CHOICES


def test_build_simple_sigmas_descends_to_zero():
    sigmas = anima_er_sde_sampling.build_simple_sigmas(10, 1.0, torch.device("cpu"))
    assert len(sigmas) == 11  # steps + 1
    assert float(sigmas[-1]) == 0.0
    # strictly descending down to the trailing 0.0
    values = [float(s) for s in sigmas]
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1)), values
    # first sigma is near the max of the flow table (~1.0 at flow_shift 1.0)
    assert 0.9 <= values[0] <= 1.0


def test_build_simple_sigmas_varies_with_flow_shift():
    a = anima_er_sde_sampling.build_simple_sigmas(20, 1.0, torch.device("cpu"))
    b = anima_er_sde_sampling.build_simple_sigmas(20, 5.0, torch.device("cpu"))
    assert not torch.equal(a, b)


def test_euler_ancestral_identity_denoiser_no_eta_returns_input():
    # denoised == x means zero velocity; with eta=0 no noise is added, so the latent is unchanged.
    latents = torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 2, 3)
    sigmas = torch.tensor([0.8, 0.5, 0.2, 0.0], dtype=torch.float32)

    def identity_denoiser(current_latents, _sigma_scalar):
        return current_latents

    out = anima_er_sde_sampling.sample_euler_ancestral_rectified_flow(
        identity_denoiser, latents.clone(), sigmas, seed=0, eta=0.0
    )
    assert torch.allclose(out, latents, atol=1e-5), out


def test_euler_ancestral_seed_is_reproducible_and_seed_sensitive():
    # Identity denoiser keeps the per-step ancestral noise in x through the terminal step, so the
    # output reflects the seeded noise (a zero/constant denoiser would be zeroed by the final x=denoised).
    latents = torch.zeros(1, 1, 1, 2, 3, dtype=torch.float32)
    sigmas = torch.tensor([0.9, 0.6, 0.3, 0.0], dtype=torch.float32)

    def identity_denoiser(current_latents, _sigma_scalar):
        return current_latents

    run_a = anima_er_sde_sampling.sample_euler_ancestral_rectified_flow(identity_denoiser, latents.clone(), sigmas, seed=42, eta=1.0)
    run_b = anima_er_sde_sampling.sample_euler_ancestral_rectified_flow(identity_denoiser, latents.clone(), sigmas, seed=42, eta=1.0)
    run_c = anima_er_sde_sampling.sample_euler_ancestral_rectified_flow(identity_denoiser, latents.clone(), sigmas, seed=99, eta=1.0)
    assert torch.equal(run_a, run_b)  # same seed -> identical
    assert not torch.equal(run_a, run_c)  # different seed -> different


def test_map_metadata_value_to_script_option_matches_and_normalizes():
    assert map_metadata_value_to_script_option("er_sde", SAMPLER_OPTION_CHOICES) == "er_sde"
    assert map_metadata_value_to_script_option("Euler", SAMPLER_OPTION_CHOICES) == "euler"
    # A1111 spellings that are NOT one of our choices must not map.
    assert map_metadata_value_to_script_option("Euler a", SAMPLER_OPTION_CHOICES) is None
    assert map_metadata_value_to_script_option("DPM++ 2M", SAMPLER_OPTION_CHOICES) is None
    assert map_metadata_value_to_script_option("beta57", SCHEDULER_OPTION_CHOICES) == "beta57"
    assert map_metadata_value_to_script_option("Karras", SCHEDULER_OPTION_CHOICES) is None
    assert map_metadata_value_to_script_option("", SAMPLER_OPTION_CHOICES) is None
    assert map_metadata_value_to_script_option(None, SAMPLER_OPTION_CHOICES) is None


def test_parse_metadata_maps_recognized_sampler_and_scheduler():
    parameters = "a prompt\nSteps: 20, Sampler: er_sde, CFG scale: 5.0, Schedule type: beta57"
    overrides = parse_a1111_png_prompt_metadata(parameters)
    assert overrides["sampler"] == "er_sde"
    assert overrides["scheduler"] == "beta57"


def test_parse_metadata_omits_unmapped_sampler_and_scheduler():
    parameters = "a prompt\nSteps: 20, Sampler: Euler a, CFG scale: 5.0, Schedule type: Karras"
    overrides = parse_a1111_png_prompt_metadata(parameters)
    assert "sampler" not in overrides
    assert "scheduler" not in overrides


def test_settings_gate_keeps_sampler_scheduler_in_settings_mode_drops_in_prompts_only():
    overrides = {
        "prompt": "p",
        "infer_steps": 20,
        "guidance_scale": 5.0,
        "sampler": "er_sde",
        "scheduler": "beta57",
    }
    settings_mode = apply_image_embed_settings_gate(overrides, prompts_only=False)
    assert settings_mode["sampler"] == "er_sde"
    assert settings_mode["scheduler"] == "beta57"

    prompts_only = apply_image_embed_settings_gate(overrides, prompts_only=True)
    assert "sampler" not in prompts_only
    assert "scheduler" not in prompts_only


def test_read_png_parsed_metadata_is_ungated(tmp_path):
    png_path = os.path.join(tmp_path, "sample.png")
    _write_png_with_parameters(png_path, "a prompt\nNegative prompt: neg\nSteps: 20, Sampler: er_sde, CFG scale: 5.0")
    parsed = read_png_parsed_metadata(png_path)
    assert parsed["prompt"] == "a prompt"
    assert parsed["negative_prompt"] == "neg"
    assert parsed["infer_steps"] == 20
    assert parsed["sampler"] == "er_sde"


def test_read_png_parsed_metadata_none_without_positive(tmp_path):
    png_path = os.path.join(tmp_path, "no_pos.png")
    _write_png_with_parameters(png_path, "Negative prompt: only neg\nSteps: 20, CFG scale: 5.0")
    assert read_png_parsed_metadata(png_path) is None


COMFYUI_PROMPT_GRAPH = json.dumps(
    {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 123,
                "steps": 45,
                "cfg": 9.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a fairy, masterpiece"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, bad hands"}},
    }
)


def test_parse_comfyui_prompt_metadata_follows_sampler_links():
    overrides = parse_comfyui_prompt_metadata(COMFYUI_PROMPT_GRAPH)
    assert overrides["prompt"] == "a fairy, masterpiece"
    assert overrides["negative_prompt"] == "blurry, bad hands"
    assert overrides["infer_steps"] == 45
    assert overrides["guidance_scale"] == 9.0
    assert overrides["seed"] == 123
    assert overrides["image_size_width"] == 832
    assert overrides["image_size_height"] == 1216
    # "euler" maps to a script sampler; "normal" scheduler does not map and is omitted.
    assert overrides["sampler"] == "euler"
    assert "scheduler" not in overrides


def test_parse_comfyui_prompt_metadata_no_sampler_node_returns_empty():
    graph = json.dumps({"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "orphan"}}})
    assert parse_comfyui_prompt_metadata(graph) == {}


def test_parse_comfyui_prompt_metadata_invalid_json_returns_empty():
    assert parse_comfyui_prompt_metadata("not json {{{") == {}


def _write_png_with_comfyui_prompt(path, prompt_json_text):
    image = Image.new("RGB", (8, 8), color=(1, 2, 3))
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("prompt", prompt_json_text)
    image.save(path, pnginfo=png_info)


def test_read_png_parsed_metadata_reads_comfyui_when_no_a1111(tmp_path):
    png_path = os.path.join(tmp_path, "comfy.png")
    _write_png_with_comfyui_prompt(png_path, COMFYUI_PROMPT_GRAPH)
    parsed = read_png_parsed_metadata(png_path)
    assert parsed["prompt"] == "a fairy, masterpiece"
    assert parsed["negative_prompt"] == "blurry, bad hands"
    assert parsed["seed"] == 123


def test_read_png_parsed_metadata_prefers_a1111_over_comfyui(tmp_path):
    png_path = os.path.join(tmp_path, "both.png")
    image = Image.new("RGB", (8, 8), color=(1, 2, 3))
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("parameters", "a1111 prompt\nSteps: 10, CFG scale: 4.0")
    png_info.add_text("prompt", COMFYUI_PROMPT_GRAPH)
    image.save(png_path, pnginfo=png_info)
    parsed = read_png_parsed_metadata(png_path)
    assert parsed["prompt"] == "a1111 prompt"


def test_decode_exif_user_comment_unicode_little_endian():
    raw = b"UNICODE\x00" + "a fairy prompt".encode("utf-16-le")
    assert decode_exif_user_comment_bytes(raw) == "a fairy prompt"


def test_decode_exif_user_comment_unicode_big_endian():
    raw = b"UNICODE\x00" + "a fairy prompt".encode("utf-16-be")
    assert decode_exif_user_comment_bytes(raw) == "a fairy prompt"


def test_decode_exif_user_comment_ascii_and_plain_str():
    assert decode_exif_user_comment_bytes(b"ASCII\x00\x00\x00hello world") == "hello world"
    assert decode_exif_user_comment_bytes("already text") == "already text"
    assert decode_exif_user_comment_bytes(b"") == ""


def test_normalize_from_image_embed_comparison_keyword():
    args = argparse.Namespace(from_image_embed=["prompt_only_and_all_settings", "/some/folder"])
    normalize_from_image_embed_arg(args)
    assert args.from_image_embed == "/some/folder"
    assert args.from_image_embed_prompt_only_and_all_settings is True
    assert args.from_image_embed_prompts_only is False


def test_normalize_from_image_embed_comparison_combined_with_ignore_negative():
    args = argparse.Namespace(
        from_image_embed=["prompt_only_and_all_settings", "ignore_negative_prompt", "/some/folder"]
    )
    normalize_from_image_embed_arg(args)
    assert args.from_image_embed == "/some/folder"
    assert args.from_image_embed_prompt_only_and_all_settings is True
    assert args.from_image_embed_ignore_negative_prompt is True


def test_normalize_lora_test_folder_defaults_multiplier():
    args = argparse.Namespace(lora_test_folder=["/loras/test"])
    normalize_lora_test_folder_arg(args)
    assert args.lora_test_folder == "/loras/test"
    assert args.lora_test_multiplier == 1.0


def test_normalize_lora_test_folder_with_multiplier():
    args = argparse.Namespace(lora_test_folder=["/loras/test", "0.8"])
    normalize_lora_test_folder_arg(args)
    assert args.lora_test_folder == "/loras/test"
    assert args.lora_test_multiplier == 0.8


def test_normalize_lora_test_folder_none():
    args = argparse.Namespace(lora_test_folder=None)
    normalize_lora_test_folder_arg(args)
    assert args.lora_test_folder is None
    assert args.lora_test_multiplier == 1.0


def test_normalize_lora_test_folder_rejects_non_float_multiplier():
    args = argparse.Namespace(lora_test_folder=["/loras/test", "strong"])
    try:
        normalize_lora_test_folder_arg(args)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-float multiplier")


def test_list_test_lora_paths_top_level_safetensors_sorted(tmp_path):
    (tmp_path / "b_lora.safetensors").write_text("x")
    (tmp_path / "a_lora.safetensors").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "ignored.ckpt").write_text("x")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "deep_lora.safetensors").write_text("x")

    result = list_test_lora_paths(str(tmp_path))
    assert result == [
        os.path.join(str(tmp_path), "a_lora.safetensors"),
        os.path.join(str(tmp_path), "b_lora.safetensors"),
    ]


def test_read_lora_trigger_prompt_text_present(tmp_path):
    lora_path = os.path.join(tmp_path, "my_lora.safetensors")
    open(lora_path, "w").close()
    with open(os.path.join(tmp_path, "my_lora.txt"), "w") as sidecar:
        sidecar.write("  triggerword, style tag  \n")
    assert read_lora_trigger_prompt_text(lora_path) == "triggerword, style tag"


def test_read_lora_trigger_prompt_text_absent(tmp_path):
    lora_path = os.path.join(tmp_path, "no_sidecar.safetensors")
    open(lora_path, "w").close()
    assert read_lora_trigger_prompt_text(lora_path) == ""


def test_compose_pre_prompt_with_lora_injection_ordering():
    assert compose_pre_prompt_with_lora_injection("userpre", "loratrigger") == "userpre loratrigger"
    assert compose_pre_prompt_with_lora_injection("", "loratrigger") == "loratrigger"
    assert compose_pre_prompt_with_lora_injection("userpre", "") == "userpre"
    assert compose_pre_prompt_with_lora_injection("", "") == ""


def test_build_args_for_test_lora_appends_lora_and_injects(tmp_path):
    lora_path = os.path.join(tmp_path, "test_lora.safetensors")
    open(lora_path, "w").close()
    with open(os.path.join(tmp_path, "test_lora.txt"), "w") as sidecar:
        sidecar.write("loratrigger")

    base_args = argparse.Namespace(
        lora_weight=["/fixed/a.safetensors"],
        lora_multiplier=[1.0],
        pre_prompt="userpre",
        lora_test_multiplier=0.7,
        lora_test_folder="/loras/test",
    )
    test_args = build_args_for_test_lora(base_args, lora_path)

    # Fixed LoRA preserved, test LoRA appended with its multiplier.
    assert test_args.lora_weight == ["/fixed/a.safetensors", lora_path]
    assert test_args.lora_multiplier == [1.0, 0.7]
    # Injection composed after the user's pre_prompt.
    assert test_args.pre_prompt == "userpre loratrigger"
    assert test_args.lora_test_prompt_prefix == "userpre loratrigger"
    assert test_args.current_test_lora == f"{lora_path} 0.7"
    # The sweep flag is cleared on the per-test args so it does not recurse.
    assert test_args.lora_test_folder is None
    # base_args is not mutated.
    assert base_args.lora_weight == ["/fixed/a.safetensors"]


def _args_for_png_metadata():
    return argparse.Namespace(
        prompt="a cat, masterpiece",
        negative_prompt="blurry",
        image_size=[1536, 1024],
        infer_steps=30,
        guidance_scale=4.5,
        seed=7,
        sampler="er_sde",
        scheduler="beta57",
        flow_shift=2.0,
        # Path-bearing fields that must NOT be embedded:
        dit="/models/dit.safetensors",
        vae="/models/vae.safetensors",
        text_encoder="/models/te.safetensors",
        lora_weight=["/loras/fixed.safetensors"],
        lora_multiplier=[1.0],
        current_source_image_path="/src/source.png",
        current_test_lora="/loras/test.safetensors 0.9",
    )


def test_build_png_generation_metadata_text_excludes_paths():
    text = build_png_generation_metadata_text(_args_for_png_metadata())
    for path_fragment in ("/models/", "/loras/", "/src/", "source.png", "dit", "vae", "text_encoder", "lora"):
        assert path_fragment not in text, f"path data {path_fragment!r} leaked into PNG metadata:\n{text}"


def test_build_png_generation_metadata_text_roundtrips_through_parser():
    text = build_png_generation_metadata_text(_args_for_png_metadata())
    parsed = parse_a1111_png_prompt_metadata(text)
    assert parsed["prompt"] == "a cat, masterpiece"
    assert parsed["negative_prompt"] == "blurry"
    assert parsed["infer_steps"] == 30
    assert parsed["guidance_scale"] == 4.5
    assert parsed["seed"] == 7
    # image_size is [height, width]; A1111 Size is WIDTHxHEIGHT.
    assert parsed["image_size_width"] == 1024
    assert parsed["image_size_height"] == 1536
    assert parsed["sampler"] == "er_sde"
    assert parsed["scheduler"] == "beta57"


def test_png_embedded_metadata_roundtrips_via_reader(tmp_path):
    text = build_png_generation_metadata_text(_args_for_png_metadata())
    png_path = os.path.join(tmp_path, "generated.png")
    _write_png_with_parameters(png_path, text)
    parsed = read_png_parsed_metadata(png_path)
    assert parsed["prompt"] == "a cat, masterpiece"
    assert parsed["seed"] == 7
    assert parsed["sampler"] == "er_sde"


def test_detect_combined_checkpoint_true_when_all_three_prefixes_present():
    keys = [
        "model.diffusion_model.net.blocks.0.weight",
        "first_stage_model.decoder.conv1.weight",
        "cond_stage_model.qwen3_06b.transformer.model.embed_tokens.weight",
    ]
    assert detect_combined_checkpoint(keys) is True


def test_detect_combined_checkpoint_false_for_split_dit_only():
    keys = ["net.blocks.0.weight", "net.blocks.1.weight"]
    assert detect_combined_checkpoint(keys) is False


def test_derive_extracted_models_folder_strips_extension():
    assert derive_extracted_models_folder("/models/anima/pearlySapphire_v10.safetensors") == "/models/anima/pearlySapphire_v10"


def test_rename_combined_component_keys_dit_adds_net_prefix():
    combined_keys = ["model.diffusion_model.blocks.0.weight", "model.diffusion_model.final_layer.weight", "other.key"]
    mapping = rename_combined_component_keys(combined_keys, "model.diffusion_model.", "net.")
    assert mapping == {
        "net.blocks.0.weight": "model.diffusion_model.blocks.0.weight",
        "net.final_layer.weight": "model.diffusion_model.final_layer.weight",
    }


def test_rename_combined_component_keys_vae_strips_prefix():
    combined_keys = ["first_stage_model.conv1.bias", "first_stage_model.decoder.conv1.weight"]
    mapping = rename_combined_component_keys(combined_keys, "first_stage_model.", "")
    assert mapping == {
        "conv1.bias": "first_stage_model.conv1.bias",
        "decoder.conv1.weight": "first_stage_model.decoder.conv1.weight",
    }


def test_rename_combined_component_keys_text_encoder_excludes_logit_scale():
    combined_keys = [
        "cond_stage_model.qwen3_06b.transformer.model.embed_tokens.weight",
        "cond_stage_model.qwen3_06b.logit_scale",  # not under .transformer. -> excluded
    ]
    mapping = rename_combined_component_keys(combined_keys, "cond_stage_model.qwen3_06b.transformer.", "")
    assert mapping == {"model.embed_tokens.weight": "cond_stage_model.qwen3_06b.transformer.model.embed_tokens.weight"}


def test_combined_checkpoint_components_cover_the_three_models():
    names = {component["name"] for component in COMBINED_CHECKPOINT_COMPONENTS}
    assert names == {"dit", "vae", "text_encoder"}
    for component in COMBINED_CHECKPOINT_COMPONENTS:
        assert component["filename"].endswith(".safetensors")
        assert component["arg"] in {"dit", "vae", "text_encoder"}


def test_build_args_for_test_lora_no_fixed_loras(tmp_path):
    lora_path = os.path.join(tmp_path, "solo_lora.safetensors")
    open(lora_path, "w").close()
    base_args = argparse.Namespace(
        lora_weight=None,
        lora_multiplier=1.0,
        pre_prompt="",
        lora_test_multiplier=1.0,
        lora_test_folder="/loras/test",
    )
    test_args = build_args_for_test_lora(base_args, lora_path)
    assert test_args.lora_weight == [lora_path]
    assert test_args.lora_multiplier == [1.0]
    assert test_args.pre_prompt == ""


def test_normalize_from_image_embed_plain_folder():
    args = argparse.Namespace(from_image_embed=["/some/folder"])
    normalize_from_image_embed_arg(args)

    assert args.from_image_embed == "/some/folder"
    assert args.from_image_embed_prompts_only is False


def test_normalize_from_image_embed_prompts_only_keyword():
    args = argparse.Namespace(from_image_embed=["prompts_only", "/some/folder"])
    normalize_from_image_embed_arg(args)

    assert args.from_image_embed == "/some/folder"
    assert args.from_image_embed_prompts_only is True
    assert args.from_image_embed_ignore_negative_prompt is False


def test_normalize_from_image_embed_ignore_negative_keyword():
    args = argparse.Namespace(from_image_embed=["ignore_negative_prompt", "/some/folder"])
    normalize_from_image_embed_arg(args)

    assert args.from_image_embed == "/some/folder"
    assert args.from_image_embed_prompts_only is False
    assert args.from_image_embed_ignore_negative_prompt is True


def test_normalize_from_image_embed_both_keywords_any_order():
    args = argparse.Namespace(from_image_embed=["ignore_negative_prompt", "prompts_only", "/some/folder"])
    normalize_from_image_embed_arg(args)

    assert args.from_image_embed == "/some/folder"
    assert args.from_image_embed_prompts_only is True
    assert args.from_image_embed_ignore_negative_prompt is True


def test_normalize_from_image_embed_none():
    args = argparse.Namespace(from_image_embed=None)
    normalize_from_image_embed_arg(args)

    assert args.from_image_embed is None
    assert args.from_image_embed_prompts_only is False
    assert args.from_image_embed_ignore_negative_prompt is False


def test_normalize_from_image_embed_rejects_extra_tokens():
    args = argparse.Namespace(from_image_embed=["/folder_a", "/folder_b"])
    try:
        normalize_from_image_embed_arg(args)
    except ValueError:
        return
    raise AssertionError("expected ValueError for multiple folder paths")


def build_minimal_generation_args_namespace():
    return argparse.Namespace(
        prompt="a bunny",
        negative_prompt="low quality",
        image_size=[1216, 832],  # [height, width]
        infer_steps=50,
        guidance_scale=3.5,
        flow_shift=5.0,
        seed=42,
        sampler="er_sde",
        scheduler="beta57",
        dit="/models/dit.safetensors",
        vae="/models/vae.safetensors",
        text_encoder="/models/te.safetensors",
    )


def test_generation_settings_dict_has_core_fields_with_width_and_height_separated():
    settings = build_generation_settings_dict(build_minimal_generation_args_namespace())
    assert settings["prompt"] == "a bunny"
    assert settings["negative_prompt"] == "low quality"
    assert settings["width"] == 832 and settings["height"] == 1216
    assert settings["steps"] == 50
    assert settings["guidance_scale"] == 3.5
    assert settings["seed"] == 42
    assert settings["sampler"] == "er_sde" and settings["scheduler"] == "beta57"
    assert settings["dit"].endswith("dit.safetensors")
    # Optional keys are absent when not applicable.
    assert "loras" not in settings
    assert "source_image" not in settings
    assert "test_lora" not in settings


def test_generation_settings_dict_includes_merged_loras_marked_enabled_when_no_record_json():
    args = build_minimal_generation_args_namespace()
    args.lora_weight = ["/loras/a.safetensors", "/loras/b.safetensors"]
    args.lora_multiplier = [0.8, 1.0]
    settings = build_generation_settings_dict(args)
    assert settings["loras"] == [
        {"path": "/loras/a.safetensors", "multiplier": 0.8, "enabled": True},
        {"path": "/loras/b.safetensors", "multiplier": 1.0, "enabled": True},
    ]


def test_generation_settings_dict_records_disabled_lora_rows_from_record_json():
    args = build_minimal_generation_args_namespace()
    # Only the enabled row was merged, but the record JSON carries the full ordered list.
    args.lora_weight = ["/loras/enabled.safetensors"]
    args.lora_multiplier = [1.0]
    args.record_lora_rows_json = json.dumps(
        [
            {"path": "/loras/enabled.safetensors", "multiplier": 1.0, "enabled": True},
            {"path": "/loras/disabled.safetensors", "multiplier": 0.5, "enabled": False},
        ]
    )
    settings = build_generation_settings_dict(args)
    assert settings["loras"] == [
        {"path": "/loras/enabled.safetensors", "multiplier": 1.0, "enabled": True},
        {"path": "/loras/disabled.safetensors", "multiplier": 0.5, "enabled": False},
    ]


def test_generation_settings_dict_ignores_unparseable_record_json_and_falls_back_to_merged():
    args = build_minimal_generation_args_namespace()
    args.lora_weight = ["/loras/a.safetensors"]
    args.lora_multiplier = [1.0]
    args.record_lora_rows_json = "{not valid json"
    settings = build_generation_settings_dict(args)
    assert settings["loras"] == [{"path": "/loras/a.safetensors", "multiplier": 1.0, "enabled": True}]


def test_generation_settings_dict_records_source_image_and_test_lora_when_set():
    args = build_minimal_generation_args_namespace()
    args.current_source_image_path = "/refs/reference.png"
    args.current_test_lora = "/loras/test.safetensors 1.0"
    settings = build_generation_settings_dict(args)
    assert settings["source_image"] == "/refs/reference.png"
    assert settings["test_lora"] == "/loras/test.safetensors 1.0"


def _lock_path_is_currently_held_by_another_fd(lock_file_path):
    """True if an exclusive non-blocking flock on lock_file_path fails because it is already held.

    Uses a separate open file description, so within one process it still contends with a lock held via
    serialize_model_file_disk_reads (flock locks are per open-file-description on Linux)."""
    import fcntl

    with open(lock_file_path, "w") as probe_file:
        try:
            fcntl.flock(probe_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(probe_file.fileno(), fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True


def test_serialize_model_file_disk_reads_is_noop_when_path_is_falsy(tmp_path):
    body_ran = []
    with serialize_model_file_disk_reads(None):
        body_ran.append(True)
    with serialize_model_file_disk_reads(""):
        body_ran.append(True)
    assert body_ran == [True, True]


def test_serialize_model_file_disk_reads_holds_and_releases_exclusive_lock(tmp_path):
    lock_file_path = str(tmp_path / "model_load_disk.lock")

    assert not _lock_path_is_currently_held_by_another_fd(lock_file_path)  # free before

    with serialize_model_file_disk_reads(lock_file_path):
        assert _lock_path_is_currently_held_by_another_fd(lock_file_path)  # held during

    assert not _lock_path_is_currently_held_by_another_fd(lock_file_path)  # released after


def test_serialize_model_file_disk_reads_releases_even_when_body_raises(tmp_path):
    lock_file_path = str(tmp_path / "model_load_disk.lock")

    class DiskReadFailure(Exception):
        pass

    try:
        with serialize_model_file_disk_reads(lock_file_path):
            raise DiskReadFailure()
    except DiskReadFailure:
        pass

    assert not _lock_path_is_currently_held_by_another_fd(lock_file_path)  # released despite the raise


def test_hold_lock_is_reentrant_within_thread_without_deadlocking(tmp_path):
    lock_file_path = str(tmp_path / "reentrant.lock")
    # Nesting the same path must NOT self-deadlock (a naive second flock on a new fd of the same file
    # would block forever), and the lock must stay held across both levels.
    with hold_exclusive_cross_process_file_lock(lock_file_path):
        assert _lock_path_is_currently_held_by_another_fd(lock_file_path)
        with hold_exclusive_cross_process_file_lock(lock_file_path):
            assert _lock_path_is_currently_held_by_another_fd(lock_file_path)  # still held (inner)
        assert _lock_path_is_currently_held_by_another_fd(lock_file_path)  # still held after inner exits
    assert not _lock_path_is_currently_held_by_another_fd(lock_file_path)  # released after outermost


def test_loading_phase_holds_the_disk_lock_across_nested_per_file_reads(tmp_path):
    lock_file_path = str(tmp_path / "model_load_disk.lock")
    args = argparse.Namespace(model_load_disk_lock_file=lock_file_path)

    assert not _lock_path_is_currently_held_by_another_fd(lock_file_path)
    with serialize_model_loading_phase(args):
        # Simulate the per-file load locks nested inside the phase: the lock stays held continuously
        # (never released between files), so a competing process cannot slip in and requeue this one.
        with serialize_model_file_disk_reads(lock_file_path):
            assert _lock_path_is_currently_held_by_another_fd(lock_file_path)
        assert _lock_path_is_currently_held_by_another_fd(lock_file_path)  # STILL held between files
        with serialize_model_file_disk_reads(lock_file_path):
            assert _lock_path_is_currently_held_by_another_fd(lock_file_path)
        assert _lock_path_is_currently_held_by_another_fd(lock_file_path)  # STILL held between files
    assert not _lock_path_is_currently_held_by_another_fd(lock_file_path)  # released when phase ends


def test_loading_phase_is_noop_when_no_lock_file(tmp_path):
    args = argparse.Namespace(model_load_disk_lock_file=None)
    ran = []
    with serialize_model_loading_phase(args):
        ran.append(True)
    assert ran == [True]


def _make_gpu_lock_args(gpu_compute_lock_file, gpu_lock_scope):
    return argparse.Namespace(gpu_compute_lock_file=gpu_compute_lock_file, gpu_lock_scope=gpu_lock_scope)


def test_gpu_phase_is_covered_by_scope():
    # Denoise is always covered, regardless of scope.
    assert gpu_phase_is_covered_by_scope(GPU_PHASE_DENOISE, GPU_LOCK_SCOPE_DENOISE_ONLY)
    assert gpu_phase_is_covered_by_scope(GPU_PHASE_DENOISE, GPU_LOCK_SCOPE_ALL_COMPUTE)
    # Text-encode and VAE-decode are covered only under the strict 'all compute' scope.
    assert not gpu_phase_is_covered_by_scope(GPU_PHASE_TEXT_ENCODE, GPU_LOCK_SCOPE_DENOISE_ONLY)
    assert not gpu_phase_is_covered_by_scope(GPU_PHASE_VAE_DECODE, GPU_LOCK_SCOPE_DENOISE_ONLY)
    assert gpu_phase_is_covered_by_scope(GPU_PHASE_TEXT_ENCODE, GPU_LOCK_SCOPE_ALL_COMPUTE)
    assert gpu_phase_is_covered_by_scope(GPU_PHASE_VAE_DECODE, GPU_LOCK_SCOPE_ALL_COMPUTE)


def test_serialize_gpu_compute_is_noop_when_no_lock_file(tmp_path):
    args = _make_gpu_lock_args(None, GPU_LOCK_SCOPE_ALL_COMPUTE)
    ran = []
    with serialize_gpu_compute(args, GPU_PHASE_DENOISE):
        ran.append(True)
    assert ran == [True]


def test_serialize_gpu_compute_denoise_only_scope_locks_denoise_not_encode_or_decode(tmp_path):
    lock_file_path = str(tmp_path / "gpu_compute.lock")
    args = _make_gpu_lock_args(lock_file_path, GPU_LOCK_SCOPE_DENOISE_ONLY)

    with serialize_gpu_compute(args, GPU_PHASE_DENOISE):
        assert _lock_path_is_currently_held_by_another_fd(lock_file_path)  # denoise IS locked

    with serialize_gpu_compute(args, GPU_PHASE_TEXT_ENCODE):
        assert not _lock_path_is_currently_held_by_another_fd(lock_file_path)  # encode is NOT locked

    with serialize_gpu_compute(args, GPU_PHASE_VAE_DECODE):
        assert not _lock_path_is_currently_held_by_another_fd(lock_file_path)  # decode is NOT locked


def test_serialize_gpu_compute_all_scope_locks_every_phase(tmp_path):
    lock_file_path = str(tmp_path / "gpu_compute.lock")
    args = _make_gpu_lock_args(lock_file_path, GPU_LOCK_SCOPE_ALL_COMPUTE)

    for gpu_phase in (GPU_PHASE_TEXT_ENCODE, GPU_PHASE_DENOISE, GPU_PHASE_VAE_DECODE):
        with serialize_gpu_compute(args, gpu_phase):
            assert _lock_path_is_currently_held_by_another_fd(lock_file_path), gpu_phase
        assert not _lock_path_is_currently_held_by_another_fd(lock_file_path), gpu_phase  # released after


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            if "tmp_path" in func.__code__.co_varnames[: func.__code__.co_argcount]:
                continue
            func()
    print("non-tmp tests passed")
