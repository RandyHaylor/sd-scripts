"""ER-SDE sampler and beta57 scheduler for Anima (rectified-flow / CONST model).

Faithful ports of:
- ComfyUI comfy/k_diffusion/sampling.py :: sample_er_sde  (ER-SDE-Solver-3, arXiv:2309.06169),
  with the flow/CONST log-SNR branch hardcoded (er_lambda = sigma / (1 - sigma)).
- ComfyUI comfy/samplers.py :: beta_scheduler, using RES4LYF's beta57 preset (alpha=0.5, beta=0.7),
  mapped onto the rectified-flow shifted sigma table.
- huggingface/diffusers FlowMatchEulerDiscreteScheduler (static-shift schedule) as
  build_diffusers_flowmatch_sigmas, the reference FlowMatch-Euler-style Anima schedule.

Anima is a flow model: the DiT outputs a velocity v at (x_t, sigma), and the denoised x0 estimate is
x0 = x_t - sigma * v  (ComfyUI CONST.calculate_denoised). ER-SDE consumes that x0 estimate.
"""

import torch
from tqdm import tqdm


def compute_flow_shifted_sigma(flow_shift: float, normalized_time: float) -> float:
    """ComfyUI time_snr_shift(alpha=flow_shift, t=normalized_time). Identity when flow_shift == 1."""
    if flow_shift == 1.0:
        return normalized_time
    return flow_shift * normalized_time / (1.0 + (flow_shift - 1.0) * normalized_time)


def build_beta57_sigmas(
    num_inference_steps: int,
    flow_shift: float,
    device: torch.device,
    alpha: float = 0.5,
    beta: float = 0.7,
) -> torch.Tensor:
    """Beta-distribution sigma schedule (RES4LYF 'beta57' = alpha 0.5 / beta 0.7).

    Mirrors ComfyUI beta_scheduler: it indexes a precomputed monotonic sigma table with
    timestep positions drawn from the Beta CDF (scipy.stats.beta.ppf). The sigma table here is
    the rectified-flow shifted schedule s[k] = time_snr_shift(flow_shift, k / total_timesteps),
    so the result stays in the flow sigma convention ([0, 1], descending toward 0).
    """
    import numpy as np
    from scipy.stats import beta as beta_distribution

    total_timesteps = 1000
    # sigma_table[k] corresponds to timestep index k (k=0 -> ~0 noise, k=total-1 -> ~max noise)
    sigma_table = [
        compute_flow_shifted_sigma(flow_shift, (k + 1) / total_timesteps) for k in range(total_timesteps)
    ]

    quantiles = 1.0 - np.linspace(0.0, 1.0, num_inference_steps, endpoint=False)
    timestep_indices = np.rint(beta_distribution.ppf(quantiles, alpha, beta) * (total_timesteps - 1))
    timestep_indices = np.clip(timestep_indices, 0, total_timesteps - 1).astype(np.int64)

    selected_sigmas = []
    previous_index = None
    for index in timestep_indices:
        if index != previous_index:
            selected_sigmas.append(float(sigma_table[int(index)]))
            previous_index = index
    selected_sigmas.append(0.0)
    return torch.tensor(selected_sigmas, dtype=torch.float32, device=device)


def build_simple_sigmas(
    num_inference_steps: int,
    flow_shift: float,
    device: torch.device,
) -> torch.Tensor:
    """'simple' sigma schedule. Faithful port of ComfyUI comfy/samplers.py :: simple_scheduler,
    indexing the rectified-flow shifted sigma table (so flow_shift still applies).

    ComfyUI evenly strides the ascending model sigma table from the high-noise end:
    sigmas[-(1 + int(x * len/steps))] for x in range(steps), then appends 0.0.
    """
    total_timesteps = 1000
    sigma_table = [
        compute_flow_shifted_sigma(flow_shift, (k + 1) / total_timesteps) for k in range(total_timesteps)
    ]

    step_stride = total_timesteps / num_inference_steps
    selected_sigmas = [float(sigma_table[-(1 + int(x * step_stride))]) for x in range(num_inference_steps)]
    selected_sigmas.append(0.0)
    return torch.tensor(selected_sigmas, dtype=torch.float32, device=device)


def build_diffusers_flowmatch_sigmas(
    num_inference_steps: int,
    flow_shift: float,
    device: torch.device,
) -> torch.Tensor:
    """Diffusers FlowMatchEulerDiscreteScheduler sigma schedule (static-shift path).

    Faithful port of huggingface/diffusers scheduling_flow_match_euler_discrete.py set_timesteps
    for the non-dynamic-shifting case, with num_train_timesteps=1000:
        timesteps = linspace(sigma_max * 1000, sigma_min * 1000, num_inference_steps)
                    with sigma_max = 1.0, sigma_min = 1 / 1000
        sigmas    = timesteps / 1000                       -> linspace(1.0, 1/1000, steps)
        sigmas    = flow_shift * sigmas / (1 + (flow_shift - 1) * sigmas)   # static linear shift
        sigmas    = cat([sigmas, 0.0])                     # trailing 0 target for the final step

    Stays in the flow sigma convention ([0, 1], descending toward 0) that Anima's DiT consumes
    directly as its time input (matching training, where timesteps are scaled to [0, 1]).
    """
    num_train_timesteps = 1000
    sigma_max = 1.0
    sigma_min = 1.0 / num_train_timesteps

    timesteps = torch.linspace(
        sigma_max * num_train_timesteps,
        sigma_min * num_train_timesteps,
        num_inference_steps,
        dtype=torch.float32,
    )
    sigmas = timesteps / num_train_timesteps
    sigmas = flow_shift * sigmas / (1.0 + (flow_shift - 1.0) * sigmas)
    sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float32)])
    return sigmas.to(device=device)


# Qwen-Image's own FlowMatchEulerDiscreteScheduler config (Qwen/Qwen-Image scheduler_config.json):
# use_dynamic_shifting=true, base_shift=0.5, max_shift=0.9, base_image_seq_len=256,
# max_image_seq_len=8192, time_shift_type="exponential". Anima is Qwen-Image-based, so these are the
# faithful resolution-aware shift constants for it.
QWEN_IMAGE_DYNAMIC_SHIFT_BASE_IMAGE_SEQ_LEN = 256
QWEN_IMAGE_DYNAMIC_SHIFT_MAX_IMAGE_SEQ_LEN = 8192
QWEN_IMAGE_DYNAMIC_SHIFT_BASE_SHIFT = 0.5
QWEN_IMAGE_DYNAMIC_SHIFT_MAX_SHIFT = 0.9


def compute_qwen_image_dynamic_shift_mu(
    image_seq_len: int,
    base_shift: float = QWEN_IMAGE_DYNAMIC_SHIFT_BASE_SHIFT,
    max_shift: float = QWEN_IMAGE_DYNAMIC_SHIFT_MAX_SHIFT,
) -> float:
    """Diffusers calculate_shift for Qwen-Image: linear interpolation of mu by packed image token count.

    mu = base_shift + (max_shift - base_shift) * (image_seq_len - base_seq) / (max_seq - base_seq)
    with the Qwen-Image seq-len endpoints. base_shift/max_shift default to the Qwen-Image config values
    but are exposed so they can be tuned. The effective time shift applied to sigmas is exp(mu)
    (time_shift_type="exponential"), so this schedule equals build_diffusers_flowmatch_sigmas at
    flow_shift = exp(mu).
    """
    base_seq = QWEN_IMAGE_DYNAMIC_SHIFT_BASE_IMAGE_SEQ_LEN
    max_seq = QWEN_IMAGE_DYNAMIC_SHIFT_MAX_IMAGE_SEQ_LEN
    slope = (max_shift - base_shift) / (max_seq - base_seq)
    intercept = base_shift - slope * base_seq
    return image_seq_len * slope + intercept


def build_diffusers_dynamic_flowmatch_sigmas(
    num_inference_steps: int,
    image_seq_len: int,
    device: torch.device,
    base_shift: float = QWEN_IMAGE_DYNAMIC_SHIFT_BASE_SHIFT,
    max_shift: float = QWEN_IMAGE_DYNAMIC_SHIFT_MAX_SHIFT,
) -> torch.Tensor:
    """Diffusers FlowMatchEulerDiscreteScheduler sigma schedule (dynamic-shifting path) for Qwen-Image.

    Faithful port of set_timesteps with use_dynamic_shifting=true and time_shift_type="exponential",
    num_train_timesteps=1000:
        sigmas = linspace(1.0, 1/1000, num_inference_steps)
        mu     = compute_qwen_image_dynamic_shift_mu(image_seq_len)
        sigmas = exp(mu) / (exp(mu) + (1 / sigmas - 1))        # time_shift(mu, 1.0, sigmas)
        sigmas = cat([sigmas, 0.0])

    image_seq_len is the packed image token count the DiT processes. For Anima (VAE /8, patch_spatial 2,
    image frame) that is (height // 16) * (width // 16). Resolution-aware: no fixed flow_shift; the shift
    grows with resolution exactly as Qwen-Image samples.
    """
    import math

    num_train_timesteps = 1000
    sigma_max = 1.0
    sigma_min = 1.0 / num_train_timesteps

    timesteps = torch.linspace(
        sigma_max * num_train_timesteps,
        sigma_min * num_train_timesteps,
        num_inference_steps,
        dtype=torch.float32,
    )
    sigmas = timesteps / num_train_timesteps
    mu = compute_qwen_image_dynamic_shift_mu(image_seq_len, base_shift, max_shift)
    exp_mu = math.exp(mu)
    sigmas = exp_mu / (exp_mu + (1.0 / sigmas - 1.0))
    sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float32)])
    return sigmas.to(device=device)


@torch.no_grad()
def sample_euler_ancestral_rectified_flow(
    predict_denoised_x0,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    seed=None,
    eta: float = 1.0,
    s_noise: float = 1.0,
) -> torch.Tensor:
    """Ancestral Euler sampler for a rectified-flow (CONST) model.

    Faithful port of ComfyUI comfy/k_diffusion/sampling.py :: sample_euler_ancestral_RF (the branch
    used for CONST/flow models). Each step takes an Euler step toward a reduced sigma_down and, when
    eta > 0, renoises to sigma_{i+1} with a seeded standard-normal sample.

    Args:
        predict_denoised_x0: callable(x, sigma_scalar_tensor) -> denoised x0 estimate (x - sigma * v).
        latents: initial noise tensor.
        sigmas: 1D tensor of length num_steps+1, flow sigmas descending to 0.0 at the end.
        seed: optional int for the stochastic renoise.
        eta: ancestral noise amount (1.0 = full ancestral, 0.0 = deterministic Euler).
        s_noise: scale on injected noise.
    """
    device = latents.device

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    def sample_standard_noise(reference: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            reference.size(), dtype=reference.dtype, layout=reference.layout, device=device, generator=generator
        )

    sigmas = sigmas.to(device=device, dtype=torch.float32)
    latents = latents.to(torch.float32)

    for i in tqdm(range(len(sigmas) - 1), desc="Denoising steps (euler_ancestral)"):
        denoised = predict_denoised_x0(latents, sigmas[i]).to(torch.float32)

        if sigmas[i + 1] == 0:
            latents = denoised
        else:
            downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
            sigma_down = sigmas[i + 1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i + 1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i + 1] ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5

            # Euler step to sigma_down (expressed via the x0 estimate)
            sigma_down_i_ratio = sigma_down / sigmas[i]
            latents = sigma_down_i_ratio * latents + (1 - sigma_down_i_ratio) * denoised

            if eta > 0:
                latents = (alpha_ip1 / alpha_down) * latents + sample_standard_noise(latents) * s_noise * renoise_coeff

    return latents


@torch.no_grad()
def sample_er_sde_rectified_flow(
    predict_denoised_x0,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    seed=None,
    s_noise: float = 1.0,
    max_stage: int = 3,
    report_solver_disagreement=None,
) -> torch.Tensor:
    """ER-SDE-Solver-3 adapted for a rectified-flow (CONST) model.

    Args:
        predict_denoised_x0: callable(x, sigma_scalar_tensor) -> denoised x0 estimate. For a flow
            model this closure runs the DiT (with CFG) to get velocity v and returns x - sigma * v.
        latents: initial noise tensor.
        sigmas: 1D tensor of length num_steps+1, flow sigmas descending to 0.0 at the end.
        seed: optional int for the stochastic noise injection.
        s_noise: scale on injected noise (1.0 = standard ER-SDE).
        max_stage: 1/2/3 selects ER-SDE-Solver order; history ramps stages up as steps accumulate.
        report_solver_disagreement: optional callable(float). Called once per step with the relative
            magnitude of the solver's higher-order correction over the stage-1 Euler update
            (||full_update - euler_update|| / (||euler_update|| + eps)) -- i.e. how strongly ER-SDE
            disagreed with plain Euler this step. Consumers (e.g. ER-SDE Solver PAG) use it with a
            one-step lag, since it is only known after this step's denoise has already run.
    """
    device = latents.device

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    def sample_standard_noise(reference: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            reference.size(), dtype=reference.dtype, layout=reference.layout, device=device, generator=generator
        )

    def noise_scaler(value: torch.Tensor) -> torch.Tensor:
        # ER-SDE noise-scaling function phi (ComfyUI default_er_sde_noise_scaler)
        return value * ((value ** 0.3).exp() + 10.0)

    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=device)

    sigmas = sigmas.to(device=device, dtype=torch.float32)
    # offset_first_sigma_for_snr: flow logit()/ratio needs sigma < 1 (sigma == 1 -> division by zero)
    if float(sigmas[0]) >= 1.0:
        sigmas = sigmas.clone()
        sigmas[0] = 1.0 - 1e-4

    # CONST (flow) branch: half_log_snr = log((1 - sigma)/sigma); er_lambda = sigma / (1 - sigma)
    er_lambdas = sigmas / (1.0 - sigmas)

    latents = latents.to(torch.float32)
    old_denoised = None
    old_denoised_d = None

    for i in tqdm(range(len(sigmas) - 1), desc="Denoising steps (er_sde)"):
        denoised = predict_denoised_x0(latents, sigmas[i]).to(torch.float32)
        stage_used = min(max_stage, i + 1)

        if sigmas[i + 1] == 0:
            latents = denoised
            if report_solver_disagreement is not None:
                report_solver_disagreement(0.0)
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # Stage 1 (Euler). Keep the pure Euler update to measure how far the higher-order stages move.
            euler_update = r_alpha * r * latents + alpha_t * (1 - r) * denoised
            latents = euler_update

            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)

                # Stage 2
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                latents = latents + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3:
                    # Stage 3
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    latents = latents + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d

            # Solver disagreement = relative size of the higher-order correction vs the Euler update
            # (measured on the deterministic update, before the stochastic renoise below).
            if report_solver_disagreement is not None:
                correction_norm = torch.linalg.vector_norm(latents - euler_update)
                euler_norm = torch.linalg.vector_norm(euler_update)
                report_solver_disagreement(float(correction_norm / (euler_norm + 1e-8)))

            if s_noise > 0:
                injected = (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
                latents = latents + alpha_t * sample_standard_noise(latents) * s_noise * injected

        old_denoised = denoised

    return latents
