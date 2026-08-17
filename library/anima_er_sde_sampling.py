"""ER-SDE sampler and beta57 scheduler for Anima (rectified-flow / CONST model).

Faithful ports of:
- ComfyUI comfy/k_diffusion/sampling.py :: sample_er_sde  (ER-SDE-Solver-3, arXiv:2309.06169),
  with the flow/CONST log-SNR branch hardcoded (er_lambda = sigma / (1 - sigma)).
- ComfyUI comfy/samplers.py :: beta_scheduler, using RES4LYF's beta57 preset (alpha=0.5, beta=0.7),
  mapped onto the rectified-flow shifted sigma table.

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


@torch.no_grad()
def sample_er_sde_rectified_flow(
    predict_denoised_x0,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    seed=None,
    s_noise: float = 1.0,
    max_stage: int = 3,
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
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # Stage 1 (Euler)
            latents = r_alpha * r * latents + alpha_t * (1 - r) * denoised

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

            if s_noise > 0:
                injected = (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
                latents = latents + alpha_t * sample_standard_noise(latents) * s_noise * injected

        old_denoised = denoised

    return latents
