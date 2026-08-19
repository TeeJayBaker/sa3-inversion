"""
RF-Inversion for Stable Audio 3.

Rout et al., 2024 — "Semantic Image Inversion and Editing using Rectified
Stochastic Differential Equations" — adapted to the SA3 rectified-flow base
checkpoints (e.g. stabilityai/stable-audio-3-medium-base).

Conventions, taken straight from the repo (`inference/sampling.py`):

    x_t = (1 - t) * x_0 + t * eps        t = 1 is noise, t = 0 is data
    DiT predicts the velocity v = dx/dt = eps - x_0

so sampling integrates t: 1 -> 0, and inversion is the *same* ODE integrated
the other way, t: 0 -> 1. Both directions share one schedule, so the two passes
visit identical timesteps — see `get_schedule` for the options.

The two controllers from the paper are optional linear blends in velocity space:

    invert  gamma: pull toward a target noise sample  v = (eps - y_t) / (1 - t)
    sample  eta:   pull back toward the source latent v = (x_t - x_0) / t

gamma = eta = 0 gives plain deterministic inversion / re-sampling, which is the
reconstruction sanity check. On medium-base that reconstructs a 10s clip to
0.018 relative error in latent space at 50 steps — see `invert`'s
`fixed_point_iters`, without which it is 0.63.
"""

import torch
from tqdm.auto import tqdm

from stable_audio_3.inference.sampling import build_schedule

EPS = 1e-3


def get_schedule(model, steps, latent_len, device, schedule="model", logsnr_range=6.0):
    """Descending timestep schedule, t: 1 -> 0. Pass a tensor to use it verbatim.

    "model"  — what SA3 uses for generation, and the best of these for inversion too.
               t = sigmoid(8.2 * u - 2.0), endpoints forced to 0 and 1. Strongly
               noise-dense: at 25 steps dt is 0.0028 next to t=1 but 0.158 next to
               t=0, and 12 of the 25 steps sit above t=0.9 (the SD3/Flux "shift
               toward noise").
    "logsnr" — uniform in log-SNR over [-logsnr_range, +logsnr_range], t =
               sigmoid(-lambda). Since dt = t(1-t) dlambda it tapers at BOTH ends.
    "linear" — evenly spaced.

    It is tempting to assume inversion wants the density at the DATA end, since
    forcing t=0 leaves "model" with one oversized interval there (its natural value
    is sigmoid(-2.0) = 0.119, so the data end is a cliff, not a taper). Measured on
    medium-base, 10s of audio, 25 steps, fixed_point_iters=2, that is backwards —
    reconstruction relative error:

        model 0.152 | logsnr 0.293 | t^2 0.345 | t^1.5 0.516 | linear 0.569

    The cliff is what `fixed_point_iters` exists to absorb; what dominates once it
    is handled is resolving the stiff field near t=1, and shifting steps away from
    there costs far more than the extra low-t resolution buys. Steps matter more
    than spacing anyway: with "model" + fp=2, 25 -> 50 -> 100 steps gives
    0.154 -> 0.018 -> 0.007.
    """
    if torch.is_tensor(schedule):
        return schedule.to(device)
    if schedule == "linear":
        return torch.linspace(1.0, 0.0, steps + 1, device=device)
    if schedule == "logsnr":
        lam = torch.linspace(-logsnr_range, logsnr_range, steps + 1, device=device)
        t = torch.sigmoid(-lam)
        t[0], t[-1] = 1.0, 0.0
        return t
    if schedule != "model":
        raise ValueError(f"unknown schedule {schedule!r}: expected 'model', 'logsnr', 'linear', or a tensor")
    return build_schedule(
        steps=steps,
        sigma_max=1.0,
        dist_shift=model.model.sampling_dist_shift,
        fallback_seq_len=latent_len,
        include_endpoint=True,
        device=device,
    )


def make_cond(model, prompt, seconds_total, latent_len, batch_size=1):
    """Build the DiT conditioning kwargs for a prompt: text + duration + null inpainting cond."""
    inner, device = model.model, model.device
    tensors = inner.conditioner(
        [{"prompt": prompt, "seconds_total": seconds_total}] * batch_size, device
    )
    # Checkpoints that declare the inpainting local-add conditioning always expect it
    # present; zeros mean "not inpainting", which is what generate() passes.
    zeros = {
        "inpaint_mask": lambda: torch.zeros(batch_size, 1, latent_len, device=device),
        "inpaint_masked_input": lambda: torch.zeros(
            batch_size, inner.io_channels, latent_len, device=device
        ),
    }
    for key in inner.local_add_cond_ids:
        if key in zeros:
            tensors[key] = [zeros[key]()]
    dtype = next(inner.model.parameters()).dtype
    return {
        k: v.to(dtype) if torch.is_tensor(v) else v
        for k, v in inner.get_conditioning_inputs(tensors).items()
    }


def _blend(v_model, v_ctrl, weight, norm_match=False):
    """Linear interpolation between the model velocity and a controller velocity."""
    if weight == 0.0:
        return v_model
    if norm_match:
        # Rescale the controller so neither field dominates, then restore the
        # model's output scale — the stabilisation used in the SAO-small notebook.
        scale = (
            v_model.norm(dim=(1, 2), keepdim=True)
            / (v_ctrl.norm(dim=(1, 2), keepdim=True) + 1e-8)
        ).clamp(0.0, 3.0)
        v = v_model + weight * (scale * v_ctrl - v_model)
        return v * (
            v_model.std(dim=(1, 2), keepdim=True) / (v.std(dim=(1, 2), keepdim=True) + 1e-8)
        )
    return v_model + weight * (v_ctrl - v_model)


@torch.inference_mode()
def invert(
    model,
    latent,
    cond,
    steps=50,
    gamma=0.0,
    target_noise=None,
    schedule="model",
    fixed_point_iters=2,
    norm_match=False,
    disable_tqdm=False,
):
    """Data -> noise. Integrates the flow ODE forward in t (0 -> 1).

    `sample` steps down from t_next to t_curr as

        x_curr = x_next + (t_curr - t_next) * v(x_next, t_next)

    so a plain forward-Euler step up — which evaluates v at (y, t_curr) instead —
    is *not* its inverse, and the mismatch compounds badly enough that throwing
    steps at it barely helps. `fixed_point_iters` instead solves that equation for
    x_next directly, by iterating

        z <- y + dt * v(z, t_next)

    from the Euler point. At convergence `sample(invert(x))` returns x. Set 0 for
    plain Euler; each iteration costs one extra model evaluation.

    2 is the sweet spot and more does not help — measured on medium-base, 10s of
    audio, "model" schedule, reconstruction relative error:

        steps=25: fp=0 0.626 | fp=2 0.154 | fp=4 0.163
        steps=50: fp=0 0.484 | fp=2 0.018 | fp=4 0.027

    Past 2 the iteration stops paying because the interval next to t=0 is too wide
    to contract (its residual falls ~0.7 per iteration, against ~0.1 elsewhere).
    Extra steps shrink that interval and are the better spend from there.

    gamma > 0 steers the trajectory toward `target_noise` so the result lands
    closer to the Gaussian prior, at the cost of reconstruction fidelity. The
    controller stays explicit (evaluated at t_curr) — the schedule ends at
    exactly t = 1, where the controller's 1/(1 - t) would blow up.
    """
    dit = model.model.model
    t = get_schedule(model, steps, latent.shape[-1], latent.device, schedule).flip(0)  # ascending
    y = latent.clone()
    noise = torch.randn_like(y) if target_noise is None else target_noise
    batch = y.shape[0]

    for t_curr, t_next in tqdm(
        list(zip(t[:-1], t[1:])), desc="invert", disable=disable_tqdm
    ):
        dt = t_next - t_curr  # > 0
        # The schedule starts at exactly t=0, which the model never saw in training;
        # nudge the evaluation point off the boundary the way SA3's own rk4 sampler does.
        t_eval = t_curr.clamp(min=1e-5).expand(batch)
        v_ctrl = (noise - y) / (1.0 - t_curr).clamp(min=EPS)

        z = y + dt * _blend(dit(y, t_eval, cfg_scale=1.0, **cond), v_ctrl, gamma, norm_match)

        t_next_eval = t_next.expand(batch)
        for _ in range(fixed_point_iters):
            v_model = dit(z, t_next_eval, cfg_scale=1.0, **cond)
            z = y + dt * _blend(v_model, v_ctrl, gamma, norm_match)

        y = z

    return y


@torch.inference_mode()
def sample(
    model,
    noise,
    cond,
    steps=50,
    cfg_scale=1.0,
    apg_scale=1.0,
    eta=0.0,
    source_latent=None,
    start=0.0,
    stop=1.0,
    schedule="model",
    norm_match=False,
    disable_tqdm=False,
):
    """Noise -> data. Euler integration of the flow ODE (t: 1 -> 0).

    eta > 0 steers back toward `source_latent` over the [start, stop] fraction of
    the trajectory, which is what preserves the original structure while the
    prompt changes the content.
    """
    dit = model.model.model
    t = get_schedule(model, steps, noise.shape[-1], noise.device, schedule)  # descending
    x = noise.clone()

    n_steps = len(t) - 1
    first, last = int(start * n_steps), int(stop * n_steps)

    for i, (t_curr, t_next) in enumerate(
        tqdm(list(zip(t[:-1], t[1:])), desc="sample", disable=disable_tqdm)
    ):
        dt = t_next - t_curr  # < 0
        v_model = dit(
            x,
            t_curr.expand(x.shape[0]),
            cfg_scale=cfg_scale,
            apg_scale=apg_scale,
            **cond,
        )
        if eta != 0.0 and first <= i <= last:
            v_ctrl = (x - source_latent) / t_curr.clamp(min=EPS)
            v = _blend(v_model, v_ctrl, eta, norm_match)
        else:
            v = v_model
        x = x + dt * v

    return x


def invert_and_edit(
    model,
    latent,
    target_prompt,
    seconds_total,
    source_prompt="",
    inversion_steps=50,
    sampling_steps=50,
    gamma=0.0,
    eta=0.0,
    start=0.0,
    stop=1.0,
    cfg_scale=1.0,
    apg_scale=1.0,
    schedule="model",
    fixed_point_iters=2,
    norm_match=False,
    seed=None,
    disable_tqdm=False,
):
    """Full pipeline: invert `latent` under `source_prompt`, re-sample under `target_prompt`.

    Returns (edited_latent, inverted_noise).
    """
    if seed is not None:
        torch.manual_seed(seed)

    latent_len = latent.shape[-1]
    batch_size = latent.shape[0]

    src_cond = make_cond(model, source_prompt, seconds_total, latent_len, batch_size)
    inverted = invert(
        model,
        latent,
        src_cond,
        steps=inversion_steps,
        gamma=gamma,
        schedule=schedule,
        fixed_point_iters=fixed_point_iters,
        norm_match=norm_match,
        disable_tqdm=disable_tqdm,
    )

    tgt_cond = make_cond(model, target_prompt, seconds_total, latent_len, batch_size)
    edited = sample(
        model,
        inverted,
        tgt_cond,
        steps=sampling_steps,
        cfg_scale=cfg_scale,
        apg_scale=apg_scale,
        eta=eta,
        source_latent=latent,
        start=start,
        stop=stop,
        schedule=schedule,
        norm_match=norm_match,
        disable_tqdm=disable_tqdm,
    )

    return edited, inverted
