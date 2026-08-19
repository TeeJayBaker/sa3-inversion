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

Two solver pairings are available, set once via `invert_and_edit(solver=...)`:
"midpoint" (default, the FireFlow scheme of Deng et al., 2024) and "fixed-point",
which is what this repo used to default to and is paired with sample's "euler".
Midpoint reconstructs several times more accurately for half the model evaluations
— see `invert` for the numbers, and for why the two pairings want opposite
schedules, which is why `solver` and `schedule` have to be changed together.

The controllers are calibrated against the solver, not just the prompt: eta needs to
be roughly a third of its old value to move the result as far, because a solver that
already reconstructs cleanly gives it no drift to fight. See `sample`.

For editing that skips the round trip altogether, see `flow_edit.py`.
"""

import torch
from tqdm.auto import tqdm

from stable_audio_3.inference.sampling import build_schedule

EPS = 1e-3


def get_schedule(model, steps, latent_len, device, schedule="model", logsnr_range=6.0):
    """Descending timestep schedule, t: 1 -> 0. Pass a tensor to use it verbatim.

    "model"  — what SA3 uses for generation, and the best of these for a fixed-point
               inversion. t = sigmoid(8.2 * u - 2.0), endpoints forced to 0 and 1. Strongly
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

    All of which holds for the fixed-point solver and inverts for the midpoint one,
    whose error floor is set by that same low-t cliff rather than by the field at
    t=1 — it wants "logsnr", and is the default. See `invert`.
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


def _integrate(x, t, velocity, solver, desc, disable_tqdm):
    """Integrate dx/dt = velocity(x, t, i, stage) along the timestep sequence `t`.

    `velocity` is the whole field being integrated, controller blend included, so
    the solver never has to know which direction it is running or what is steering
    it. `i` is the step index, for controllers with a start/stop window.

    `stage` names *which* evaluation within the step this is — 0 for the one at the
    interval's leading edge, 1 for the midpoint corrector. Nothing here uses it, and
    a controller that only reads `x` and `t` does not need it either; it exists so
    that something caching per-evaluation state can tell the two apart when the
    midpoint solver calls in twice for one step. See `attn_inject`.

    "euler"    — one evaluation per step at the interval's leading edge.
    "midpoint" — FireFlow (Deng et al., 2024, arXiv:2412.07517). Evaluate at the
                 interval midpoint instead, and carry that velocity over as the
                 next step's predictor rather than paying for a fresh evaluation:

                     v      = cached midpoint velocity from step i-1
                     x_mid  = x + dt/2 * v
                     v_mid  = velocity(x_mid, t + dt/2)
                     x      = x + dt * v_mid

                 That is second-order accuracy for one evaluation per step, the
                 reuse being sound because rectified-flow trajectories are near
                 straight, so v barely moves over half a step.
    """
    x = x.clone()
    v_cache = None
    steps = list(zip(t[:-1], t[1:]))

    for i, (t_curr, t_next) in enumerate(tqdm(steps, desc=desc, disable=disable_tqdm)):
        dt = t_next - t_curr
        if solver == "euler":
            x = x + dt * velocity(x, t_curr, i, 0)
        elif solver == "midpoint":
            v = velocity(x, t_curr, i, 0) if v_cache is None else v_cache
            v_cache = velocity(x + 0.5 * dt * v, t_curr + 0.5 * dt, i, 1)
            x = x + dt * v_cache
        else:
            raise ValueError(f"unknown solver {solver!r}: expected 'euler' or 'midpoint'")

    return x


@torch.inference_mode()
def invert(
    model,
    latent,
    cond,
    steps=50,
    gamma=0.0,
    target_noise=None,
    schedule="logsnr",
    solver="midpoint",
    fixed_point_iters=2,
    norm_match=False,
    attn=None,
    disable_tqdm=False,
):
    """Data -> noise. Integrates the flow ODE forward in t (0 -> 1).

    `solver` picks how. It must match what `sample` is given on the way back —
    "midpoint" pairs with itself, "fixed-point" pairs with sample's "euler" — since
    reconstruction comes from the two passes discretising the ODE the same way, not
    from either being accurate on its own. Mismatch them and a round trip that should
    reconstruct at 0.0187 comes back at 0.118 instead.

        solver         schedule   NFE    small-music-base   medium-base
        midpoint       logsnr     102          0.0033          0.0044   <- default
        midpoint       model      102          0.0282          0.0054
        fixed-point    model      200          0.0187          0.0247

    (Reconstruction relative error, 50 steps, both passes counted. The source clip is a
    10s rain recording, not included in the repo, so exact figures will differ.)
    "midpoint" + "logsnr" is the default on both checkpoints: half the model
    evaluations for roughly a fifth of the error, and it leaves the inverted latent
    at std 1.00 against fixed-point's 1.07, so it lands closer to the prior too.
    That last point makes `gamma` largely redundant under it — the inversion already
    arrives at the prior, so gamma > 0 mostly just costs reconstruction.

    Note the middle row. `solver` and `schedule` are a pair, not two independent
    knobs: midpoint on the "model" schedule is *worse* than the fixed-point default
    it replaces, so changing one without the other is a regression, not a trade.

    Note the schedule preference *inverts* with the solver, which is the same cliff
    argument `get_schedule` makes from the other side. Fixed-point iteration is what
    absorbs the wide interval "model" leaves next to t=0, so it is free to spend its
    steps resolving the stiff field at t=1 and "model" wins. The midpoint scheme has
    no such absorber, so that one interval sets its error floor — on small-music-base
    it plateaus at ~0.024 no matter how many steps you add — and it wants "logsnr",
    which tapers at both ends. Feed it "model" and you get the plateau; feed
    fixed-point "logsnr" and it degrades to 0.0334.

    `fixed_point_iters` applies to solver="fixed-point" only, and is what makes that
    pairing work at all. `sample` steps down from t_next to t_curr as

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

    `attn` is an optional callable announcing the identity of each model evaluation
    just before it happens, as `attn(step, stage)`. It is what `attn_inject` uses to
    key its feature cache. `step` is counted from the *noise* end, opposite to the
    loop's own index, so that inversion and sampling name the same interval by the
    same number even though they walk it in opposite directions.
    """
    dit = model.model.model
    t = get_schedule(model, steps, latent.shape[-1], latent.device, schedule).flip(0)  # ascending
    y = latent.clone()
    noise = torch.randn_like(y) if target_noise is None else target_noise
    batch = y.shape[0]
    last = len(t) - 2  # this pass runs t ascending, so step i is `last - i` from the noise end

    if solver != "fixed-point":
        # The schedule starts at exactly t=0, which the model never saw in training;
        # nudge the evaluation point off the boundary as SA3's own rk4 sampler does.
        # The midpoint solver never evaluates at t=1, where the controller blows up.
        def velocity(z, t_at, i, stage):
            if attn is not None:
                attn(last - i, stage)
            v_ctrl = (noise - z) / (1.0 - t_at).clamp(min=EPS)
            v_model = dit(z, t_at.clamp(min=1e-5).expand(batch), cfg_scale=1.0, **cond)
            return _blend(v_model, v_ctrl, gamma, norm_match)

        return _integrate(y, t, velocity, solver, "invert", disable_tqdm)

    for i, (t_curr, t_next) in enumerate(tqdm(
        list(zip(t[:-1], t[1:])), desc="invert", disable=disable_tqdm
    )):
        # Both of this step's evaluation points share one key, so the last write wins
        # — the converged fixed-point iterate at t_next, which is exactly where
        # `sample`'s Euler step will evaluate the same interval on the way back.
        if attn is not None:
            attn(last - i, 0)
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
    schedule="logsnr",
    solver="midpoint",
    norm_match=False,
    attn=None,
    disable_tqdm=False,
):
    """Noise -> data. Integrates the flow ODE (t: 1 -> 0).

    eta > 0 steers back toward `source_latent` over the [start, stop] fraction of
    the trajectory, which is what preserves the original structure while the
    prompt changes the content.

    **eta is calibrated against the solver.** It does not carry over from the old
    fixed-point default, and the difference is large enough to look like a broken
    edit rather than a weaker one. Drift from source, 10s of rain -> "a drum
    breakbeat", 50 steps, cfg 6, stop=0.9:

        eta            0.0    0.05   0.10   0.15   0.20   0.25   0.30   0.40
        midpoint      1.262  0.850  0.594  0.411  0.286  0.207  0.155  0.090
        fixed-point   1.392    —    —      —     0.680    —      —     0.429

    So eta wants to be roughly a *third* of its fixed-point value for the same
    movement: the old default of 0.4 corresponds to about 0.15 here, and 0.4 under
    midpoint returns very nearly the source. The reason is that eta pulls toward
    `source_latent`, and under a solver that already reconstructs to 0.003 there is
    no solver drift left for it to fight, so the same weight simply locks the output
    down. Midpoint also holds the output scale better — std 0.69-0.72 against
    fixed-point's 0.82-0.85, where the source itself is 0.664.

    `solver` must match the one `invert` was given: "midpoint" is its own partner,
    "euler" is the partner of inversion's "fixed-point". See `invert`.

    `attn` is the same optional evaluation-identity callback `invert` takes, and
    here the loop index already counts from the noise end, so it is passed through
    as it stands. See `attn_inject`.
    """
    dit = model.model.model
    t = get_schedule(model, steps, noise.shape[-1], noise.device, schedule)  # descending

    n_steps = len(t) - 1
    first, last = int(start * n_steps), int(stop * n_steps)

    def velocity(x, t_at, i, stage):
        if attn is not None:
            attn(i, stage)
        v_model = dit(
            x,
            t_at.expand(x.shape[0]),
            cfg_scale=cfg_scale,
            apg_scale=apg_scale,
            **cond,
        )
        if eta != 0.0 and first <= i <= last:
            v_ctrl = (x - source_latent) / t_at.clamp(min=EPS)
            return _blend(v_model, v_ctrl, eta, norm_match)
        return v_model

    return _integrate(noise, t, velocity, solver, "sample", disable_tqdm)


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
    cfg_scale=7.0,
    apg_scale=1.0,
    schedule="logsnr",
    solver="midpoint",
    fixed_point_iters=2,
    norm_match=False,
    seed=None,
    disable_tqdm=False,
):
    """Full pipeline: invert `latent` under `source_prompt`, re-sample under `target_prompt`.

    `solver` is set once here and the matched partner is chosen for each pass, since
    the pairing is the whole reason a round trip reconstructs — see `invert`. It
    defaults to "midpoint" + schedule="logsnr"; pass solver="fixed-point" with
    schedule="model" to get the older pairing back as a matched set.

    If you are porting `eta` values from the fixed-point default, scale them to
    roughly a third — see `sample`.

    Returns (edited_latent, inverted_noise).
    """
    if seed is not None:
        torch.manual_seed(seed)

    sample_solver = "euler" if solver == "fixed-point" else solver

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
        solver=solver,
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
        solver=sample_solver,
        norm_match=norm_match,
        disable_tqdm=disable_tqdm,
    )

    return edited, inverted
