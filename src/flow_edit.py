"""
FlowEdit for Stable Audio 3.

Kulikov et al., 2024 — "FlowEdit: Inversion-Free Text-Based Editing Using
Pre-Trained Flow Models" (ICCV 2025) — adapted to the SA3 rectified-flow base
checkpoints, sharing `rf_inversion`'s schedule and conditioning helpers.

The premise is that inverting to noise and re-sampling is a detour. RF-Inversion
transports the clip all the way to t = 1 and back, and everything the two passes
disagree about — solver error, prompt mismatch, the stiff field near t = 1 — lands
in the output as drift that `eta` then has to fight back. FlowEdit never leaves the
data end. It integrates an ODE directly from the source clip to the edited one,
driven by the *difference* of two velocities:

    eps ~ N(0, I)                          resampled every step
    z_src = (1 - t) * x_src + t * eps      a point on the source's own noising path
    z_tgt = z_src + delta                  the same point, carrying the edit so far
    delta <- delta + dt * (v(z_tgt, t | target) - v(z_src, t | source))

with `delta` starting at zero, so the returned `x_src + delta` starts as the input
and moves only as far as the two prompts disagree. Whatever the model would have
said about both clips — the transients, the tempo, the room — cancels in the
subtraction and is never re-synthesised, which is where the structure preservation
comes from. There is no inversion pass, nothing to reconstruct, and no accumulated
solver error to cancel: at `source_prompt == target_prompt` the velocities are
equal term for term and the output is the input to the last bit (see
`test/test_flow_edit.py`).

`z_src` is exact by construction rather than solved for, which is the other half of
the trade: it costs a fresh Gaussian every step, so unlike `invert`/`sample` this is
stochastic, and re-running with a different seed gives a different edit.

The step is plain Euler, and deliberately not the midpoint solver `rf_inversion`
offers. FireFlow's velocity reuse cannot apply here at all — eps is redrawn every
step, so the field last step's velocity belonged to no longer exists — which puts a
genuine midpoint step at 4 DiT calls against Euler's 2. It is the better solver, and
freezing eps to make the field deterministic shows it plainly: at a matched 100
calls, midpoint at 25 steps lands 0.115 from a converged reference where Euler at 50
lands 0.214, the two shrinking at first- and second-order rates respectively. That
error simply is not the one that matters. Hand the same extra budget back to the
stochastic algorithm and the spread across seeds goes

    euler,    n_avg=1,  50 calls    0.103
    midpoint, n_avg=1, 100 calls    0.099
    euler,    n_avg=2, 100 calls    0.056

so the second DiT call belongs in `n_avg`, not in a higher-order step. Worth
re-testing only if the field is ever made deterministic (a fixed eps across steps),
which would flip the ranking straight back.

Cost is 2 DiT calls per step, but only for the steps inside the edit window, which
on the noise-dense "model" schedule is a minority of them. Measured on
small-music-base, 10s, 50 steps, t_start = 0.9:

    flow_edit                             50 DiT calls    9.6s
    invert_and_edit                      200 DiT calls   24.9s
    invert_and_edit, midpoint + logsnr   102 DiT calls   14.8s

Those are costs, not verdicts. Movement away from the source is not edit quality —
the last row drifts least of the three precisely because the better inversion lets
`eta` hold on harder — so which one sounds right is a listening question, and
section 6a of the notebook sets them up side by side.
"""

import torch
from tqdm.auto import tqdm

from rf_inversion import get_schedule, make_cond


@torch.inference_mode()
def flow_edit(
    model,
    latent,
    target_prompt,
    seconds_total,
    source_prompt="",
    steps=50,
    t_start=0.9,
    t_stop=0.0,
    n_avg=1,
    src_cfg=1.8,
    tgt_cfg=7.0,
    apg_scale=1.0,
    schedule="model",
    seed=None,
    disable_tqdm=False,
):
    """Edit `latent` from `source_prompt` to `target_prompt`. No inversion pass.

    `t_start` / `t_stop` bound the region of the trajectory where the edit ODE runs,
    as thresholds in t rather than step counts. The reference implementation gates
    this with step indices (`n_max` / `n_min`), which does not survive the move to
    SA3: the "model" schedule is heavily noise-dense — 12 of 25 steps sit above
    t = 0.9 — so a step count means a completely different t here than it does on
    SD3, and it would also shift under `steps` and `schedule`. A t threshold is
    invariant to both, and reads the same way as `start`/`stop` in `sample`.

    `t_start` is the edit strength dial. delta only accumulates below it, so lower
    keeps more of the source — small-music-base, 10s of rain (not in the repo) -> "a drum breakbeat",
    50 steps, seed 0, drift being ||edit - source|| / ||source|| in latent space:

        t_start   0.5     0.6     0.7     0.8     0.9     1.0
        drift     0.085   0.109   0.135   0.178   0.283   0.534
        DiT calls 24      28      34      40      50      100

    Structure is decided at high t, so 1.0 lets the prompt rewrite the piece
    outright while 0.5 leaves a recognisable treatment of the original.

    The cost column is not a rounding artefact: on the "model" schedule 25 of those
    50 steps sit above t = 0.9, so t_start doubles as a step budget and the edit ODE
    gets fewer, coarser steps as you lower it. That is usually the direction you
    want anyway — a weaker edit needs less resolution — but it does mean t_start and
    `steps` are not independent here. "logsnr" spreads steps far more evenly in t
    (34 of 50 below 0.9) if you would rather they were; it measured slightly lower
    drift at 0.262, close enough to be a listening call rather than a settled one.

    `t_stop` > 0 hands the last stretch to plain SDEdit-style sampling under the
    target prompt alone, re-noising to t_stop first. It sharpens the result at the
    cost of source fidelity; 0 (off) is the paper's default for SD3 and is what the
    numbers above use.

    `src_cfg` / `tgt_cfg` are deliberately asymmetric — the difference between them
    is most of the edit's push, since a shared guidance component would cancel in
    the subtraction along with everything else.

    `n_avg` averages the velocity difference over that many Gaussian draws per step,
    trading DiT calls for less seed-to-seed variance. 1 is the paper's default, and
    the variance it leaves is real rather than cosmetic — three seeds at t_start=0.9
    drift 0.283 / 0.231 / 0.236, and two of those outputs differ from each other by
    0.208, which is most of the distance either one travelled from the source. Reach
    for n_avg (3 gave 0.230) when you want a stable result to tune against; otherwise
    treat the seed as part of the edit and audition a few.

    Returns the edited latent.
    """
    if seed is not None:
        torch.manual_seed(seed)

    dit = model.model.model
    latent_len, batch = latent.shape[-1], latent.shape[0]

    src_cond = make_cond(model, source_prompt, seconds_total, latent_len, batch)
    tgt_cond = make_cond(model, target_prompt, seconds_total, latent_len, batch)

    t = get_schedule(model, steps, latent_len, latent.device, schedule)  # descending
    x_src = latent
    delta = torch.zeros_like(x_src)
    tail = None  # set once the SDEdit phase starts, if it ever does

    for t_curr, t_next in tqdm(
        list(zip(t[:-1], t[1:])), desc="flowedit", disable=disable_tqdm
    ):
        if t_curr > t_start:
            continue  # above the edit window: delta stays at zero
        dt = t_next - t_curr  # < 0
        t_eval = t_curr.expand(batch)

        if t_curr > t_stop:
            v_delta = torch.zeros_like(x_src)
            for _ in range(n_avg):
                noise = torch.randn_like(x_src)
                z_src = (1.0 - t_curr) * x_src + t_curr * noise
                # z_tgt is z_src displaced by the edit so far, so the pair sits at
                # the same t on two trajectories that differ only by delta.
                z_tgt = z_src + delta
                v_src = dit(z_src, t_eval, cfg_scale=src_cfg, apg_scale=apg_scale, **src_cond)
                v_tgt = dit(z_tgt, t_eval, cfg_scale=tgt_cfg, apg_scale=apg_scale, **tgt_cond)
                v_delta += (v_tgt - v_src) / n_avg
            delta = delta + dt * v_delta
        else:
            if tail is None:
                # Re-noise the edit to t_curr and finish under the target prompt only.
                noise = torch.randn_like(x_src)
                tail = x_src + delta + t_curr * (noise - x_src)
            v = dit(tail, t_eval, cfg_scale=tgt_cfg, apg_scale=apg_scale, **tgt_cond)
            tail = tail + dt * v

    return x_src + delta if tail is None else tail
