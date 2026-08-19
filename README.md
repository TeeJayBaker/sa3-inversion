# Stable Audio 3 — rectified flow inversion and editing

Inversion and text-based editing on the un-post-trained rectified-flow checkpoints from
[stable-audio-3](https://github.com/Stability-AI/stable-audio-3).

- `inversion.ipynb` — notebook for editing and explanation
- `src/rf_inversion.py` — inversion and re-sampling ([RF-Inversion](https://arxiv.org/abs/2410.10792),
  stepped with the [FireFlow](https://arxiv.org/abs/2412.07517) midpoint solver)
- `src/flow_edit.py` — inversion-free editing ([FlowEdit](https://arxiv.org/abs/2412.08629))
- `src/attn_inject.py` — self-attention feature injection ([FireFlow](https://arxiv.org/abs/2412.07517))
- `test/` — sanity checks, e.g. `uv run python test/test_rf_inversion.py`

## Models

Any `-base` checkpoint works — the inversion code is model-agnostic and reads the conditioning
layout off the loaded model. Switch with `MODEL_ID` in the notebook.

| | `medium-base` | `small-music-base` | `small-sfx-base` |
|---|---|---|---|
| params | 1.4B | 459M | 459M |
| checkpoint | 9.2 GB | 2.1 GB | 2.1 GB |
| domain | music + SFX | music only | SFX only |
| s/step (M1 Pro, MPS) | 0.29 | 0.097 | — |
| recon rel-err @ 50 steps | **0.004** | 0.003 | — |
| round trip @ 50 steps, 10s audio | ~29 s | **10 s** | — |

`small-music-base` is roughly 3x faster per step for about the same reconstruction error, and it
still inverts off-domain material acceptably.

Do not use the post-trained `medium` / `small-music` / `small-sfx` checkpoints: their objective is
`rf_denoiser`, sampled by ping-pong, which re-noises with a fresh Gaussian every step. There is no
deterministic trajectory to invert.

## Setup

```bash
uv sync
uv run jupyter lab inversion.ipynb
```

## How it works

Stable Audio 3's flow convention is `x_t = (1 - t)·x_0 + t·ε`, with `t = 1` noise and `t = 0` data.
The DiT predicts the velocity `v = dx/dt = ε - x_0`, so sampling integrates `t: 1 → 0` and inversion
is the same ODE integrated the other way, `t: 0 → 1`. Both passes share one schedule so they visit
identical timesteps, and one solver — they have to discretise it the same way or the round trip
does not close.
