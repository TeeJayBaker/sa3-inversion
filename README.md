# Stable Audio 3 — rectified flow inversion

RF-Inversion ([Rout et al., 2024](https://arxiv.org/abs/2410.10792)) on 
the un-post-trained rectified-flow checkpoint from
[stable-audio-3](https://github.com/Stability-AI/stable-audio-3).

- `inversion.ipynb` — notebook for editing and explanation
- `rf_inversion.py` — the inversion implementation

## Models

Any `-base` checkpoint works — the inversion code is model-agnostic and reads the conditioning
layout off the loaded model. Switch with `MODEL_ID` in the notebook.

| | `medium-base` | `small-music-base` | `small-sfx-base` |
|---|---|---|---|
| params | 1.4B | 459M | 459M |
| checkpoint | 9.2 GB | 2.1 GB | 2.1 GB |
| domain | music + SFX | music only | SFX only |
| s/step (M1 Pro, MPS) | 0.29 | 0.097 | — |
| recon rel-err @ 25 steps | 0.152 | 0.153 | — |
| recon rel-err @ 50 steps | **0.018** | 0.026 | — |
| round trip @ 50 steps, 10s audio | ~30 s | **19 s** | — |

`small-music-base` is roughly 3x faster per step for about 1.4x the reconstruction error, and it
still inverts off-domain material acceptably.

Do not use the post-trained `medium` / `small-music` / `small-sfx` checkpoints: they are distilled
for few-step ping-pong sampling and do not invert cleanly.

## Setup

```bash
uv sync
uv run jupyter lab inversion.ipynb
```

## How it works

Stable Audio 3's flow convention is `x_t = (1 - t)·x_0 + t·ε`, with `t = 1` noise and `t = 0` data.
The DiT predicts the velocity `v = dx/dt = ε - x_0`, so sampling integrates `t: 1 → 0` and inversion
is the same ODE integrated the other way, `t: 0 → 1`. Both passes share one schedule so they visit
identical timesteps.
