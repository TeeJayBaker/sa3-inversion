"""Sanity checks for `flow_edit`. Run: `uv run python test/test_flow_edit.py`.

The identity check is the useful one. With the same prompt and guidance on both
sides, `z_tgt == z_src` and the two DiT calls get byte-identical inputs, so the
velocity difference is exactly zero and `delta` never leaves zero. That makes the
expected output the *input itself*, exactly — not approximately — which is a much
sharper test than a reconstruction tolerance. A sign error, a schedule flip, a
mismatched conditioning dict or a stray noise draw all break it.
"""

import sys
from pathlib import Path

# Runnable from anywhere: the modules under test live in src/, the audio one level up.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torchaudio

from stable_audio_3.model import StableAudioModel
from stable_audio_3.inference.audio_utils import prepare_audio

from flow_edit import flow_edit

MODEL_ID = "small-music-base"
AUDIO = ROOT / "audio" / "loop.wav"
SECONDS = 10.0
STEPS = 20  # enough to exercise the loop; the identity result is step-count independent


def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")
    model = StableAudioModel.from_pretrained(MODEL_ID, device=device)
    sr, ds = model.model.sample_rate, model.same.downsampling_ratio

    wav, in_sr = torchaudio.load(str(AUDIO))
    available = int(wav.shape[-1] / in_sr * sr)   # SECONDS is a cap, not a target
    n = max((min(int(SECONDS * sr), available) // ds) * ds, ds)
    audio = prepare_audio(wav, in_sr=in_sr, target_sr=sr, target_length=n,
                          target_channels=2, device=device)
    audio = audio / audio.abs().max().clamp(min=1e-6)
    latent = model.same.encode(audio.to(next(model.same.parameters()).dtype))
    seconds_total = n / sr
    print(f"latent: {tuple(latent.shape)} ({seconds_total:.2f}s)")

    failures = []

    def check(name, ok, detail):
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            failures.append(name)

    # 1. Identity — same prompt, same guidance, so the edit must be a no-op.
    same = flow_edit(
        model, latent, target_prompt="a drum breakbeat", seconds_total=seconds_total,
        source_prompt="a drum breakbeat", steps=STEPS, t_start=1.0,
        src_cfg=6.0, tgt_cfg=6.0, seed=0, disable_tqdm=True,
    )
    err = ((same - latent).norm() / latent.norm()).item()
    check("identity (same prompt+cfg)", err < 1e-6, f"rel-err {err:.2e}, expected ~0")

    # 2. The same holds with the empty source prompt both sides, and with n_avg > 1.
    same2 = flow_edit(
        model, latent, target_prompt="", seconds_total=seconds_total, source_prompt="",
        steps=STEPS, t_start=1.0, src_cfg=1.0, tgt_cfg=1.0, n_avg=2,
        seed=0, disable_tqdm=True,
    )
    err2 = ((same2 - latent).norm() / latent.norm()).item()
    check("identity (empty prompt, n_avg=2)", err2 < 1e-6, f"rel-err {err2:.2e}, expected ~0")

    # 3. A real edit must actually move, and must stay finite.
    edited = flow_edit(
        model, latent, target_prompt="a drum breakbeat", seconds_total=seconds_total,
        source_prompt="", steps=STEPS, t_start=0.9, seed=0, disable_tqdm=True,
    )
    drift = ((edited - latent).norm() / latent.norm()).item()
    check("edit moves", 0.05 < drift < 5.0, f"drift {drift:.3f}, expected 0.05..5")
    check("edit finite", bool(torch.isfinite(edited).all()), "no NaN/Inf")

    # 4. t_start must be monotone in drift — the edit-strength dial has to work.
    drifts = []
    for ts in (0.6, 0.8, 1.0):
        out = flow_edit(
            model, latent, target_prompt="a drum breakbeat", seconds_total=seconds_total,
            source_prompt="", steps=STEPS, t_start=ts, seed=0, disable_tqdm=True,
        )
        drifts.append(((out - latent).norm() / latent.norm()).item())
    monotone = all(a < b for a, b in zip(drifts, drifts[1:]))
    check("t_start monotone", monotone,
          " < ".join(f"{d:.3f}" for d in drifts) + "  (t_start 0.6, 0.8, 1.0)")

    # 5. t_stop tail runs without blowing up.
    tailed = flow_edit(
        model, latent, target_prompt="a drum breakbeat", seconds_total=seconds_total,
        source_prompt="", steps=STEPS, t_start=0.9, t_stop=0.3, seed=0, disable_tqdm=True,
    )
    check("t_stop tail finite", bool(torch.isfinite(tailed).all()),
          f"drift {((tailed - latent).norm() / latent.norm()).item():.3f}")

    print()
    if failures:
        raise SystemExit(f"{len(failures)} failed: {', '.join(failures)}")
    print("all checks passed")


if __name__ == "__main__":
    main()
