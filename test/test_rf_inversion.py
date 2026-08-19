"""Reconstruction checks for `rf_inversion`. Run: `uv run python test/test_rf_inversion.py [MODEL_ID]`.

Guards the numbers quoted in `invert`'s docstring, and in particular that the
solver pairing holds: a round trip only reconstructs when both passes discretise
the ODE the same way, so a mismatched pair must be visibly worse than a matched one.
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

from rf_inversion import invert, sample, make_cond

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "small-music-base"
AUDIO, SECONDS, STEPS = ROOT / "audio" / "loop.wav", 10.0, 50

NFE = [0]


def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = StableAudioModel.from_pretrained(MODEL_ID, device=device)
    sr, ds = model.model.sample_rate, model.same.downsampling_ratio

    wav, in_sr = torchaudio.load(str(AUDIO))
    available = int(wav.shape[-1] / in_sr * sr)   # SECONDS is a cap, not a target
    n = max((min(int(SECONDS * sr), available) // ds) * ds, ds)
    audio = prepare_audio(wav, in_sr=in_sr, target_sr=sr, target_length=n,
                          target_channels=2, device=device)
    audio = audio / audio.abs().max().clamp(min=1e-6)
    latent = model.same.encode(audio.to(next(model.same.parameters()).dtype))
    cond = make_cond(model, "", n / sr, latent.shape[-1])
    model.model.model.register_forward_pre_hook(
        lambda *_: NFE.__setitem__(0, NFE[0] + 1))
    print(f"{MODEL_ID} on {device}, latent {tuple(latent.shape)}, {STEPS} steps\n")

    def roundtrip(inv_solver, smp_solver, schedule):
        """Pass None for every argument to exercise the library defaults."""
        kw = {} if schedule is None else {"schedule": schedule}
        NFE[0] = 0
        inv = invert(model, latent, cond, steps=STEPS, disable_tqdm=True,
                     **({} if inv_solver is None else {"solver": inv_solver}), **kw)
        rec = sample(model, inv, cond, steps=STEPS, disable_tqdm=True,
                     **({} if smp_solver is None else {"solver": smp_solver}), **kw)
        return ((rec - latent).norm() / latent.norm()).item(), inv.std().item(), NFE[0]

    failures = []

    def check(name, ok, detail):
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            failures.append(name)

    fp_err, fp_std, fp_nfe = roundtrip("fixed-point", "euler", "model")
    mp_err, mp_std, mp_nfe = roundtrip("midpoint", "midpoint", "logsnr")
    bad_err, _, _ = roundtrip("fixed-point", "midpoint", "model")
    dflt_err, _, dflt_nfe = roundtrip(None, None, None)

    # Content-dependent, so these bounds are loose; they catch breakage, not drift.
    check("fixed-point/euler + model reconstructs", fp_err < 0.05,
          f"rel-err {fp_err:.4f} at NFE {fp_nfe}")
    check("midpoint + logsnr reconstructs", mp_err < 0.02,
          f"rel-err {mp_err:.4f} at NFE {mp_nfe}")
    check("midpoint is cheaper", mp_nfe < fp_nfe, f"NFE {mp_nfe} vs {fp_nfe}")
    check("midpoint is more accurate", mp_err < fp_err, f"{mp_err:.4f} vs {fp_err:.4f}")
    check("midpoint lands nearer the prior", abs(mp_std - 1.0) < abs(fp_std - 1.0),
          f"inverted std {mp_std:.3f} vs {fp_std:.3f}")
    check("mismatched solver pair is worse", bad_err > fp_err,
          f"fixed-point+midpoint {bad_err:.3f} vs matched {fp_err:.4f}")
    # The defaults are a matched pair; if solver and schedule ever drift apart in
    # the signatures, this is what catches it.
    check("library defaults are the good pairing", abs(dflt_err - mp_err) < 1e-6,
          f"defaults {dflt_err:.4f} at NFE {dflt_nfe}, midpoint+logsnr {mp_err:.4f}")

    # Controllers under the midpoint solver — a combination neither paper tests, so
    # nothing but these checks stands behind it.
    tgt = make_cond(model, "a drum breakbeat", n / sr, latent.shape[-1])
    inv = invert(model, latent, cond, steps=STEPS, disable_tqdm=True)
    etas = [0.0, 0.15, 0.3, 0.6]
    outs = [sample(model, inv, tgt, steps=STEPS, cfg_scale=6.0, eta=e,
                   source_latent=latent, stop=0.9, disable_tqdm=True) for e in etas]
    drifts = [((o - latent).norm() / latent.norm()).item() for o in outs]
    check("eta finite under midpoint", all(torch.isfinite(o).all() for o in outs),
          "no NaN/Inf across eta " + ", ".join(str(e) for e in etas))
    check("eta monotone under midpoint", all(a > b for a, b in zip(drifts, drifts[1:])),
          " > ".join(f"{d:.3f}" for d in drifts))
    # eta pulls toward the source, so the output scale must not run away from it.
    stds = [o.std().item() for o in outs]
    check("eta holds output scale", all(0.5 < s / latent.std().item() < 1.8 for s in stds),
          "std " + "/".join(f"{s:.2f}" for s in stds) + f" vs source {latent.std():.2f}")

    # eta=1 makes the controller the entire field, and its solution lands on the source
    # exactly — but only if it is never switched off, i.e. stop=1.0.
    full = sample(model, inv, tgt, steps=STEPS, cfg_scale=6.0, eta=1.0,
                  source_latent=latent, start=0.0, stop=1.0, disable_tqdm=True)
    err_full = ((full - latent).norm() / latent.norm()).item()
    check("eta=1 at stop=1 returns the input", err_full < 1e-4,
          f"drift {err_full:.2e}, expected ~0")

    gam = invert(model, latent, cond, steps=STEPS, gamma=0.3, disable_tqdm=True)
    check("gamma finite under midpoint", bool(torch.isfinite(gam).all()),
          f"inverted std {gam.std():.3f}")

    print()
    if failures:
        raise SystemExit(f"{len(failures)} failed: {', '.join(failures)}")
    print("all checks passed")


if __name__ == "__main__":
    main()
