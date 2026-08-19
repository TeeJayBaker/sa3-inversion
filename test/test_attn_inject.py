"""Checks for `attn_inject`. Run: `uv run python test/test_attn_inject.py [MODEL_ID]`.

The identity check is the useful one, and it is exact rather than tolerant. A
cache captured from a run and injected back into that same run puts each
`self_attn` V back exactly where it came from, so the second run has to
reproduce the first *bit for bit*. Everything the mechanism can get wrong shows
up there: a step index counted from the wrong end, the midpoint solver's two
evaluations colliding on one key, the memory-token offset, a tensor left on the
wrong device. `test_flow_edit.py` leans on the same kind of check for the same
reason.

The other structural check is the entry count. Under the midpoint solver the
predictor is only evaluated on the first step of a pass and reused thereafter,
so a window of N steps must produce exactly N stage-1 entries per layer plus one
stage-0 entry if and only if the window reaches the pass's own first step. That
number is a fingerprint of the keying scheme; nothing else lands on it.
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

from rf_inversion import invert, sample, make_cond, get_schedule
from attn_inject import FeatureCache, capture, inject, self_attn_modules, attn_edit

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "small-music-base"
AUDIO, SECONDS, STEPS = ROOT / "audio" / "loop.wav", 10.0, 20


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
    latent_len = latent.shape[-1]
    seconds_total = n / sr
    src_cond = make_cond(model, "", seconds_total, latent_len)
    tgt_cond = make_cond(model, "a drum breakbeat", seconds_total, latent_len)
    n_layers = len(self_attn_modules(model))
    print(f"{MODEL_ID} on {device}, latent {tuple(latent.shape)}, "
          f"{n_layers} layers, {STEPS} steps\n")

    failures = []

    def check(name, ok, detail):
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            failures.append(name)

    def new_cache(**kw):
        kw.setdefault("inject_steps", STEPS)
        kw.setdefault("scope", "all")
        return FeatureCache(latent_len=latent_len, **kw)

    # `attn_edit` takes a threshold in t and turns it into a step count off the
    # shared schedule; check 8 hand-wires the same window and must agree.
    t_grid = get_schedule(model, STEPS, latent_len, latent.device, "logsnr")
    window = int((t_grid[:-1] > 0.9).sum().item())

    torch.manual_seed(0)
    noise = torch.randn_like(latent)

    # 1. Capture must be a pure observer: the run it watches is unchanged.
    plain = sample(model, noise, src_cond, steps=STEPS, disable_tqdm=True)
    cache = new_cache()
    with capture(model, cache):
        watched = sample(model, noise, src_cond, steps=STEPS, attn=cache, disable_tqdm=True)
    check("capture does not perturb", torch.equal(plain, watched),
          f"max|diff| {(plain - watched).abs().max().item():.2e}, expected exactly 0")

    # 2. Injecting that cache back into the same run reproduces it exactly.
    with inject(model, cache):
        echoed = sample(model, noise, src_cond, steps=STEPS, attn=cache, disable_tqdm=True)
    check("inject own cache is a no-op", torch.equal(plain, echoed),
          f"max|diff| {(plain - echoed).abs().max().item():.2e}, expected exactly 0")

    # 3. Entry count is a fingerprint of the keying scheme. Under midpoint the
    #    predictor is evaluated once per pass, on that pass's first step, so a
    #    full-width window gets one extra stage-0 entry per layer and a narrow
    #    one — which in `sample` starts at the noise end, where the pass does
    #    begin — gets it too, while `invert` starts at the data end and does not.
    want = n_layers * (STEPS + 1)
    check("sample cache size", len(cache) == want,
          f"{len(cache)} entries, expected {n_layers} layers x ({STEPS} + 1)")

    narrow = new_cache(inject_steps=4)
    with capture(model, narrow):
        invert(model, latent, src_cond, steps=STEPS, attn=narrow, disable_tqdm=True)
    check("invert cache size", len(narrow) == n_layers * 4,
          f"{len(narrow)} entries, expected {n_layers} layers x 4 steps "
          f"(no predictor entry: invert starts at the data end)")

    # 4. The patch must not outlive its context manager.
    leaked = [m for m in self_attn_modules(model) if "apply_attn" in m.__dict__]
    after = sample(model, noise, src_cond, steps=STEPS, disable_tqdm=True)
    check("patch is restored", not leaked and torch.equal(plain, after),
          f"{len(leaked)} modules still patched, post-restore max|diff| "
          f"{(plain - after).abs().max().item():.2e}")

    # 5. A round trip that carries its own features across still reconstructs.
    #    Inversion and sampling visit almost the same points, so the cached V is
    #    almost what sampling would have computed; injecting it must therefore
    #    perturb the round trip only slightly, not derail it.
    full = new_cache()
    with capture(model, full):
        inv = invert(model, latent, src_cond, steps=STEPS, attn=full, disable_tqdm=True)
    with inject(model, full):
        rec = sample(model, inv, src_cond, steps=STEPS, attn=full, disable_tqdm=True)
    inv_plain = invert(model, latent, src_cond, steps=STEPS, disable_tqdm=True)
    rec_plain = sample(model, inv_plain, src_cond, steps=STEPS, disable_tqdm=True)
    err = ((rec - latent).norm() / latent.norm()).item()
    err_plain = ((rec_plain - latent).norm() / latent.norm()).item()
    check("full injection still reconstructs", err < 10 * err_plain + 0.02,
          f"rel-err {err:.4f} with injection vs {err_plain:.4f} without")

    # 6. The CFG wrinkle. Inversion runs at cfg 1 (batch 1 inside the DiT) and
    #    sampling at cfg 6 (batch 2, cond then uncond), so the cached tensor has
    #    to be placed into a batch it does not match. Both placements must run,
    #    stay finite, and — since they are genuinely different interventions —
    #    must not agree with each other.
    outs = {}
    for name, kw in (("both", {"cfg_half": "both"}), ("cond", {"cfg_half": "cond"}),
                     ("audio", {"scope": "audio"})):
        c = new_cache(inject_steps=window, **kw)
        if "scope" in kw:
            c.latent_len = latent_len
        with capture(model, c):
            iv = invert(model, latent, src_cond, steps=STEPS, attn=c, disable_tqdm=True)
        with inject(model, c):
            outs[name] = sample(model, iv, tgt_cond, steps=STEPS, cfg_scale=6.0,
                                attn=c, disable_tqdm=True)
    check("cfg batch mismatch handled", all(torch.isfinite(o).all() for o in outs.values()),
          "no NaN/Inf for cfg_half both/cond and scope audio")
    split = ((outs["both"] - outs["cond"]).norm() / outs["both"].norm()).item()
    check("cfg halves differ", split > 0.05,
          f"||both - cond|| / ||both|| = {split:.3f}")
    # scope='audio' leaves the 64 memory tokens alone, so it must differ from 'all'
    # but not wildly — it is the same injection over a shorter slice.
    span = ((outs["audio"] - outs["both"]).norm() / outs["both"].norm()).item()
    check("scope audio is a narrower version of the same thing", 0.01 < span < 1.5,
          f"||audio - all|| / ||all|| = {span:.3f}")

    # 7. strength is the dial: more injection must sit closer to the source.
    inv_ref = invert(model, latent, src_cond, steps=STEPS, disable_tqdm=True)
    cache6 = new_cache(inject_steps=window)
    with capture(model, cache6):
        invert(model, latent, src_cond, steps=STEPS, attn=cache6, disable_tqdm=True)
    drifts = []
    for s in (0.0, 0.5, 1.0):
        cache6.strength = s
        with inject(model, cache6):
            out = sample(model, inv_ref, tgt_cond, steps=STEPS, cfg_scale=6.0,
                         attn=cache6, disable_tqdm=True)
        drifts.append(((out - latent).norm() / latent.norm()).item())
    cache6.strength = 1.0
    # Not a monotone dial: drift saturates after the first increment and can tick
    # back up, under either cfg_half. What holds is that any injection at all pulls
    # well inside the uninjected run, which is the claim worth pinning.
    check("strength pulls toward the source", all(d < 0.8 * drifts[0] for d in drifts[1:]),
          " -> ".join(f"{d:.3f}" for d in drifts) + "  (strength 0.0, 0.5, 1.0)")
    check("strength 0 is a no-op", abs(drifts[0] - (
        (sample(model, inv_ref, tgt_cond, steps=STEPS, cfg_scale=6.0,
                disable_tqdm=True) - latent).norm() / latent.norm()).item()) < 1e-6,
        f"drift {drifts[0]:.4f} matches the uninjected run")

    # 8. The driver has to reproduce the pieces it wraps.
    edited, _ = attn_edit(
        model, latent, target_prompt="a drum breakbeat", seconds_total=seconds_total,
        steps=STEPS, t_start=0.9, cfg_scale=6.0, seed=0, disable_tqdm=True,
    )
    drift = ((edited - latent).norm() / latent.norm()).item()
    check("attn_edit moves and stays finite",
          bool(torch.isfinite(edited).all()) and 0.02 < drift < 5.0,
          f"drift {drift:.3f}, expected 0.02..5")
    check("attn_edit matches the manual route", abs(drift - drifts[-1]) < 1e-6,
          f"{drift:.4f} vs hand-wired {drifts[-1]:.4f} (t_start 0.9 -> {window} steps)")

    print()
    if failures:
        raise SystemExit(f"{len(failures)} failed: {', '.join(failures)}")
    print("all checks passed")


if __name__ == "__main__":
    main()
