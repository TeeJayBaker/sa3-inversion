"""
FireFlow attention feature injection for Stable Audio 3.

Deng et al., 2024 — "FireFlow: Fast Inversion of Rectified Flow for Image Semantic
Editing" (arXiv:2412.07517) — the *editing* half of the paper whose midpoint solver
`rf_inversion` already runs. It descends from RF-Edit rather than from anything in
velocity space: the inversion pass records what self-attention had to say about the
source, and the sampling pass hands those tensors back.

Self-attention is

    out = softmax(q k^T / sqrt(d)) v

so V holds the content each position collects and Q, K only decide the weights on
it. Recording V along the source's own trajectory and substituting it on the way
back,

    invert (source prompt, t: 0 -> 1)   cache  v[layer, step]
    sample (target prompt, t: 1 -> 0)   v <- (1 - strength) * v + strength * v_src

leaves the new prompt deciding where every position looks and changes what it finds
when it gets there. Nothing is added to the velocity, so unlike `eta` this never
drags the trajectory toward a fixed target — it changes what the model has to say in
the first place. Nor does it cost anything per step: the same 102 DiT calls as a
plain round trip (measured 5.1 s + 9.4 s against an unpatched 5.0 s + 9.2 s on MPS),
plus 0.7 MB of cache per layer per step, which is 113 MB for a 10 s clip over twenty
layers and eight steps.

**This is an experiment, not a port.** In images self-attention runs over space,
which is why substituting V pins layout. Here the sequence is 64 learned memory
tokens followed by one token per latent frame — 108 of them for a 10 s clip, about
10.8 frames per second — so self-attention runs over TIME, and what injection can
pin is temporal structure. No paper has tried this on an audio DiT and none of it
carried over from the image results; everything below was measured here.

Two clips, neither included in the repo, both on small-music-base at 50 steps, cfg 6,
the default midpoint +
logsnr pairing. `drift` is ||edit - source|| / ||source|| in latent space; `env_r`
the correlation between the decoded broadband RMS envelope and the source's, which
is the timing axis; `mel_d` the mean absolute log-mel difference in dB, which is the
timbre axis; `off-edit` the distance from the uncontrolled edit, whose ceiling — the
value the source itself scores — is 0.84 on the first clip. Rows are paired by drift
so that the mechanisms are compared at matched movement rather than at matched knob.

    10 s of rain -> "a drum breakbeat"    NFE   drift  env_r  mel_d  off-edit
    free edit, no control                 102   1.262   0.27   19.9    0.00
    eta 0.05                              102   0.850   0.49   14.4    0.41
    inject t_start 0.94                   102   0.837   0.57   18.2    0.81
    inject t_start 0.94, feature "kv"     102   0.830   0.74   17.5    0.87
    eta 0.10                              102   0.594   0.69   10.7    0.56
    inject t_start 0.85                   102   0.561   0.82    9.3    0.83
    eta 0.15                              102   0.411   0.84    8.0    0.67
    inject t_start 0.78                   102   0.425   0.91    6.4    0.84
    flowedit t_start 1.0                  100   0.534   0.97   15.5    0.81
    flowedit t_start 0.9                   50   0.283   0.98    7.7    0.85

On that clip injection preserves more of the source than `eta` at matched movement.
It sits above `eta`'s envelope-correlation curve at every matched drift, by 0.07 to
0.08, and the
"kv" row does it while moving the spectrum 17.5 dB where `eta` at the same drift
moves it 14.4 — more edit and more structure at once, which is the corner one is
actually aiming at. The `off-edit` column says something `eta` cannot: `eta` returns
a point part way between the free edit and the source, while injection lands nearly
as far from the free edit as the source itself does at half the drift. It relocates
rather than interpolates, which is a real structural difference between the two.

Every column above measures distance from the source. None of them measures whether
the edit did what the prompt asked, so none of this is a verdict on edit quality —
CLAP text-audio similarity is the missing axis and has not been run. Read the tables
as "how much of the source survived per unit of movement" and nothing more.

    10 s of generated drums -> "ambient    NFE   drift  env_r  mel_d  off-edit
    harmonic synth pads"
    free edit, no control                 102   1.151   0.29   25.5    0.00
    eta 0.05                              102   0.823   0.57   17.8    0.32
    inject t_start 0.94                   102   0.846   0.24   17.4    1.00
    eta 0.10                              102   0.573   0.89   12.7    0.53
    inject t_start 0.68                   102   0.579   0.82    9.4    1.00
    flowedit t_start 1.0                  100   0.913   0.40   25.8    0.46

And on that one it holds less structure than `eta` at both operating points. The
ordering reverses with the material. Two clips is two clips, but a result that
changes sign
between them is not a result you should tune against, and the same reversal runs
through every secondary knob FireFlow exposes — `layers`, `scope`, `feature`, and
"cond" against "both" — each of which either reversed its preference between the two
sources or washed out entirely.
The only settings with a stable verdict are the window (see `FeatureCache`) and
cfg_half="uncond" (diverges on both, see `_mix`).

The reason is worth knowing, because it is not the classifier-free-guidance wrinkle
that at first looks like the problem. Replaying a cache into the run that produced
it is bitwise exact, and with the *same* prompt on both passes injection is extremely
effective: an eight-step window — the width that does nothing at all under a real
edit — pulls a cfg-6 sample from drift 0.701 back to 0.301 and its envelope
correlation from 0.70 to 0.98. What breaks under an edit is that the sampling
trajectory leaves the inversion trajectory within a few steps once the prompt and the
guidance differ, so the cached V stops describing the state it is being pushed into.
Recording the cache at cfg 6 as well, so the batch matches exactly and no broadcast
is needed, changes nothing (0.331 / 0.94 against the broadcast cache's 0.299 / 0.94):
the mismatch that matters is in the trajectory, not in the batch.

So: worth having, worth auditioning, not worth believing over your ears. FlowEdit
still reaches a lower drift with a higher envelope correlation *and* a larger
spectral change for half the model calls, and nothing measured here changes that.
Reach for injection when you want the inverted latent anyway — to sweep prompts
against one fixed noise — and want the source's structure back by some route other
than `eta`'s pull toward it.

The two mechanical problems the port had to solve are documented at
`FeatureCache.__call__` (telling the midpoint solver's two evaluations apart) and
`FeatureCache._mix` (SA3's real classifier-free guidance, which FLUX being
guidance-distilled means FireFlow never had to face).
"""

import contextlib

import torch

from rf_inversion import get_schedule, invert, make_cond, sample


def self_attn_modules(model):
    """The DiT's `self_attn` Attention modules, in depth order.

    20 of them on small-music-base and 24 on medium-base, which is why every `layers`
    argument here defaults to None ("all of them") rather than to a pair of indices.
    Cross-attention is left alone, as in FireFlow: it carries the prompt, and that is
    the half of the model an edit is supposed to be changing.
    """
    for _, mod in model.model.model.named_modules():
        if type(mod).__name__ == "ContinuousTransformer":
            return [block.self_attn for block in mod.layers]
    raise RuntimeError("no ContinuousTransformer found under model.model.model")


class FeatureCache:
    """Records self-attention tensors on one pass and replays them on another.

    Configuration and storage in one object, because `capture` and `inject` have to
    agree about every field of it — the key layout above all — and splitting them
    would only invite the two passes to disagree.

    `inject_steps` is how many steps, counted from the noise end, are recorded and
    replayed. It is the edit-strength dial, the only knob here whose behaviour held
    up across both test clips, and the one to reach for first. It is also a step
    count, and therefore means nothing on its own: on the default logsnr schedule at
    50 steps, ten steps reach only t = 0.97 while twenty-five reach t = 0.50, so half
    the range does nothing and the other half does everything. Rain -> "a drum
    breakbeat", every other setting at its default:

        window     4      8     10     14     16     18     20     25
        reaches  0.994  0.983  0.973  0.933  0.897  0.843  0.769  0.500
        drift    1.247  1.198  1.150  0.837  0.684  0.561  0.425  0.176
        env_r     0.20   0.17   0.16   0.57   0.71   0.82   0.91   0.99

    The free edit is at drift 1.262, env_r 0.27, so everything left of about t = 0.95
    is not merely weak but actively worse than no injection at all: the state there
    is still essentially noise, the source's V is not yet describing any audio, and
    substituting it only perturbs. The mechanism starts working where the audio
    starts existing. `attn_edit` therefore takes a threshold in t rather than a step
    count, which is the argument `flow_edit` makes about the FlowEdit reference
    implementation's `n_max`.

    `layers` restricts injection to `range(*layers)` — FireFlow's `start_layer_index`
    / `end_layer_index`, where the paper's default is the deep half of the stack.
    Restricting the range preserves less of the source at matched movement:

        layers, 16-step window    0-20   15-20   |   20-step window   0-20   15-20
        drift                    0.684   0.799   |                   0.425   0.586
        env_r                     0.71    0.49   |                    0.91    0.66

    The right-hand pair is the sharper comparison: five layers reach drift 0.586 with
    env_r 0.66 where the whole stack passes through 0.561 with 0.82. A sweep under
    scope="audio" — 0-5, 0-10, 5-15, 10-20, 15-20 — put the shallow ranges further
    behind (env_r 0.26 to 0.44 against 0.52) and the deep ones level with the stack.

    Read that as preservation only. drift and env_r both measure distance from the
    source and neither says whether the edit did what the prompt asked, so a range
    that holds less structure may simply be editing harder. None is the default
    because it preserved most, not because the alternatives were auditioned.

    `feature` is "v" (FireFlow's `replace_v`) or "kv", which replaces the attention
    weights as well so the routing is the source's too. On rain "kv" is a consistent
    improvement — 0.74 against 0.57 of envelope correlation at a 14-step window and
    the same drift, 0.85 against 0.82 at 18 — for twice the memory. On the drum loop
    it made no difference at all (0.27 against 0.28). Worth trying, not worth
    assuming; the default stays "v" because that is what the paper does.

    `strength` interpolates between the live tensor and the cached one. At 1.0 the
    substitution is exact and bitwise, which is what makes the identity check in
    `test/test_attn_inject.py` exact rather than tolerant. It is a real dial and a
    redundant one: strength 0.5 at a 16-step window lands at drift 1.064 / env_r 0.35,
    which is where a 12-step window at full strength already sits (1.03 / 0.31).
    Widen the window and leave this alone unless you want to interpolate between two
    of them.

    `cfg_half` decides where the cached tensor goes when sampling runs under real
    classifier-free guidance and the DiT has concatenated cond and uncond along the
    batch. See `_mix`, which is where that wrinkle is actually solved.

    `scope` is "all" (every token) or "audio" (the latent frames only, skipping the
    64 memory tokens, which needs `latent_len`). "all" is the default: it matches the
    reference, needs no extra argument, and was the better of the two on rain at a
    16-step window — drift 0.684 / env_r 0.71 against 0.860 / 0.52. On the drum loop
    "audio" was ahead instead. The memory tokens are a global scratchpad rather than
    part of the signal, so that carrying the source's copy of them across helps at
    all is mildly surprising, and it should be read as a strength offset rather than
    as a difference in kind.

    `device` sends the recording somewhere other than the model's. On MPS, where
    memory is unified, it is close to free — a 25-step recording costs 5.4 s + 9.6 s
    on the CPU against 5.0 s + 9.3 s on device — so reach for it whenever the cache
    is large. Each entry is 0.7 MB on small-music-base with a 10 s clip.
    """

    def __init__(
        self,
        inject_steps,
        layers=None,
        feature="kv",
        strength=1.0,
        cfg_half="cond",
        scope="all",
        latent_len=None,
        device=None,
    ):
        if feature not in ("v", "kv"):
            raise ValueError(f"unknown feature {feature!r}: expected 'v' or 'kv'")
        if cfg_half not in ("both", "cond", "uncond"):
            raise ValueError(
                f"unknown cfg_half {cfg_half!r}: expected 'both', 'cond', or 'uncond'"
            )
        if scope not in ("audio", "all"):
            raise ValueError(f"unknown scope {scope!r}: expected 'audio' or 'all'")
        if scope == "audio" and latent_len is None:
            raise ValueError("scope='audio' needs latent_len to find the audio tokens")

        self.layers = None if layers is None else tuple(layers)
        self.inject_steps = inject_steps
        self.latent_len = latent_len
        self.feature = feature
        self.strength = strength
        self.cfg_half = cfg_half
        self.scope = scope
        self.device = device
        self.store = {}
        self._pos = None

    def __call__(self, step, stage):
        """Announce which model evaluation the next DiT call belongs to.

        This is the `attn` callback `rf_inversion.invert` and `.sample` accept, and
        the reason `_integrate` had to grow a `stage` argument at all. The default
        midpoint solver evaluates the velocity twice per step — a predictor at the
        interval's leading edge, a corrector at its midpoint — so a cache keyed on
        the step alone would have the two collide and the second silently overwrite
        the first. FireFlow hits the same problem and solves it the same way, keying
        on `str(t) + '_' + str(second_order) + '_' + str(layer_id)`.

        Keying on t itself is not available here. The two passes walk one schedule in
        opposite directions, so their midpoints agree mathematically but not
        necessarily in the last bit of a float, and a dict key has to be exact. Both
        callers instead count `step` from the noise end — `sample`'s loop index as it
        stands, `invert`'s reflected — so the two name the same interval by the same
        number and no float is ever compared.

        One evaluation per pass goes unmatched, and that is inherent to the solver
        rather than to this scheme. The midpoint velocity is reused as the next
        step's predictor, so stage 0 is only ever evaluated on a pass's very first
        step; inversion's first step is at the data end and sampling's at the noise
        end, so those are different intervals. Sampling therefore looks up one
        stage-0 key that inversion never wrote, misses it, and passes through
        uninjected. One evaluation out of a hundred and two.
        """
        self._pos = (step, stage) if step < self.inject_steps else None

    def __len__(self):
        return len(self.store)

    @property
    def nbytes(self):
        return sum(
            t.numel() * t.element_size()
            for entry in self.store.values()
            for t in entry.values()
        )

    def clear(self):
        self.store.clear()

    def _key(self, layer):
        if self._pos is None:
            return None
        if self.layers is not None and not self.layers[0] <= layer < self.layers[1]:
            return None
        return (*self._pos, layer)

    def _put(self, key, k, v):
        live = {"k": k, "v": v}
        self.store[key] = {
            name: (
                live[name].detach().clone()
                if self.device is None
                else live[name].detach().to(self.device)
            )
            for name in self.feature
        }
        return k, v

    def _take(self, key, k, v):
        cached = self.store.get(key)
        if not cached or self.strength == 0.0:
            return k, v
        # `feature` is read here rather than at capture time, so one recording can be
        # replayed under a narrower setting without being taken again.
        if "k" in self.feature and "k" in cached:
            k = self._mix(k, cached["k"])
        if "v" in self.feature and "v" in cached:
            v = self._mix(v, cached["v"])
        return k, v

    def _mix(self, cur, cached):
        """Write `cached` into `cur` over the chosen CFG rows and token positions.

        The classifier-free-guidance wrinkle lives here and has no counterpart in
        FireFlow. FLUX.1-dev is guidance-distilled — guidance is a scalar through an
        MLP embedder — so their batch is 1 in both directions and the cached tensor
        drops straight back in. SA3 does the real thing: at `cfg_scale != 1.0` the
        DiT concatenates cond and uncond along the batch dimension (`models/dit.py`,
        `batch_inputs = torch.cat([x, x], dim=0)`, conditional first). Inversion runs
        at cfg 1 and sampling at cfg 6, so a batch-1 recording meets a batch-2 tensor
        and something has to decide where it goes.

        Write the guided velocity as v = v_uncond + s (v_cond - v_uncond), that is
        (1 - s) v_uncond + s v_cond. A perturbation of the conditional half alone
        therefore reaches the output at weight s = 6; of the unconditional half
        alone, at weight 1 - s = -5, amplified *and* sign-flipped; of both, at weight
        1. Broadcasting across both halves is thus the only placement that applies
        the injection at the strength it was recorded, and the measurement agrees —
        rain, 16-step window, scope="audio":

            cfg_half   drift   env_r   mel_d   std
            both       0.860    0.52    12.3   0.81
            cond       0.769    0.50    15.4   0.71
            uncond     2.679    0.15    40.3   1.79

        (The source is at std 0.66.) "uncond" diverges, which is the -5 arriving as
        promised, and it does so on both test clips. "both" is the unit-weight
        placement and never diverges. "cond" is the same intervention at six times
        the strength, and on this clip the metrics prefer "both" — at drift 0.769 a
        wider window reaches env_r 0.62 where "cond" reaches 0.50.

        "cond" is nonetheless the default, chosen by ear over these numbers. It won
        on the drum loop too. Since the metrics here measure preservation only and
        say nothing about whether the edit did what the prompt asked, they are the
        weaker evidence of the two.

        Worth knowing before spending effort here: the batch mismatch is not what
        limits the method. Recording the cache at cfg 6 as well, so that no broadcast
        is needed at all, lands at drift 0.331 / env_r 0.94 against the broadcast
        cache's 0.299 / 0.94. See the module docstring for what does limit it.

        The other choice made here is `scope`. Positions 0-63 are the transformer's
        learned memory tokens and 64 onward the audio frames; "audio" writes only the
        tail. See the class docstring for why "all" is the default anyway.
        """
        cached = cached.to(device=cur.device, dtype=cur.dtype)
        batch, src, seq = cur.shape[0], cached.shape[0], cur.shape[2]
        if batch == src:
            rows = [slice(0, batch)]
        elif batch == 2 * src:
            rows = {
                "both": [slice(0, src), slice(src, batch)],
                "cond": [slice(0, src)],
                "uncond": [slice(src, batch)],
            }[self.cfg_half]
        else:
            raise ValueError(
                f"cached batch {src} fits neither the live batch {batch} nor half of it"
            )
        first = 0 if self.scope == "all" else seq - self.latent_len
        out = cur.clone()
        for rng in rows:
            if self.strength == 1.0:
                # Assign rather than lerp: a + 1.0 * (b - a) is not bitwise b, and
                # that exactness is what `test/test_attn_inject.py` checks against.
                out[rng, :, first:] = cached[:, :, first:]
            else:
                out[rng, :, first:] = torch.lerp(
                    cur[rng, :, first:], cached[:, :, first:], self.strength
                )
        return out


def _wrap(module, layer, cache, hook):
    orig = module.apply_attn  # the bound class method; the instance dict is still empty

    def patched(q, k, v, **kwargs):
        key = cache._key(layer)
        if key is not None:
            k, v = hook(key, k, v)
        return orig(q, k, v, **kwargs)

    return patched


@contextlib.contextmanager
def _patch(model, cache, hook):
    """Swap `apply_attn` on the chosen self-attention modules for the duration.

    `Attention.apply_attn` is the single choke point every attention path in SA3 goes
    through — flash, varlen, flex and plain SDPA are branches inside it, and all
    three call sites in `Attention.forward` route through it — so one interception
    there covers every case, and it sees q, k and v already projected, head-split,
    qk-normalised and rotated. It is a method rather than a module, which is why this
    is a monkeypatch and not a `register_forward_pre_hook`: there is no submodule to
    hook.
    """
    modules = list(enumerate(self_attn_modules(model)))
    if cache.layers is not None:
        modules = modules[slice(*cache.layers)]
    # Whatever was on the instance before, including another patch: put it back on the
    # way out rather than clearing the slot, so nesting cannot silently undo it.
    saved = [module.__dict__.get("apply_attn") for _, module in modules]
    try:
        for layer, module in modules:
            module.apply_attn = _wrap(module, layer, cache, hook)
        yield cache
    finally:
        for (_, module), was in zip(modules, saved):
            if was is None:
                module.__dict__.pop("apply_attn", None)
            else:
                module.apply_attn = was


def capture(model, cache, clear=True):
    """Record self-attention features from every DiT call made inside the block.

    Pass the same cache to `invert` as its `attn` argument, so that it learns which
    evaluation each call belongs to:

        with capture(model, cache):
            inverted = invert(model, latent, src_cond, steps=50, attn=cache)

    Recording is a pure observation — the pass it watches returns bitwise what it
    would have returned unwatched, which `test/test_attn_inject.py` checks — so an
    inversion is worth capturing even when you are not sure you will inject it.
    """
    if clear:
        cache.clear()
    return _patch(model, cache, cache._put)


def inject(model, cache):
    """Substitute recorded features back into every DiT call made inside the block.

        with inject(model, cache):
            edited = sample(model, inverted, tgt_cond, steps=50, cfg_scale=6.0,
                            attn=cache)

    The step count and schedule have to match the ones the cache was recorded under,
    since the keys are step indices. Nothing checks this, and a mismatch degrades to
    silent cache misses rather than to an error, which is why `attn_edit` takes one
    `steps` and one `schedule` for both passes.
    """
    return _patch(model, cache, cache._take)


def attn_edit(
    model,
    latent,
    target_prompt,
    seconds_total,
    source_prompt="",
    steps=50,
    t_start=0.9,
    layers=None,
    strength=1.0,
    feature="kv",
    cfg_half="cond",
    scope="all",
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
    """Invert under `source_prompt`, re-sample under `target_prompt`, and carry the
    source's self-attention features across between the two passes.

    Same shape and same cost as `rf_inversion.invert_and_edit` — this is that
    pipeline with a recorder around the first pass and a substitution in the second.
    `eta` still works and composes with it; leave it at 0 to see what injection does
    on its own.

    `t_start` is the edit-strength dial and the argument to reach for first. Features
    are recorded and replayed over the part of the trajectory above it, so *lower*
    keeps more of the source — the opposite sense to `eta`, and the same sense as
    `flow_edit`'s argument of the same name. Rain -> "a drum breakbeat", drift being
    ||edit - source|| / ||source|| and env_r the envelope correlation with the source:

        t_start  0.975   0.96   0.94   0.90   0.85   0.78   0.68   0.51
        window      10     12     14     16     18     20     22     25
        drift    1.150  1.031  0.837  0.684  0.561  0.425  0.322  0.176
        env_r     0.16   0.31   0.57   0.71   0.82   0.91   0.96   0.99

    The uncontrolled edit is at drift 1.262 with env_r 0.27, so above about 0.95 this
    is worse than doing nothing — there is no audio in the state yet for the cached V
    to describe, and substituting it only perturbs. The useful range is roughly 0.75
    to 0.95 and the default is 0.9. See `FeatureCache` for why this is a threshold in
    t rather than the step count FireFlow's `inject_step` counts.

    `steps` is shared by both passes rather than split into `inversion_steps` and
    `sampling_steps` the way `invert_and_edit` splits it. The cache is keyed by step
    index, so two different step counts would key the same interval differently and
    every lookup would miss; one number is the honest signature.

    The remaining arguments configure the cache and are documented on `FeatureCache`
    — `layers` (None preserved most of the source; no range was judged by ear), `feature`,
    `strength`, `cfg_half` (the alternatives and what the metrics said about them
    are in `FeatureCache._mix`), `scope`. Everything from `gamma` onward passes straight
    through to `invert` and `sample` and means what it means there.

    Returns (edited_latent, inverted_noise), as `invert_and_edit` does.
    """
    if seed is not None:
        torch.manual_seed(seed)

    latent_len, batch_size = latent.shape[-1], latent.shape[0]
    # `t_start` is a threshold in t; the cache needs the step count that reaches it.
    # Both passes share this schedule, so both derive the same window.
    t = get_schedule(model, steps, latent_len, latent.device, schedule)
    inject_steps = int((t[:-1] > t_start).sum().item())
    cache = FeatureCache(
        inject_steps=inject_steps,
        layers=layers,
        feature=feature,
        strength=strength,
        cfg_half=cfg_half,
        scope=scope,
        latent_len=latent_len,
    )

    src_cond = make_cond(model, source_prompt, seconds_total, latent_len, batch_size)
    with capture(model, cache):
        inverted = invert(
            model,
            latent,
            src_cond,
            steps=steps,
            gamma=gamma,
            schedule=schedule,
            solver=solver,
            fixed_point_iters=fixed_point_iters,
            norm_match=norm_match,
            attn=cache,
            disable_tqdm=disable_tqdm,
        )

    tgt_cond = make_cond(model, target_prompt, seconds_total, latent_len, batch_size)
    with inject(model, cache):
        edited = sample(
            model,
            inverted,
            tgt_cond,
            steps=steps,
            cfg_scale=cfg_scale,
            apg_scale=apg_scale,
            eta=eta,
            source_latent=latent,
            start=start,
            stop=stop,
            schedule=schedule,
            solver="euler" if solver == "fixed-point" else solver,
            norm_match=norm_match,
            attn=cache,
            disable_tqdm=disable_tqdm,
        )

    return edited, inverted
