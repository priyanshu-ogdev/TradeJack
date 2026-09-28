# Automatic Trading Logic — Review

Scope: `execution/live_inference_server.py`'s `_think()` (the model-inference step)
and `_on_update()`/`_maybe_act()` (the raw-tick and decision loop around it) — the
actual live decision-making pipeline, as distinct from the risk/composition layers
reviewed in earlier passes.

## The finding: a real train/serve skew in observation normalization

`_think()` normalized its observation like this:

```python
seq = np.stack(list(self.obs_buffer), axis=0)
mean = seq.mean(axis=0)
std = seq.std(axis=0) + 1e-8
norm = (seq - mean) / std
```

A fresh mean/std, recomputed from scratch every single decision, using only
whatever's currently in the observation window.

But `physics/lob_env.py`'s `TradeJackLOBEnv` — the actual environment the deployed
model was trained inside — normalizes differently:

```python
def _update_ema(self):
    seq = self._get_raw_sequence()
    alpha = 0.01
    batch_mean = np.mean(seq, axis=0)
    batch_var = np.var(seq, axis=0)
    self.obs_mean = (1 - alpha) * self.obs_mean + alpha * batch_mean
    self.obs_var = (1 - alpha) * self.obs_var + alpha * batch_var
    # called once per env step, i.e. once per raw tick

def _get_observation(self):
    seq = self._get_raw_sequence()
    norm_seq = (seq - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)
```

A slowly-evolving EWMA (`alpha=0.01`, so roughly a 100-tick memory horizon),
carried across the entire episode, not reset every step.

These are **fundamentally different normalizations**, not a minor numerical
discrepancy. A per-window recompute re-centers every single decision to
zero-mean/unit-variance over just the recent window — which silently throws away
exactly the kind of signal the EWMA baseline exists to preserve: genuine price
drift relative to a longer-run baseline, and the fact that a fresh volatility
spike should look *unusually large* relative to a baseline that hasn't caught up
yet, not just "normal" because the window's own std absorbed it immediately. The
deployed model was trained to interpret inputs on the EWMA's scale; feeding it
inputs on a completely different scale at inference time means it was never
actually seeing what it was trained to see, regardless of how good the RL
training itself was.

## A second, related bug I nearly introduced myself while designing the fix

The naive fix — put the EWMA update inside `_think()` — would have been wrong.
`_think()` only runs once per **decision**, and `decision_interval_ticks` can make
that far less frequent than once per raw tick (a slower "swing" sleeve might only
decide once every 300 raw depth updates). Training's `_update_ema()` runs once per
raw env `step()` — once per tick, unconditionally. Updating the EWMA inside
`_think()` would have replicated the right formula at the wrong cadence, which is
a second, different skew (the EWMA's effective memory horizon in real ticks would
shrink by a factor of `decision_interval_ticks` relative to training). Caught this
by tracing exactly where raw ticks vs. decisions happen in the code
(`_on_update()` vs. `_maybe_act()`/`_think()`) before writing anything, not after.

## The fix

- `self.obs_mean`/`self.obs_var` are now persistent state on the server,
  initialized to zeros/ones exactly matching `TradeJackLOBEnv.reset()`.
- `_update_obs_ema()` mirrors `_update_ema()`'s formula exactly (same `alpha`,
  same batch-mean/batch-var-over-current-window computation).
- It's called from `_on_update()`, immediately after a new observation is
  appended to the buffer — once per raw tick, matching training's cadence
  precisely regardless of `decision_interval_ticks`.
- `_think()` now reads the already-updated `self.obs_mean`/`self.obs_var` instead
  of recomputing anything fresh, using the identical formula
  `(seq - mean) / (sqrt(var) + eps)`.

## Verification

This is pure numpy — no torch/gymnasium/pydantic_settings needed — so it was
directly executable in this sandbox, unlike most of `live_inference_server.py` as
a whole class. Extracted both formulas verbatim (the real training-side code and
the real new inference-side code) and ran them in parallel against a synthetic
440-tick series with genuine price drift and an injected volatility spike:

- **Bit-for-bit identical** normalization output between training and the fixed
  inference code across every one of the 440 steps (max absolute difference:
  `0.0`).
- Confirmed the old (buggy) per-window formula produces measurably different
  output at the volatility spike specifically — the scenario where an EWMA
  baseline and a fresh per-window recompute are expected to diverge most.

Full existing test suite (composition layer, performance tracker, live composer —
36 tests) re-run after the change: still green, confirming no regression in the
composition-layer wiring built on top of `_think()`'s output.

What's **not** verified here, and can't be from this sandbox: the actual
end-to-end effect on a real trained model's decisions (needs `torch`/
`gymnasium`/`stable_baselines3`, none available). The claim verified is that the
*input* the model now receives at inference matches what it received at
training, exactly — not what the model does with that input, which depends on
the specific trained weights and can only be checked by actually running one.
