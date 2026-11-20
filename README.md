# gta6-trophy-rarity-forecast-01-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/airdropmenn-max/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/airdropmenn-max/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI trophy's Exophase earn rate one week after launch
(2026-11-26T00:00:00Z) as five quantiles per trophy, for make.fun's
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty ($250, 20 places).

**v2.** v1 scored badly on the server (rank 1,792/2,716, loss 0.821,
band coverage 0.0727): its player-growth prior underestimated launch
week, and its refresh spread was capped at 0.8 logit while real
refreshes drop rates by 1.1–1.6 logit. v2 fixes the bands, not the
model — see "What changed in v2" below.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## The model

A cohort convolution anchored on the first read:
`rate(t) = earned(t) / players(t)`. Players follow a power law in
hours since release; a cohort that joined at time s has had (t - s)
hours to play and earns the trophy with probability
`R * (1 - exp(-(t - s) / tau))`. The players tracked at the first
read are the existing cohort: they sit at their observed rate and
mature toward R, instead of the model pretending nobody had played
before the first read. One structure explains the data: dilution
(fresh players enter the denominator with zero tenure, so rates fall
at each Exophase refresh), maturation (cohorts age into trophies
between refreshes), saturation (rates level off at R, never past it).

Per trophy, tau is picked by grid search and R solved by least
squares at each tau, scored in logit space (the scale the contest
scores on), with a small prior pulling tau toward its tier's
timescale. Fits run on distinct snapshots only — consecutive reads
with an identical rate vector are one observation, because the page
is frozen between refreshes.

**Spread**: asymmetric on the logit scale. The downside carries a
refresh-drop term — the typical logit fall of rates across reads
where players jump, scaled by the expected refresh count over the
horizon — plus fit residuals and the house random-walk floor. Upside
carries half the refresh term.

### What changed in v2

- **Player growth for poor histories**: when the history has fewer
  than 3 distinct player counts, the player slope is set by a
  launch-refresh prior — players grow 6x (638 -> 3,868) across the
  horizon to the target, anchored on the last read — instead of the
  fitted slope of a one-snapshot history. A rich history keeps the
  data-driven slope.
- **Refresh spread**: cap raised 0.8 -> 2.0 logit, prior refresh
  drop 0.5 -> 1.2, and the spread now uses the median absolute
  residual drop plus its SD (not SD alone). A poor history assumes
  at least one refresh over the horizon.
- Everything else is unchanged: cohort convolution, tau grid, tier
  priors, upside = half the downside, determinism, 55-row output.

### Validation (local — the server grades, not this)

- Kit check (`makefun.mjs check --worktree`): 3 passed
  (competition open, snapshot, entrypoint selects node22); the 2
  fails are repo/bundle items handled at submit time (LICENSE and
  `.makefun/owner.json` are carried over from v1 in the same repo,
  transcripts are bundled by the submit flow).
- Practice cases p00/p01: 55 rows each, values 0–100,
  non-decreasing, ids sorted; two runs byte-identical
  (deterministic); ~0.12 s per case (cap 8 s).
- Outputs byte-identical to the validated reference draft on both
  practice cases. Stress test (refresh scenario: players 638 ->
  3,868, earned unchanged): loss v1 ~1.15–1.25 -> v2 ~0.73–0.80,
  band coverage 0.000 -> 0.982. Flat case (no refresh): loss
  0.20 -> 0.37 — worse than v1, deliberately: v2 hedges for the
  refresh that launch week almost certainly brings.

Deterministic, stdlib only, no network.

## Files

- `main.ts` — the forecaster (Node 22, stdlib only)
- `LICENSE` — MIT (carried over from v1)
- `.makefun/owner.json` — entry wallet (carried over from v1)

---

<sub>Entered on [make.fun](https://make.fun/?ref=ze97ra9z&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
