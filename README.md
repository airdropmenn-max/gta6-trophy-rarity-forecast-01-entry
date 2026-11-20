# gta6-trophy-rarity-forecast-01-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/airdropmenn-max/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/airdropmenn-max/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

One-week trophy rarity forecast for GTA VI: for all 55 trophies, the
share of Exophase-tracked players expected to have earned each one by
2026-11-26T00:00:00Z, as five quantiles per trophy. Built for make.fun's
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty ($250, 20 places).

**Version 2.** The first version's bands were too narrow on the server
(rank 1,792 of 2,716, loss 0.821, coverage 0.0727): player growth was
underestimated for launch week and the refresh spread stopped at 0.8
logit when real refreshes pull rates down 1.1–1.6 logit. This version
widens the bands for those two effects; the model itself is unchanged.

## Running it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## How it works

The tracked-player count follows a power law in hours since release,
and each trophy is earned through a cohort process: players who joined
at time s have had (h - s) hours to play, and their chance of holding
the trophy climbs toward an eventual rate R with a timescale tau. The
players tracked at the first read count as the opening cohort — they
start at the rate actually observed and mature toward R. One mechanism
then explains the three things the data does: rates fall when a
refresh floods the denominator with new players, rise between
refreshes as cohorts age into trophies, and level off at R.

Each trophy's tau is chosen by grid search and its R by least squares
at every candidate tau, judged in logit space (the scale the contest
scores on) with a weak prior toward the tier's typical timescale.
Fitting uses distinct snapshots only — consecutive reads with an
unchanged rate vector count as one observation, because the source
page is frozen between refreshes.

The bands are asymmetric on the logit scale. Below the median they
carry the full refresh term; above it, half. The refresh term comes
from residual drops: how far rates fell across a player-count jump,
minus the fall the fitted model already expected, scaled by the
refreshes the horizon should contain. Fit residuals and a
random-walk floor sit underneath both sides.

## What v2 changes

- **Thin histories use a launch prior.** With fewer than 3 distinct
  player counts on record, the player slope is set by assuming the
  launch refresh is still coming — players multiply 6x (638 ->
  3,868) by the target, anchored on the latest read. A rich history
  keeps its fitted slope.
- **Wider refresh bands.** The spread cap rises 0.8 -> 2.0 logit,
  the no-jump prior drop 0.5 -> 1.2, and the spread now uses the
  median absolute residual drop plus its standard deviation (not the
  standard deviation alone). A thin history is assumed to see at
  least one refresh over the horizon.
- Nothing else moves: same cohort convolution, tau grid, tier
  priors, half-strength upside, determinism, 55-row output.

## Checks (local — the server grades, not these)

- Practice cases p00/p01: 55 rows each, ids sorted, values in range,
  quantiles non-decreasing; two runs byte-identical (deterministic);
  ~0.1 s per case against an 8 s cap.
- This revision re-authors the program in a fresh session; its
  outputs are byte-identical to the pre-revision program on p00/p01
  and on five synthetic cases covering rich history, thin history, a
  single snapshot, a no-jump history, and a player jump with falling
  rates.
- Stress test (refresh scenario: players 638 -> 3,868, earned
  unchanged): loss v1 ~1.15–1.25 -> v2 ~0.73–0.80, band coverage
  0.000 -> 0.982. Flat case (no refresh): loss 0.20 -> 0.37 — worse
  than v1 on purpose: v2 hedges for the refresh that launch week
  almost certainly brings.

Deterministic, stdlib only, no network.

## Files

- `main.ts` — the forecaster (Node 22, stdlib only)
- `LICENSE` — MIT (carried over from v1)
- `.makefun/owner.json` — entry wallet (carried over from v1)

---

<sub>Entered on [make.fun](https://make.fun/?ref=ze97ra9z&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
