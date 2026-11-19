# Week-One Rarity Forecaster

Forecasts every GTA VI trophy's Exophase earn rate one week after launch
(2026-11-26T00:00:00Z) as five quantiles per trophy, for make.fun's
**gta6-trophy-rarity-forecast-01** bounty ($250, 20 places).

## The model

`rate(t) = earned(t) / players(t)` — fitted as two separate log-time trends,
then divided. Exophase refreshes on its own schedule: new players dilute
every rate while earned counts grow only for trophies players actually
earn, so the two sides move at different speeds and a persistence median
is wrong in both directions.

- **Players**: least-squares fit of log players vs log(1 + hours since
  release), slope shrunk toward a launch-growth prior of 1.0, so a flat
  history still forecasts growth instead of freezing the player count.
- **Earned**: per trophy, least-squares fit of log(earned + 0.5) vs
  log(1 + hours since release); earned counts are recovered exactly from
  `rate / 100 * players_tracked`. Slopes shrink toward players-slope
  times a difficulty factor (rarer trophies grow slower than the player
  base; the 100% trophy grows exactly with it).
- **Spread**: asymmetric on the logit scale. The downside carries a
  refresh-drop term — the typical logit fall of rates across reads where
  players jump, scaled by the expected refresh count over the horizon —
  plus fit residuals and the house random-walk floor. Upside carries half
  the refresh term.

Deterministic, stdlib only, no network. Run time is milliseconds per case.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Files

- `main.ts` — the forecaster (Node 22, stdlib only)
- `LICENSE` — MIT
- `.makefun/owner.json` — entry wallet
