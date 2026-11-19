# gta6-trophy-rarity-forecast-01-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/airdropmenn-max/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/airdropmenn-max/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI trophy's Exophase earn rate one week after launch
(2026-11-26T00:00:00Z) as five quantiles per trophy, for make.fun's
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty ($250, 20 places).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

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

## Files

- `main.ts` — the forecaster (Node 22, stdlib only)
- `LICENSE` — MIT
- `.makefun/owner.json` — entry wallet

---

<sub>Entered on [make.fun](https://make.fun/?ref=ze97ra9z&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
