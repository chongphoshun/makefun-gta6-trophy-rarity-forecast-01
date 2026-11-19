# GTA VI trophy rarity forecast — per-trophy logit growth curves

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/chongphoshun/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/chongphoshun/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast every GTA VI PlayStation 5 trophy's Exophase earn rate at 2026-11-26T00:00:00Z, as five quantiles per trophy. Scored by mean pinball loss on the logit scale. Prize $250; a reward needs a loss strictly below the best house entry.

## The one idea

Adoption curves are roughly linear in log-time on the logit scale, but the slope is per-trophy: story trophies climb fast and bend, grind/collectible trophies crawl. `main.py` fits, per trophy, a weighted least-squares line of `logit(rate)` against `ln(1 + hours since release)`, shrinks the slope toward a prior for the trophy's class (story / mission / grind / feat, from tier + description keywords, prior sd 0.30), anchors the forecast at the last observed rate, damps growth toward a per-class saturation cap, and sets the spread from a random-walk term plus slope uncertainty, fit residual, and a base 0.35 logit for refresh/dilution shocks. A flat history — as on launch night, when every read is identical — falls back to the prior instead of a zero slope, which is exactly what the house curve baseline cannot do.

## Validation (local)

Practice answers are not in the kit (the answering reads are the house's), so no practice loss can be computed locally; the practice cases only check format and refresh-coping. What was checked locally:

| check | result |
|---|---|
| determinism (2 runs, p00 + p01) | byte-identical outputs |
| format (55 rows, ids once, monotone quantiles, 0–100, < 65,536 bytes) | valid on both cases |
| runtime | ~0.05–0.08 s per run (limit 8 s) |
| synthetic final-like holdout (120 reads over 2.5 days, known growth curves + a dilution level shift, horizon 6,510 min) | pinball **0.252** vs house persistence baseline **0.299**; typical miss 2.36 pp vs 7.77 pp |
| same synthetic case, adverse truths | post-checkpoint dilution ×0.7: 0.320 vs 0.246 (loses); growth stalls: 0.286 vs 0.213 (loses) |

Honest read: the edge is real when the trend the history shows keeps running into the target week, and it reverses if a fresh dilution shock or a stall hits after the deadline — the fitted slope adapts to what the final history shows, but it cannot foresee a regime break. Median forecast rises ~1.25× over the last rate on the (flat) practice cases; spread is ~3× the house baseline's in log(q95/q05), which is what a refresh that moves every rate at once demands.

## Run it

Python 3.13, standard library only, no network, deterministic:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
