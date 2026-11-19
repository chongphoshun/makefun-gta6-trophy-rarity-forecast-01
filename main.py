"""gta6-trophy-rarity-forecast-01 entry: per-trophy logit growth curves.

Forecasts each GTA VI PS5 trophy's Exophase earn rate at the case target.

Model, per trophy:
  * Work on the logit scale against x = ln(1 + hours since release
    2026-11-19T00:00:00Z).  Adoption curves are roughly linear in log-time:
    story trophies climb fast early and then bend, grind/collectible
    trophies crawl.
  * Fit a weighted least-squares line logit(rate) ~ x over the case's
    history (recent reads weighted more, 24 h half-life).  Shrink the
    fitted slope toward a prior slope for the trophy's class (story /
    mission / grind / feat, from tier + description keywords) with a
    prior sd of 0.30 logit per log-hour, so a flat history (zero
    information, as on launch night) falls back to the prior instead of
    to a zero slope.
  * Anchor the forecast at the last observed rate and add
    slope * (x_target - x_last): level jumps from Exophase refreshes do
    not leak into the slope, and the trend the history actually shows
    (growth net of new-player dilution) is what extrapolates.
  * Damp growth as the rate approaches a per-class saturation cap, so
    story trophies bend toward their ceiling instead of running away.
  * Spread (honest, widening with horizon): random-walk term
    0.5*sqrt(days) as in the house baseline, plus slope uncertainty
    (posterior slope sd * dx), plus the fit's residual sd, plus a base
    0.35 logit for refresh/dilution shocks the history cannot see.

Python 3.13, standard library only, no network, deterministic.
Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release instant the log-time clock runs from (house curve baseline).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Prior slope sd (logit per unit ln(1+hours)) used to shrink fitted slopes.
PRIOR_SD = 0.30
# Base spread (logit) for refresh / dilution shocks no history can see.
BASE_SIGMA = 0.35
# Random-walk spread per sqrt(day), as in the house baseline.
RW_SIGMA = 0.5
# Residual-sd floor: rates are printed to 2 decimals, so a perfect fit is
# still not worth more than this much confidence.
RESID_FLOOR = 0.05
RESID_CAP = 0.50
# Weight half-life for history reads (hours).
HALF_LIFE_H = 24.0

GRIND_WORDS = (
    "complete 250", "complete 50", "complete 8", "complete 3",
    "win 20", "discover 100", "waink", "meditate", "hunt a unique",
    "rob 5", "rob 3", "earn $", "spend $", "purchase all",
    "register 10", "explore all", "gold medal", "complete 10",
)


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def log_hours(at: datetime) -> float:
    """ln(1 + hours since release), floored at 0 for pre-release reads."""
    h = (at - RELEASE).total_seconds() / 3600.0
    return math.log1p(max(0.0, h))


def classify(trophy: dict) -> tuple[float, float]:
    """(prior slope, saturation cap %) from tier + description."""
    tier = trophy.get("tier", "bronze")
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    text = desc + " " + name
    if tier == "platinum":
        return 0.45, 3.0
    if tier == "gold":
        if "100% completion" in desc:
            return 0.30, 8.0
        if "final mission" in desc:
            return 0.70, 25.0
        return 0.55, 20.0
    grind = any(w in text for w in GRIND_WORDS)
    if tier == "silver":
        return (0.30, 45.0) if grind else (0.45, 60.0)
    # bronze
    if "complete chapter" in desc:
        return 0.85, 92.0
    if "final mission" in desc:
        return 0.65, 75.0
    if desc.startswith("complete "):
        return 0.65, 80.0
    if grind:
        return 0.30, 40.0
    return 0.50, 75.0


def fit_slope(xs: list[float], ys: list[float], ws: list[float]) -> tuple[float, float, float]:
    """Weighted OLS slope of y on x; returns (slope, slope_se, resid_sd).

    slope_se is inf when the data carry no slope information (flat
    history, < 3 distinct x, or zero x spread).
    """
    n = len(xs)
    sw = sum(ws)
    if n < 3 or sw <= 0:
        return 0.0, math.inf, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return 0.0, math.inf, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    dof = max(1, n - 2)
    sse = sum(w * (y - (inter + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    resid = math.sqrt(sse / (sw * dof / n)) if sw > 0 else 0.0
    resid = min(RESID_CAP, max(RESID_FLOOR, resid))
    se = resid * math.sqrt(n / (sw * sxx)) if sxx > 0 else math.inf
    # Quantised rates: if the whole history spans almost nothing, the
    # slope is unidentified no matter what the arithmetic says.
    if max(ys) - min(ys) < 0.02:
        return 0.0, math.inf, 0.0
    return slope, se, resid


def forecast_trophy(trophy: dict, pos: int, history: list[dict], horizon_min: float,
                    target: datetime) -> list[float | int]:
    idx = pos  # rates are in trophies-list order, so use the position
    prior_slope, cap = classify(trophy)
    # Last non-null rate, and the series behind it.
    series = []  # (read time, rate) in history order
    for read in history:
        rates = read.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        series.append((parse_time(read["at"]), float(rates[idx])))
    if not series:
        last_rate = 0.01
        last_at = parse_time(history[-1]["at"]) if history else RELEASE
        xs, ys, ws = [], [], []
    else:
        last_at, last_rate = series[-1]
        xs = [log_hours(at) for at, _ in series]
        ys = [logit(r) for _, r in series]
        # Recent reads count more; half-life 24 h back from the last read.
        t_last = last_at.timestamp()
        ws = [0.5 ** max(0.0, (t_last - at.timestamp()) / 3600.0 / HALF_LIFE_H)
              for at, _ in series]
    slope_hat, se_hat, resid = fit_slope(xs, ys, ws)
    # Shrink toward the prior: posterior slope and sd.
    if math.isinf(se_hat):
        slope = prior_slope
        slope_sd = PRIOR_SD
    else:
        w_data = 1.0 / (se_hat * se_hat)
        w_prior = 1.0 / (PRIOR_SD * PRIOR_SD)
        slope = (slope_hat * w_data + prior_slope * w_prior) / (w_data + w_prior)
        slope_sd = math.sqrt(1.0 / (w_data + w_prior))
    x_last = log_hours(last_at)
    x_tgt = log_hours(target)
    dx = max(0.0, x_tgt - x_last)
    lg_last = logit(last_rate)
    lg_lin = lg_last + slope * dx
    p_lin = inv_logit(lg_lin)
    # Saturation: damp only the growth above the last rate, toward the cap.
    if p_lin > last_rate and last_rate < cap:
        room = cap - last_rate
        p_med = last_rate + room * (1.0 - math.exp(-(p_lin - last_rate) / room))
    else:
        p_med = p_lin
    p_med = clamp_pct(p_med)
    # Spread: random walk + slope uncertainty + fit residual + base shock.
    days = max(0.0, horizon_min) / 1440.0
    sigma = math.sqrt((RW_SIGMA * math.sqrt(days)) ** 2
                      + (slope_sd * dx) ** 2 + resid ** 2 + BASE_SIGMA ** 2)
    lg_med = logit(p_med)
    qs = []
    for z in Z:
        qs.append(round4(clamp_pct(inv_logit(lg_med + z * sigma))))
    # Keep quantiles non-decreasing after rounding.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case.get("history"):
        raise ValueError(f"{name} has no history")
    target = parse_time(case["target"])
    horizon = float(case["horizon_minutes"])
    rows = []
    for pos, trophy in enumerate(case["trophies"]):
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(trophy, pos, case["history"], horizon, target)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
