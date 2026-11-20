// Week-One Rarity Forecaster v2 — gta6-trophy-rarity-forecast-01
//
// For every GTA VI trophy, forecasts its Exophase earn rate at the
// case's target as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95),
// scored by mean pinball loss on the logit scale.
//
// The model is a cohort convolution anchored on the first read:
//
//   rate(h) = [E_existing(h) + E_new(h)] / P(h)
//   E_existing(h) = P0 * [r0 * e^-((h-h0)/tau) + R * (1 - e^-((h-h0)/tau))]
//   E_new(h)      = R * ∫_{h0}^h P'(s) * (1 - e^-((h-s)/tau)) ds
//
// P(h) is the player count, a power law in hours since release. A
// cohort that joined at time s has had (h - s) hours to play by time
// h and earns the trophy with probability R * (1 - exp(-(h - s)/tau)):
// R is the trophy's eventual earn rate, tau its timescale. The
// players tracked at the first read are the existing cohort: they sit
// at their observed rate r0 and mature toward R, instead of the model
// pretending nobody had played before the first read. One structure
// explains the three things the data does:
//
//   - dilution: fresh players enter the denominator with zero tenure,
//     so rates FALL at each Exophase refresh (launch night: 638 ->
//     3,868 players, almost every rate lower);
//   - maturation: between refreshes, cohorts age into trophies and
//     rates rise, fastest for easy trophies;
//   - saturation: rates level off at R and never pass it.
//
// Per trophy, tau is picked by grid search and R solved by least
// squares at each tau, scored in logit space (the scale the contest
// scores on), with a small prior pulling tau toward its tier's
// timescale — the prior only breaks ties a flat history cannot, and
// the data overwhelms it when the history is rich.
//
// Two things the fits are careful about, because the page is frozen
// between refreshes:
//
//   - Fits run on distinct snapshots only. Consecutive reads with an
//     identical rate vector are one observation; fitting all 100+
//     reads would let the frozen plateaus dominate and inflate the
//     residuals with step-vs-smooth misfit that is an artifact of the
//     observation process, not forecast uncertainty.
//   - The refresh spread uses RESIDUAL drops: the observed logit fall
//     across a refresh minus the fall the fitted model already
//     predicts. The predictable part of a refresh (dilution from
//     known player growth) lives in the median; only the
//     unpredictable part belongs in the spread.
//
// What v2 changes, and why (v1's bands missed the refresh):
// v1 scored badly on the server (rank 1,792/2,716, loss 0.821,
// band coverage 0.0727). Its player-growth prior underestimated
// launch week, and its refresh spread was capped at 0.8 logit while
// real refreshes drop rates by 1.1-1.6 logit. So v2:
//
//   - when the history is poor (fewer than 3 distinct player
//     counts), the player slope is set by a launch-refresh prior:
//     players grow 6x (638 -> 3,868) across the horizon to the
//     target, instead of the fitted slope of a one-snapshot history;
//   - the refresh-spread cap rises from 0.8 to 2.0 logit, the prior
//     refresh drop from 0.5 to 1.2, and the spread now uses the
//     median absolute residual drop plus its SD (not SD alone);
//   - a poor history assumes at least one refresh over the horizon.
//
// The cohort convolution, the tau grid, the tier priors, and the
// asymmetric spread (upside = half the downside) are unchanged.
//
// Everything is deterministic: pure functions of the case file, no
// clock, no randomness, no network. Output rows are sorted by trophy
// id and rounded to 4 decimals.
//
// Usage (local): mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
// Grader: node /bot/main.ts with /input and /output.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels the contest scores, in row order. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** House random-walk spread, logit units per sqrt day. */
const SIGMA_RW = 0.5;
/** Prior slope of log players against log(1 + hours since release). */
const PRIOR_PLAYERS_SLOPE = 1.0;
/** Player multiplier across one launch refresh (638 -> 3,868). */
const LAUNCH_REFRESH_PLAYERS = 6.0;
/** Upper bound on the player slope while the launch prior is in force. */
const MAX_PLAYERS_SLOPE = 3.0;
/** Cap on the refresh spread term, logit units. */
const MAX_REFRESH_SPREAD = 2.0;
/** Prior refresh drop (logit) when the history shows no refresh jump. */
const PRIOR_REFRESH_DROP = 1.2;
/** Prior gap between refreshes, in days, when none are observed. */
const PRIOR_REFRESH_GAP_DAYS = 0.25;
/** Release instant used when the case names no usable release date. */
const FALLBACK_RELEASE = '2026-11-19T00:00:00Z';
/** Timescales (hours) searched per trophy. */
const TAU_GRID = [1, 1.5, 2.5, 4, 6, 10, 16, 25, 40, 65, 100, 160, 250, 400, 650, 1000];
/** Prior timescale (hours) per tier, used only to break ties. */
const TAU_PRIOR: Record<string, number> = { platinum: 250, gold: 120, silver: 60, bronze: 30 };
/** Weight of the tau prior in the fit score (a tie-breaker, not a fit). */
const TAU_PRIOR_WEIGHT = 0.002;
/** Integration step (hours) for the cohort convolution. */
const GRID_STEP = 0.25;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
  notes: string;
}

const clamp = (x: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, x));
const logit = (pct: number): number => {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;

/** Least-squares line y = a + b*x; returns slope, intercept, residual SD. */
function fitLine(xs: number[], ys: number[]): { b: number; a: number; sd: number } {
  const n = xs.length;
  if (n === 0) return { b: 0, a: 0, sd: 0.25 };
  if (n === 1) return { b: 0, a: ys[0], sd: 0.25 };
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (ys[i] - (a + b * xs[i])) ** 2;
  const sd = n > 2 ? Math.sqrt(ss / (n - 2)) : 0.25;
  return { b, a, sd };
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((p, q) => p - q);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function sdOf(values: number[]): number {
  if (values.length < 2) return 0;
  const m = values.reduce((s, v) => s + v, 0) / values.length;
  return Math.sqrt(values.reduce((s, v) => s + (v - m) ** 2, 0) / values.length);
}

/**
 * For one trophy timescale tau, the model's rate (percent) at each
 * query time, given the eventual rate R:
 *
 *   rate(h) = [P0 * r0 * e^-((h-h0)/tau) + R * K(h)] / P(h)
 *   K(h) = P0 * (1 - e^-((h-h0)/tau))
 *          + (P(h) - P(h0)) - e^-((h-h0)/tau) * U(h)
 *   U(h) = ∫_{h0}^h P'(s) * e^((s-h0)/tau) ds
 *
 * P(h) is the fitted players power law, P0 and r0 the observed
 * players and rate at the first read (time h0). The integral is
 * marched once with the trapezoid rule over the sorted query times.
 * Returns, per query time, the pair [base, k]:
 * rate = base + R * k, both already divided by P(h) so the caller
 * works in rate units directly.
 */
function rateParts(
  queryHours: number[],
  h0: number,
  p0: number,
  r0: number,
  tau: number,
  aP: number,
  bP: number,
): { base: number; k: number }[] {
  const out: { base: number; k: number }[] = new Array(queryHours.length);
  const order = queryHours.map((h, i) => i).sort((p, q) => queryHours[p] - queryHours[q]);
  const pAt = (h: number): number => Math.exp(aP + bP * Math.log1p(Math.max(0, h)));
  const pPrimeAt = (h: number): number => (pAt(h) * bP) / (1 + Math.max(0, h));
  const pH0 = pAt(h0);
  let h = h0;
  let u = 0;
  let prevF = pPrimeAt(h0) * 1; // P'(h0) * exp(0)
  for (const idx of order) {
    const target = Math.max(h0, queryHours[idx]);
    while (h < target - 1e-9) {
      const step = Math.min(GRID_STEP, target - h);
      const nextF = pPrimeAt(h + step) * Math.exp((h + step - h0) / tau);
      u += 0.5 * (prevF + nextF) * step;
      prevF = nextF;
      h += step;
    }
    const dt = target - h0;
    const decay = Math.exp(-dt / tau);
    const p = pAt(target);
    const bigK = p0 * (1 - decay) + (p - pH0) - decay * u;
    const base = (p0 * r0 * decay) / Math.max(1, p);
    const k = bigK / Math.max(1, p);
    out[idx] = { base, k };
  }
  return out;
}

interface TrophyFit {
  median: number;
  residE: number;
  tau: number;
  r: number;
  p0: number;
  r0: number;
}

function forecastCase(c: CaseFile): { id: string; q: number[] }[] {
  const releaseMs = (() => {
    const parsed = Date.parse(c.game?.release_date ?? '');
    return Number.isFinite(parsed) ? parsed : Date.parse(FALLBACK_RELEASE);
  })();
  const hoursSinceRelease = (iso: string): number =>
    Math.max(0, (Date.parse(iso) - releaseMs) / 3600000);
  const xOf = (iso: string): number => Math.log1p(hoursSinceRelease(iso));

  const reads = (c.history ?? []).filter(
    (r) => r && Number.isFinite(r.players_tracked) && r.players_tracked > 0 && Array.isArray(r.rates),
  );
  const last = reads[reads.length - 1];
  const horizonDays = Math.max(0.01, (c.horizon_minutes ?? 0) / 1440);
  const hTarget = hoursSinceRelease(c.target ?? c.at);

  // ---- Distinct snapshots: consecutive reads with an identical rate
  // vector are one observation (the page is frozen between refreshes).
  const snaps: Read[] = [];
  for (const r of reads) {
    const prev = snaps[snaps.length - 1];
    const same = prev != null
      && prev.players_tracked === r.players_tracked
      && prev.rates.length === r.rates.length
      && prev.rates.every((v, i) => v === r.rates[i]);
    if (!same) snaps.push(r);
  }
  const snapHours = snaps.map((r) => hoursSinceRelease(r.at));
  const h0 = snapHours.length > 0 ? snapHours[0] : 0;

  // ---- Players model: log players vs log(1 + hours since release),
  // fitted on the distinct snapshots. When the history is poor (fewer
  // than 3 distinct player counts), the launch refresh is still
  // ahead of the last read, so the slope is set by the launch prior:
  // players grow LAUNCH_REFRESH_PLAYERS-fold across the horizon to
  // the target, anchored on the last read. A rich history keeps the
  // data-driven slope, shrunk toward the prior.
  let aP = 0;
  let bP = PRIOR_PLAYERS_SLOPE;
  let residPlayers = 0.25;
  const distinctPlayers = new Set(snaps.map((r) => r.players_tracked)).size;
  const historyPoor = distinctPlayers < 3;
  if (snaps.length >= 2) {
    const xs = snaps.map((r) => xOf(r.at));
    const ys = snaps.map((r) => Math.log(r.players_tracked));
    const fit = fitLine(xs, ys);
    if (historyPoor && last) {
      const hoursAhead = Math.max(1, hTarget - hoursSinceRelease(last.at));
      bP = clamp(
        Math.log(LAUNCH_REFRESH_PLAYERS) / Math.log1p(hoursAhead),
        PRIOR_PLAYERS_SLOPE,
        MAX_PLAYERS_SLOPE,
      );
      aP = Math.log(last.players_tracked) - bP * xOf(last.at);
      residPlayers = Math.max(0.25, fit.sd);
    } else {
      const distinct = distinctPlayers - 1;
      bP = clamp((distinct * fit.b + PRIOR_PLAYERS_SLOPE) / (distinct + 1), 0, 1.5);
    }
    const mx = xs.reduce((s, v) => s + v, 0) / xs.length;
    const my = ys.reduce((s, v) => s + v, 0) / ys.length;
    aP = my - bP * mx;
    residPlayers = fit.sd;
  } else if (last) {
    aP = Math.log(last.players_tracked) - bP * xOf(last.at);
    if (historyPoor) {
      const hoursAhead = Math.max(1, hTarget - hoursSinceRelease(last.at));
      bP = clamp(
        Math.log(LAUNCH_REFRESH_PLAYERS) / Math.log1p(hoursAhead),
        PRIOR_PLAYERS_SLOPE,
        MAX_PLAYERS_SLOPE,
      );
      aP = Math.log(last.players_tracked) - bP * xOf(last.at);
    }
  }

  // ---- Phase 1: fit every trophy (tau by grid search, R by least
  // squares) on the distinct snapshots.
  const fits: TrophyFit[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const t = c.trophies[i];
    const obs: { h: number; rate: number }[] = [];
    let lastRate: number | null = null;
    for (let j = 0; j < snaps.length; j++) {
      const rate = snaps[j].rates[i];
      if (rate == null || !Number.isFinite(rate)) continue;
      lastRate = rate;
      obs.push({ h: snapHours[j], rate });
    }

    let median: number;
    let residE = 0.25;
    let tau = TAU_PRIOR[t.tier] ?? 30;
    let r = lastRate ?? 0.5;
    let p0 = snaps.length > 0 ? snaps[0].players_tracked : 0;
    let r0 = obs.length > 0 ? obs[0].rate : 0.5;
    if (obs.length === 0 || lastRate == null) {
      median = 0.5;
      residE = 1.0;
    } else if (lastRate >= 99.99) {
      median = 99.99;
      residE = 0.05;
      r = 100;
    } else if (obs.length >= 2) {
      const tauPrior = TAU_PRIOR[t.tier] ?? 30;
      const queryHours = [...obs.map((o) => o.h), hTarget];
      let best: { tau: number; r: number; err: number } | null = null;
      for (const cand of TAU_GRID) {
        const parts = rateParts(queryHours, h0, p0, r0, cand, aP, bP);
        let num = 0;
        let den = 0;
        for (let k = 0; k < obs.length; k++) {
          num += parts[k].k * (obs[k].rate - parts[k].base);
          den += parts[k].k * parts[k].k;
        }
        const candR = den > 1e-12 ? clamp(num / den, 0.01, 100) : lastRate;
        let ss = 0;
        for (let k = 0; k < obs.length; k++) {
          const d = logit(obs[k].rate) - logit(parts[k].base + candR * parts[k].k);
          ss += d * d;
        }
        const err = ss / obs.length + TAU_PRIOR_WEIGHT * (Math.log(cand) - Math.log(tauPrior)) ** 2;
        if (best == null || err < best.err) best = { tau: cand, r: candR, err };
      }
      const chosen = best as { tau: number; r: number; err: number };
      tau = chosen.tau;
      r = chosen.r;
      const partsBest = rateParts(queryHours, h0, p0, r0, tau, aP, bP);
      let ss = 0;
      for (let k = 0; k < obs.length; k++) {
        const d = logit(obs[k].rate) - logit(partsBest[k].base + r * partsBest[k].k);
        ss += d * d;
      }
      residE = Math.max(0.15, Math.sqrt(ss / Math.max(1, obs.length)) || 0.25);
      median = clamp(partsBest[obs.length].base + r * partsBest[obs.length].k, 0.01, 99.99);
    } else {
      // Single snapshot: tau is the tier prior, R is the observed rate.
      tau = TAU_PRIOR[t.tier] ?? 30;
      r = obs[0].rate;
      const parts = rateParts([obs[0].h, hTarget], h0, p0, r0, tau, aP, bP);
      median = clamp(parts[1].base + r * parts[1].k, 0.01, 99.99);
      residE = 0.5;
    }
    fits.push({ median, residE, tau, r, p0, r0 });
  }

  // ---- Phase 2: refresh spread from RESIDUAL drops — the observed
  // logit fall across a refresh minus the fall the fitted model already
  // predicts. Only the unpredictable part belongs in the spread.
  let jumpCount = 0;
  const residDrops: number[] = [];
  for (let j = 1; j < snaps.length; j++) {
    const prev = snaps[j - 1];
    const cur = snaps[j];
    if (cur.players_tracked <= prev.players_tracked * 1.03) continue;
    jumpCount++;
    for (let i = 0; i < c.trophies.length; i++) {
      const rPrev = prev.rates[i];
      const rCur = cur.rates[i];
      if (rPrev == null || rCur == null) continue;
      if (!(rPrev > 0.01 && rPrev < 99.99 && rCur > 0.01 && rCur < 99.99)) continue;
      const fit = fits[i];
      if (fit.r >= 99.99) continue;
      const parts = rateParts(
        [snapHours[j - 1], snapHours[j]], h0, fit.p0, fit.r0, fit.tau, aP, bP,
      );
      const predPrev = clamp(parts[0].base + fit.r * parts[0].k, 0.01, 99.99);
      const predCur = clamp(parts[1].base + fit.r * parts[1].k, 0.01, 99.99);
      const obsDrop = logit(rPrev) - logit(rCur);
      const predDrop = logit(predPrev) - logit(predCur);
      residDrops.push(obsDrop - predDrop);
    }
  }
  const spanDays = snaps.length >= 2
    ? Math.max(0.05, (Date.parse(snaps[snaps.length - 1].at) - Date.parse(snaps[0].at)) / 86400000)
    : 0.05;
  const expectedRefreshes = Math.max(
    historyPoor ? 1 : 0,
    jumpCount > 0 ? (jumpCount / spanDays) * horizonDays : horizonDays / PRIOR_REFRESH_GAP_DAYS,
  );
  const sRefresh = residDrops.length > 0
    ? clamp((medianOf(residDrops.map((v) => Math.abs(v))) + sdOf(residDrops)) * Math.sqrt(expectedRefreshes), 0, MAX_REFRESH_SPREAD)
    : clamp(PRIOR_REFRESH_DROP * Math.sqrt(expectedRefreshes), 0, MAX_REFRESH_SPREAD);

  // ---- Phase 3: asymmetric spread on the logit scale, per trophy.
  const sRw = SIGMA_RW * Math.sqrt(horizonDays);
  const rows: { id: string; q: number[] }[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const fit = fits[i];
    const sDown = Math.max(
      0.2,
      Math.sqrt(sRw * sRw + fit.residE * fit.residE + residPlayers * residPlayers + sRefresh * sRefresh),
    );
    const sUp = Math.max(
      0.2,
      Math.sqrt(
        sRw * sRw + fit.residE * fit.residE + residPlayers * residPlayers + (0.5 * sRefresh) ** 2,
      ),
    );
    const base = logit(fit.median);
    const q = LEVELS.map((tau, k) => {
      const z = Z[k];
      const s = z < 0 ? sDown : z > 0 ? sUp : 0;
      return round4(clamp(invLogit(base + z * s), 0.01, 99.99));
    });
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    rows.push({ id: String(c.trophies[i].id), q });
  }

  rows.sort((a, b) => Number(a.id) - Number(b.id));
  return rows;
}

/** Forecast one case file and write its output file. */
function runCase(inputPath: string, outputPath: string): void {
  const c = JSON.parse(readFileSync(inputPath, 'utf8')) as CaseFile;
  const rows = forecastCase(c);
  const out = { trophies: rows.map((r) => ({ id: Number(r.id), q: r.q })) };
  writeFileSync(outputPath, JSON.stringify(out));
}

function main(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const files = readdirSync(INPUT_DIR)
    .filter((f) => f.endsWith('.in'))
    .sort();
  for (const file of files) {
    runCase(join(INPUT_DIR, file), join(OUTPUT_DIR, file.replace(/\.in$/, '.out')));
  }
}

main();
