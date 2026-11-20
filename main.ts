// Week-one trophy rarity forecaster, v2 — gta6-trophy-rarity-forecast-01
//
// Predicts, for each of GTA VI's 55 trophies, the share of Exophase-tracked
// players who will have earned it a week after launch, as five quantiles
// (5th, 25th, 50th, 75th, 95th). The contest scores mean pinball loss on
// the logit scale, so the bands matter as much as the medians.
//
// How the forecast is built. The tracked-player count grows like a power
// law in hours since release, and each trophy is earned by a cohort
// process: players who joined at time s have had (h - s) hours to play,
// and earn the trophy with a probability that rises toward an eventual
// rate R with a timescale tau. The players already tracked at the first
// read are treated as the opening cohort — they start at their observed
// rate and mature toward R — rather than pretending the game launched
// with nobody having played. That single mechanism produces the three
// patterns in the data: dilution when a refresh floods the denominator
// with brand-new players, maturation as cohorts age into trophies
// between refreshes, and saturation as rates settle at R.
//
// Fitting is per trophy: tau comes from a grid search, R from least
// squares at each candidate tau, judged in logit space with a weak prior
// that nudges tau toward its tier's typical timescale. Only distinct
// snapshots are fitted — back-to-back reads with an unchanged rate
// vector are one observation, because the source page sits frozen
// between refreshes and the plateaus would otherwise swamp the fit.
//
// The bands are asymmetric on the logit scale. The lower side carries a
// refresh term built from residual drops — how far rates actually fell
// across a player-count jump, minus the fall the fitted model already
// expected — scaled by how many refreshes the horizon should contain.
// The upper side carries half that term. Both sides also carry the
// fit residuals and a random-walk floor.
//
// Why v2 exists. v1's bands were far too narrow on the server
// (rank 1,792 of 2,716, loss 0.821, band coverage 0.0727): its player
// prior grew too slowly for launch week, and its refresh spread stopped
// at 0.8 logit while genuine refreshes drag rates down 1.1-1.6 logit.
// v2 widens the bands for exactly those two effects and changes nothing
// else: when fewer than three distinct player counts are on record, the
// player slope follows a launch prior (players multiply 6x, 638 to
// 3,868, by the target, anchored on the latest read); the refresh
// spread cap moves to 2.0 logit, the no-jump prior drop to 1.2, the
// spread uses the median absolute residual drop plus its spread, and a
// thin history is assumed to see at least one refresh.
//
// Deterministic throughout: pure functions of the case file, no clock,
// no randomness, no network. Rows come out sorted by trophy id, values
// rounded to four decimals.
//
// Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
// The grader runs node /bot/main.ts with /input and /output.

import * as fs from 'node:fs';
import * as path from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The five quantile levels, in output order. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard-normal z for each quantile level. */
const ZSCORES = [-1.645, -0.674, 0, 0.674, 1.645];
/** Random-walk width, logit per root-day. */
const RANDOM_WALK_SIGMA = 0.5;
/** Fallback slope for log players against log(1 + hours). */
const BASE_PLAYER_SLOPE = 1.0;
/** How much the player count multiplies over one launch refresh. */
const LAUNCH_MULTIPLIER = 6.0;
/** Ceiling on the player slope while the launch prior applies. */
const PLAYER_SLOPE_CAP = 3.0;
/** Ceiling on the refresh-spread term, in logit units. */
const REFRESH_SPREAD_CAP = 2.0;
/** Assumed refresh drop (logit) when no jump is visible in the history. */
const DEFAULT_REFRESH_DROP = 1.2;
/** Assumed days between refreshes when none are visible. */
const DEFAULT_REFRESH_INTERVAL_DAYS = 0.25;
/** Release date used when the case gives none usable. */
const DEFAULT_RELEASE_ISO = '2026-11-19T00:00:00Z';
/** Candidate trophy timescales, in hours. */
const TIMESCALE_GRID = [1, 1.5, 2.5, 4, 6, 10, 16, 25, 40, 65, 100, 160, 250, 400, 650, 1000];
/** Typical timescale (hours) per trophy tier, tie-breaker only. */
const TIMESCALE_PRIOR: Record<string, number> = { platinum: 250, gold: 120, silver: 60, bronze: 30 };
/** How hard the tier prior pulls in the fit score. */
const TIMESCALE_PRIOR_WEIGHT = 0.002;
/** Step size (hours) for marching the cohort integral. */
const INTEGRATION_STEP = 0.25;

interface TrophyDef { id: number; name: string; description: string; tier: string; points: number; secret: boolean; }
interface SnapshotRead { slot: string; at: string; players_tracked: number; trophies_earned: number; rates: Array<number | null>; }
interface InputCase { id: string; at: string; horizon_minutes: number; target: string; levels: number[]; game: { name: string; platform: string; release_date: string; source: string }; trophies: TrophyDef[]; history: SnapshotRead[]; notes: string; }
interface TrophyFit { median: number; residual: number; timescale: number; eventual: number; firstPlayers: number; firstRate: number; }
interface PlayerFitInput { snapshots: SnapshotRead[]; toLogHour: (iso: string) => number; toHours: (iso: string) => number; lastRead: SnapshotRead | undefined; targetHour: number; }
interface TrophyFitContext { trophyIndex: number; trophy: TrophyDef; snapshots: SnapshotRead[]; snapshotHours: number[]; startHour: number; targetHour: number; playerIntercept: number; playerSlope: number; }
interface SpreadContext { snapshots: SnapshotRead[]; snapshotHours: number[]; fits: TrophyFit[]; startHour: number; playerIntercept: number; playerSlope: number; horizonDays: number; sparseHistory: boolean; }

/** Clamp x into [lo, hi]. */
const bound = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
/** Percent (0-100) to logit, with the edges pulled inside (0.01, 99.99). */
const toLogit = (pct: number): number => {
  const p = bound(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
};
/** Logit back to percent. */
const fromLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** Round to four decimals. */
const roundTo4dp = (x: number): number => Math.round(x * 10000) / 10000;

/**
 * Ordinary least squares for y = intercept + slope * x. The residual
 * spread defaults to 0.25 when there are too few points to measure it.
 */
function linearFit(xs: number[], ys: number[]): { slope: number; intercept: number; residualSd: number } {
  const n = xs.length;
  if (n === 0) return { slope: 0, intercept: 0, residualSd: 0.25 };
  if (n === 1) return { slope: 0, intercept: ys[0], residualSd: 0.25 };
  const meanX = xs.reduce((s, v) => s + v, 0) / n;
  const meanY = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - meanX) * (xs[i] - meanX);
    sxy += (xs[i] - meanX) * (ys[i] - meanY);
  }
  const slope = sxx > 1e-12 ? sxy / sxx : 0;
  const intercept = meanY - slope * meanX;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (ys[i] - (intercept + slope * xs[i])) ** 2;
  const residualSd = n > 2 ? Math.sqrt(ss / (n - 2)) : 0.25;
  return { slope, intercept, residualSd };
}

/** Middle value (average of the two middles for even counts). */
function median(values: number[]): number {
  if (values.length === 0) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const mid = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 1 ? ordered[mid] : (ordered[mid - 1] + ordered[mid]) / 2;
}

/** Population standard deviation (divides by n). */
function stdDev(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

/**
 * Splits the model's rate at each query hour into a part that does not
 * depend on the eventual rate R and a part that scales with it:
 *
 *   rate(h) = base(h) + R * gain(h)
 *
 * with base from the opening cohort decaying at tau and gain from the
 * cohorts that joined since, marched with the trapezoid rule. The
 * cohort integral is advanced once over the query hours in ascending
 * order, so one call covers every observation plus the target.
 */
function decomposeRate(
  queryHours: number[],
  startHour: number,
  firstPlayers: number,
  firstRate: number,
  timescale: number,
  playerIntercept: number,
  playerSlope: number,
): { base: number; gain: number }[] {
  const out: { base: number; gain: number }[] = new Array(queryHours.length);
  const order = queryHours.map((h, i) => i).sort((p, q) => queryHours[p] - queryHours[q]);
  const playersAt = (h: number): number =>
    Math.exp(playerIntercept + playerSlope * Math.log1p(Math.max(0, h)));
  const playersDeriv = (h: number): number => (playersAt(h) * playerSlope) / (1 + Math.max(0, h));
  const playersAtStart = playersAt(startHour);
  let h = startHour;
  let u = 0;
  let prevF = playersDeriv(startHour);
  for (const idx of order) {
    const target = Math.max(startHour, queryHours[idx]);
    while (h < target - 1e-9) {
      const step = Math.min(INTEGRATION_STEP, target - h);
      const nextF = playersDeriv(h + step) * Math.exp((h + step - startHour) / timescale);
      u += 0.5 * (prevF + nextF) * step;
      prevF = nextF;
      h += step;
    }
    const dt = target - startHour;
    const decay = Math.exp(-dt / timescale);
    const p = playersAt(target);
    const bigK = firstPlayers * (1 - decay) + (p - playersAtStart) - decay * u;
    const base = (firstPlayers * firstRate * decay) / Math.max(1, p);
    const gain = bigK / Math.max(1, p);
    out[idx] = { base, gain };
  }
  return out;
}

/**
 * Fits the player-count power law, log players against log(1 + hours
 * since release), on the distinct snapshots. When the history is thin
 * — fewer than three distinct player counts — the launch refresh is
 * still ahead of the latest read, so the slope comes from the launch
 * prior instead: the count multiplies LAUNCH_MULTIPLIER-fold across
 * the hours left to the target, anchored on the latest read. A rich
 * history keeps the fitted slope, shrunk toward the base slope. The
 * intercept is always the mean-based one for the chosen slope.
 */
function fitPlayerGrowth(input: PlayerFitInput): { intercept: number; slope: number; residual: number; sparse: boolean } {
  const { snapshots, toLogHour, toHours, lastRead, targetHour } = input;
  const distinctCount = new Set(snapshots.map((s) => s.players_tracked)).size;
  const sparse = distinctCount < 3;
  let slope = BASE_PLAYER_SLOPE;
  let intercept = 0;
  let residual = 0.25;
  if (snapshots.length >= 2) {
    const xs = snapshots.map((s) => toLogHour(s.at));
    const ys = snapshots.map((s) => Math.log(s.players_tracked));
    const line = linearFit(xs, ys);
    if (sparse && lastRead) {
      const ahead = Math.max(1, targetHour - toHours(lastRead.at));
      slope = bound(
        Math.log(LAUNCH_MULTIPLIER) / Math.log1p(ahead),
        BASE_PLAYER_SLOPE,
        PLAYER_SLOPE_CAP,
      );
    } else {
      const distinct = distinctCount - 1;
      slope = bound((distinct * line.slope + BASE_PLAYER_SLOPE) / (distinct + 1), 0, 1.5);
    }
    const meanX = xs.reduce((s, v) => s + v, 0) / xs.length;
    const meanY = ys.reduce((s, v) => s + v, 0) / ys.length;
    intercept = meanY - slope * meanX;
    residual = line.residualSd;
  } else if (lastRead) {
    if (sparse) {
      const ahead = Math.max(1, targetHour - toHours(lastRead.at));
      slope = bound(
        Math.log(LAUNCH_MULTIPLIER) / Math.log1p(ahead),
        BASE_PLAYER_SLOPE,
        PLAYER_SLOPE_CAP,
      );
    }
    intercept = Math.log(lastRead.players_tracked) - slope * toLogHour(lastRead.at);
  }
  return { intercept, slope, residual, sparse };
}

/**
 * Fits one trophy: picks its timescale by grid search and solves its
 * eventual rate by least squares at each candidate, scored in logit
 * space with a weak pull toward the tier's typical timescale. The
 * edge cases are handled up front — no usable observations, a trophy
 * already earned by essentially everyone, or a single snapshot.
 */
function fitSingleTrophy(ctx: TrophyFitContext): TrophyFit {
  const { trophyIndex, trophy, snapshots, snapshotHours, startHour, targetHour, playerIntercept, playerSlope } = ctx;
  const obs: { hour: number; rate: number }[] = [];
  let lastRate: number | null = null;
  for (let j = 0; j < snapshots.length; j++) {
    const rate = snapshots[j].rates[trophyIndex];
    if (rate == null || !Number.isFinite(rate)) continue;
    lastRate = rate;
    obs.push({ hour: snapshotHours[j], rate });
  }
  let centerRate: number;
  let residual = 0.25;
  let timescale = TIMESCALE_PRIOR[trophy.tier] ?? 30;
  let eventual = lastRate ?? 0.5;
  const firstPlayers = snapshots.length > 0 ? snapshots[0].players_tracked : 0;
  const firstRate = obs.length > 0 ? obs[0].rate : 0.5;
  if (obs.length === 0 || lastRate == null) {
    centerRate = 0.5;
    residual = 1.0;
  } else if (lastRate >= 99.99) {
    centerRate = 99.99;
    residual = 0.05;
    eventual = 100;
  } else if (obs.length >= 2) {
    const priorTau = TIMESCALE_PRIOR[trophy.tier] ?? 30;
    const queryHours = [...obs.map((o) => o.hour), targetHour];
    let best: { timescale: number; eventual: number; score: number } | null = null;
    for (const candidate of TIMESCALE_GRID) {
      const parts = decomposeRate(queryHours, startHour, firstPlayers, firstRate, candidate, playerIntercept, playerSlope);
      let num = 0;
      let den = 0;
      for (let k = 0; k < obs.length; k++) {
        num += parts[k].gain * (obs[k].rate - parts[k].base);
        den += parts[k].gain * parts[k].gain;
      }
      const candidateR = den > 1e-12 ? bound(num / den, 0.01, 100) : lastRate;
      let ss = 0;
      for (let k = 0; k < obs.length; k++) {
        const d = toLogit(obs[k].rate) - toLogit(parts[k].base + candidateR * parts[k].gain);
        ss += d * d;
      }
      const score = ss / obs.length + TIMESCALE_PRIOR_WEIGHT * (Math.log(candidate) - Math.log(priorTau)) ** 2;
      if (best == null || score < best.score) best = { timescale: candidate, eventual: candidateR, score };
    }
    const chosen = best as { timescale: number; eventual: number; score: number };
    timescale = chosen.timescale;
    eventual = chosen.eventual;
    const partsBest = decomposeRate(queryHours, startHour, firstPlayers, firstRate, timescale, playerIntercept, playerSlope);
    let ss = 0;
    for (let k = 0; k < obs.length; k++) {
      const d = toLogit(obs[k].rate) - toLogit(partsBest[k].base + eventual * partsBest[k].gain);
      ss += d * d;
    }
    residual = Math.max(0.15, Math.sqrt(ss / Math.max(1, obs.length)) || 0.25);
    centerRate = bound(partsBest[obs.length].base + eventual * partsBest[obs.length].gain, 0.01, 99.99);
  } else {
    // One snapshot only: the timescale is the tier prior and the
    // eventual rate is the rate that was observed.
    timescale = TIMESCALE_PRIOR[trophy.tier] ?? 30;
    eventual = obs[0].rate;
    const parts = decomposeRate([obs[0].hour, targetHour], startHour, firstPlayers, firstRate, timescale, playerIntercept, playerSlope);
    centerRate = bound(parts[1].base + eventual * parts[1].gain, 0.01, 99.99);
    residual = 0.5;
  }
  return { median: centerRate, residual, timescale, eventual, firstPlayers, firstRate };
}

/**
 * Measures the refresh spread from residual drops. For every
 * player-count jump between consecutive snapshots, the observed logit
 * fall of each trophy's rate is compared with the fall the fitted
 * model already predicts; only the leftover — the part no model
 * could have foreseen — goes into the spread. The spread is the
 * median absolute leftover plus its spread, scaled by the square
 * root of the refreshes expected over the horizon. With no jumps on
 * record, the prior drop stands in instead.
 */
function measureRefreshSpread(ctx: SpreadContext): number {
  const { snapshots, snapshotHours, fits, startHour, playerIntercept, playerSlope, horizonDays, sparseHistory } = ctx;
  let jumpCount = 0;
  const residualDrops: number[] = [];
  for (let j = 1; j < snapshots.length; j++) {
    const prev = snapshots[j - 1];
    const cur = snapshots[j];
    if (cur.players_tracked <= prev.players_tracked * 1.03) continue;
    jumpCount++;
    for (let i = 0; i < fits.length; i++) {
      const ratePrev = prev.rates[i];
      const rateCur = cur.rates[i];
      if (ratePrev == null || rateCur == null) continue;
      if (!(ratePrev > 0.01 && ratePrev < 99.99 && rateCur > 0.01 && rateCur < 99.99)) continue;
      const fit = fits[i];
      if (fit.eventual >= 99.99) continue;
      const parts = decomposeRate(
        [snapshotHours[j - 1], snapshotHours[j]], startHour, fit.firstPlayers, fit.firstRate, fit.timescale, playerIntercept, playerSlope,
      );
      const predictedPrev = bound(parts[0].base + fit.eventual * parts[0].gain, 0.01, 99.99);
      const predictedCur = bound(parts[1].base + fit.eventual * parts[1].gain, 0.01, 99.99);
      const observedDrop = toLogit(ratePrev) - toLogit(rateCur);
      const predictedDrop = toLogit(predictedPrev) - toLogit(predictedCur);
      residualDrops.push(observedDrop - predictedDrop);
    }
  }
  const spanDays = snapshots.length >= 2
    ? Math.max(0.05, (Date.parse(snapshots[snapshots.length - 1].at) - Date.parse(snapshots[0].at)) / 86400000)
    : 0.05;
  const expectedRefreshes = Math.max(
    sparseHistory ? 1 : 0,
    jumpCount > 0 ? (jumpCount / spanDays) * horizonDays : horizonDays / DEFAULT_REFRESH_INTERVAL_DAYS,
  );
  if (residualDrops.length > 0) {
    return bound((median(residualDrops.map((v) => Math.abs(v))) + stdDev(residualDrops)) * Math.sqrt(expectedRefreshes), 0, REFRESH_SPREAD_CAP);
  }
  return bound(DEFAULT_REFRESH_DROP * Math.sqrt(expectedRefreshes), 0, REFRESH_SPREAD_CAP);
}

/**
 * Turns each trophy's fitted median into its five quantiles. The
 * lower side of the band is wider than the upper side — the refresh
 * term counts in full below the median and at half strength above
 * it — and both sides sit on the fit residuals, the player-fit
 * residual, and the random-walk floor. Quantiles are forced
 * non-decreasing so a wide band can never invert the order.
 */
function buildQuantileRows(
  fits: TrophyFit[],
  trophies: TrophyDef[],
  horizonDays: number,
  playerResidual: number,
  refreshSpread: number,
): { id: string; quantiles: number[] }[] {
  const randomWalk = RANDOM_WALK_SIGMA * Math.sqrt(horizonDays);
  const rows: { id: string; quantiles: number[] }[] = [];
  for (let i = 0; i < trophies.length; i++) {
    const fit = fits[i];
    const downSpread = Math.max(
      0.2,
      Math.sqrt(randomWalk * randomWalk + fit.residual * fit.residual + playerResidual * playerResidual + refreshSpread * refreshSpread),
    );
    const upSpread = Math.max(
      0.2,
      Math.sqrt(
        randomWalk * randomWalk + fit.residual * fit.residual + playerResidual * playerResidual + (0.5 * refreshSpread) ** 2,
      ),
    );
    const center = toLogit(fit.median);
    const quantiles = QUANTILES.map((level, k) => {
      const z = ZSCORES[k];
      const s = z < 0 ? downSpread : z > 0 ? upSpread : 0;
      return roundTo4dp(bound(fromLogit(center + z * s), 0.01, 99.99));
    });
    for (let k = 1; k < quantiles.length; k++) quantiles[k] = Math.max(quantiles[k], quantiles[k - 1]);
    rows.push({ id: String(trophies[i].id), quantiles });
  }
  rows.sort((a, b) => Number(a.id) - Number(b.id));
  return rows;
}

/**
 * Forecasts one case end to end: dedupe the reads into distinct
 * snapshots, fit the player growth, fit every trophy, measure the
 * refresh spread, and build the quantile rows.
 */
function buildForecast(c: InputCase): { id: string; quantiles: number[] }[] {
  const releaseMs = (() => {
    const parsed = Date.parse(c.game?.release_date ?? '');
    return Number.isFinite(parsed) ? parsed : Date.parse(DEFAULT_RELEASE_ISO);
  })();
  const hoursSinceRelease = (iso: string): number =>
    Math.max(0, (Date.parse(iso) - releaseMs) / 3600000);
  const logHours = (iso: string): number => Math.log1p(hoursSinceRelease(iso));

  const reads = (c.history ?? []).filter(
    (r) => r && Number.isFinite(r.players_tracked) && r.players_tracked > 0 && Array.isArray(r.rates),
  );
  const lastRead = reads[reads.length - 1];
  const horizonDays = Math.max(0.01, (c.horizon_minutes ?? 0) / 1440);
  const targetHour = hoursSinceRelease(c.target ?? c.at);

  // Collapse the frozen plateaus: consecutive reads with the same
  // player count and the same rate vector are a single observation.
  const snapshots: SnapshotRead[] = [];
  for (const r of reads) {
    const prev = snapshots[snapshots.length - 1];
    const duplicate = prev != null
      && prev.players_tracked === r.players_tracked
      && prev.rates.length === r.rates.length
      && prev.rates.every((v, i) => v === r.rates[i]);
    if (!duplicate) snapshots.push(r);
  }
  const snapshotHours = snapshots.map((r) => hoursSinceRelease(r.at));
  const startHour = snapshotHours.length > 0 ? snapshotHours[0] : 0;

  const players = fitPlayerGrowth({ snapshots, toLogHour: logHours, toHours: hoursSinceRelease, lastRead, targetHour });

  const fits: TrophyFit[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    fits.push(fitSingleTrophy({
      trophyIndex: i,
      trophy: c.trophies[i],
      snapshots,
      snapshotHours,
      startHour,
      targetHour,
      playerIntercept: players.intercept,
      playerSlope: players.slope,
    }));
  }

  const spread = measureRefreshSpread({
    snapshots,
    snapshotHours,
    fits,
    startHour,
    playerIntercept: players.intercept,
    playerSlope: players.slope,
    horizonDays,
    sparseHistory: players.sparse,
  });

  return buildQuantileRows(fits, c.trophies, horizonDays, players.residual, spread);
}

/** Reads one case file, forecasts it, and writes the output file. */
function processCaseFile(inputPath: string, outputPath: string): void {
  const parsed = JSON.parse(fs.readFileSync(inputPath, 'utf8')) as InputCase;
  const rows = buildForecast(parsed);
  const payload = { trophies: rows.map((r) => ({ id: Number(r.id), q: r.quantiles })) };
  fs.writeFileSync(outputPath, JSON.stringify(payload));
}

/** Pairs every input case file with its output path, in filename order. */
function listCaseJobs(): { source: string; target: string }[] {
  return fs.readdirSync(IN_DIR)
    .filter((f) => f.endsWith('.in'))
    .sort()
    .map((f) => ({ source: path.join(IN_DIR, f), target: path.join(OUT_DIR, f.replace(/\.in$/, '.out')) }));
}

function main(): void {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  for (const job of listCaseJobs()) processCaseFile(job.source, job.target);
}

main();
