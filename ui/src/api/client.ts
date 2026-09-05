/**
 * Typed HTTP client.
 *
 * The types come from `schema.d.ts`, generated from FastAPI's OpenAPI
 * schema: not one field is written by hand here. If the API renames
 * something, the frontend stops compiling instead of showing `undefined`
 * on screen.
 */
import type { components } from "./schema";

type S = components["schemas"];

export type SymbolSpec = S["SymbolSpecOut"];
export type SymbolList = S["SymbolListOut"];
export type Coverage = S["CoverageOut"];
export type Strategy = S["StrategyOut"];
export type RunConfigIn = S["RunConfigIn"];
export type EdgeReport = S["EdgeResponse"];
export type EdgeStat = S["EdgeStatOut"];
export type BreakevenPrior = S["BreakevenPriorOut"];
export type Breakeven = S["BreakevenOut"];
export type BacktestResponse = S["BacktestResponse"];
export type RunSummary = S["RunSummaryOut"];
export type RunDetail = S["RunDetailOut"];
export type Metrics = S["MetricsOut"];
export type Execution = S["ExecutionOut"];
export type Equity = S["EquityOut"];
export type EquityPoint = S["EquityPoint"];
export type Trades = S["TradesOut"];
export type Trade = S["TradeOut"];
export type CompareResponse = S["CompareResponse"];
export type CompareMetricRow = S["CompareMetricRow"];
export type CompareConfigRow = S["CompareConfigRow"];
export type ValidateResponse = S["ValidateResponse"];
export type Vocabulary = S["VocabularyOut"];
export type IndicatorDef = S["IndicatorOut"];
export type ParamDef = S["ParamOut"];
export type SaveStrategyResponse = S["SaveStrategyResponse"];
export type Preview = S["PreviewResponse"];
export type AttemptsPanel = S["AttemptsPanelOut"];
export type RunStatus = S["RunSummaryOut"]["status"];

export type WalkForwardResponse = S["WalkForwardResponse"];
export type WalkForwardWindow = S["WalkForwardWindowOut"];
export type PermutationResponse = S["PermutationResponse"];
export type PermutationTest = S["PermutationTestOut"];
export type MultipleTestingResponse = S["MultipleTestingResponse"];
export type TickResolveResponse = S["TickResolveResponse"];
export type BatchResponse = S["BatchResponse"];
export type BatchRunOut = S["BatchRunOut"];
export type ConsistencyOut = S["ConsistencyOut"];

export type ScreenJob = S["ScreenJobOut"];
export type ScreenReport = S["ScreenReportOut"];
export type ScreenCell = S["ScreenCellOut"];
export type TrialPanelOut = S["TrialPanelOut"];
export type Uncertainty = S["UncertaintyOut"];
export type Gates = S["GatesOut"];
export type GateRow = S["GateRowOut"];
export type AmbiguityPrior = S["AmbiguityPriorOut"];
export type Tradability = S["TradabilityOut"];
export type TradabilityCell = S["TradabilityCellOut"];

export type LiveSession = S["LiveSessionOut"];
export type Progress = S["ProgressOut"];
export type DownloadJob = S["DownloadJobOut"];
export type LiveDetail = S["LiveDetailOut"];
export type LiveEvent = S["LiveEventOut"];
export type LiveTrade = S["LiveTradeOut"];
export type LiveComparison = S["LiveComparisonOut"];
export type TradeDeviation = S["TradeDeviationOut"];
export type UnmatchedTrade = S["UnmatchedTradeOut"];

/** A grid is a free-form map of dotted spec paths to the values to try. */
export type ParameterGrid = Record<string, unknown[]>;

/** Error carrying the sentence the API wrote for a human. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      0,
      "The server is not responding. Check that the backend is running (python run.py).",
    );
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        // FastAPI validation errors: field + reason
        detail = body.detail
          .map((item) => {
            const entry = item as { loc?: unknown[]; msg?: string };
            const where = (entry.loc ?? []).slice(1).join(".");
            return where ? `${where}: ${entry.msg}` : String(entry.msg);
          })
          .join("; ");
      }
    } catch {
      /* the body was not JSON: the status is all that is left */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  symbols: () => request<SymbolList>("/api/symbols"),
  coverage: (symbol: string, timeframe: string) =>
    request<Coverage>(
      `/api/symbols/${encodeURIComponent(symbol)}/coverage?timeframe=${timeframe}`,
    ),
  strategies: () => request<Strategy[]>("/api/strategies"),
  validate: (spec: unknown) => post<ValidateResponse>("/api/strategies/validate", { spec }),

  // The editor renders what the registry declares and nothing else: a second
  // copy of this list in the frontend is how a spec becomes valid on screen
  // and invalid on the server.
  vocabulary: () => request<Vocabulary>("/api/vocabulary"),
  saveStrategy: (spec: unknown, overwrite = false) =>
    post<SaveStrategyResponse>("/api/strategies", { spec, overwrite }),

  // What the strategy is about to cost, before any backtest is run.
  preview: (body: { spec?: unknown; strategy_id?: string; config: RunConfigIn }) =>
    post<Preview>("/api/strategies/preview", body),

  edge: (body: { strategy_id: string; config: RunConfigIn; horizons?: number[] }) =>
    post<EdgeReport>("/api/edge", body),
  // `spec` sends an unsaved strategy straight from the editor. It is counted
  // in the campaign like every other attempt: there is deliberately no path
  // that runs a backtest without registering it.
  backtest: (body: {
    strategy_id?: string;
    spec?: unknown;
    config: RunConfigIn;
    force?: boolean;
    progress_token?: string;
  }) => post<BacktestResponse>("/api/backtest", body),

  runs: (params: { symbol?: string; strategy_id?: string; limit?: number } = {}) => {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") query.set(key, String(value));
    }
    const suffix = query.toString();
    return request<RunSummary[]>(`/api/runs${suffix ? `?${suffix}` : ""}`);
  },
  run: (runId: string) => request<RunDetail>(`/api/runs/${runId}`),
  equity: (runId: string, points = 2000) =>
    request<Equity>(`/api/runs/${runId}/equity?points=${points}`),
  trades: (runId: string, offset: number, limit: number, ambiguousOnly = false) =>
    request<Trades>(
      `/api/runs/${runId}/trades?offset=${offset}&limit=${limit}` +
        (ambiguousOnly ? "&ambiguous_only=true" : ""),
    ),
  compare: (runIds: string[], points = 1500) =>
    post<CompareResponse>("/api/runs/compare", { run_ids: runIds, points }),
  deleteRun: (runId: string) =>
    request<{ run_id: string; deleted: boolean }>(`/api/runs/${runId}`, { method: "DELETE" }),

  // -- validation --------------------------------------------------------

  walkForward: (body: {
    progress_token?: string;
    run_id: string;
    mode?: "rolling" | "anchored";
    train_days?: number;
    test_days?: number;
    min_train_trades?: number;
    grid?: ParameterGrid | null;
  }) => post<WalkForwardResponse>("/api/validation/walkforward", body),

  permutation: (body: {
    progress_token?: string;
    run_id: string;
    iterations?: number;
    tests?: ("random_entries" | "permuted_returns")[];
    block_bars?: number;
    seed?: number;
  }) => post<PermutationResponse>("/api/validation/permutation", body),

  multipleTesting: (body: { run_id: string }) =>
    post<MultipleTestingResponse>("/api/validation/multiple-testing", body),

  tickResolve: (body: { run_id: string; progress_token?: string }) =>
    post<TickResolveResponse>("/api/validation/tick-resolve", body),

  // -- batch -------------------------------------------------------------

  batch: (body: {
    strategy_id: string;
    symbols: string[];
    config: RunConfigIn;
    max_workers?: number;
    progress_token?: string;
  }) => post<BatchResponse>("/api/batch", body),

  // -- screening ---------------------------------------------------------

  // A campaign is minutes of work: it starts a job and the page polls it.
  screen: (body: {
    strategy_ids: string[];
    symbols: string[];
    timeframes: string[];
    config: RunConfigIn;
    min_trades?: number;
    permutation_iterations?: number;
  }) => post<ScreenJob>("/api/screen", body),
  screenJob: (jobId: string) => request<ScreenJob>(`/api/screen/${jobId}`),

  // Stage zero of the funnel: what the broker's spread makes untestable.
  tradability: (params: { symbols?: string[]; timeframes?: string[] } = {}) => {
    const query = new URLSearchParams();
    if (params.symbols?.length) query.set("symbols", params.symbols.join(","));
    if (params.timeframes?.length) query.set("timeframes", params.timeframes.join(","));
    const suffix = query.toString();
    return request<Tradability>(`/api/tradability${suffix ? `?${suffix}` : ""}`);
  },

  // -- the live runner ---------------------------------------------------

  // The backend does not run the runner: it reads the diaries that
  // `scripts.run_live` writes, so opening this page cannot disturb a
  // strategy that is trading.
  liveSessions: () => request<LiveSession[]>("/api/live"),
  liveSession: (sessionId: string, events = 100) =>
    request<LiveDetail>(
      `/api/live/${encodeURIComponent(sessionId)}?events=${events}`,
    ),
  liveComparison: (sessionId: string) =>
    request<LiveComparison>(
      `/api/live/${encodeURIComponent(sessionId)}/comparison`,
    ),

  // -- how far in a long call is -----------------------------------------

  // Polled beside the request it describes, never instead of it. The token is
  // the caller's: `watch()` in progress.ts mints one and drives both halves.
  progress: (token: string) => request<Progress>(`/api/progress/${token}`),

  // -- filling the cache from the broker ---------------------------------

  download: (body: { symbol: string; timeframe: string; start: string; end: string }) =>
    post<DownloadJob>("/api/data/download", body),
  downloadJob: (jobId: string) => request<DownloadJob>(`/api/data/download/${jobId}`),
};
