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

  edge: (body: { strategy_id: string; config: RunConfigIn; horizons?: number[] }) =>
    post<EdgeReport>("/api/edge", body),
  backtest: (body: { strategy_id: string; config: RunConfigIn; force?: boolean }) =>
    post<BacktestResponse>("/api/backtest", body),

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
    run_id: string;
    mode?: "rolling" | "anchored";
    train_days?: number;
    test_days?: number;
    min_train_trades?: number;
    grid?: ParameterGrid | null;
  }) => post<WalkForwardResponse>("/api/validation/walkforward", body),

  permutation: (body: {
    run_id: string;
    iterations?: number;
    tests?: ("random_entries" | "permuted_returns")[];
    block_bars?: number;
    seed?: number;
  }) => post<PermutationResponse>("/api/validation/permutation", body),

  multipleTesting: (body: { run_id: string }) =>
    post<MultipleTestingResponse>("/api/validation/multiple-testing", body),

  tickResolve: (body: { run_id: string }) =>
    post<TickResolveResponse>("/api/validation/tick-resolve", body),

  // -- batch -------------------------------------------------------------

  batch: (body: {
    strategy_id: string;
    symbols: string[];
    config: RunConfigIn;
    max_workers?: number;
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
};
