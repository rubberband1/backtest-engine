import { useEffect, useState } from "react";
import { api, type SymbolList } from "./api/client";
import { BatchPage } from "./pages/BatchPage";
import { ComparePage } from "./pages/ComparePage";
import { LivePage } from "./pages/LivePage";
import { ResultPage } from "./pages/ResultPage";
import { RunPage } from "./pages/RunPage";
import { ScreenPage } from "./pages/ScreenPage";
import { StrategyPage } from "./pages/StrategyPage";
import { ValidationPage } from "./pages/ValidationPage";
import { useRoute } from "./router";

export function App() {
  const route = useRoute();
  const [environment, setEnvironment] = useState<SymbolList | null>(null);

  useEffect(() => {
    api.symbols().then(setEnvironment).catch(() => setEnvironment(null));
  }, []);

  const onFixture = environment?.source === "fixture";

  return (
    <div className="app">
      {/* Invented data has to announce itself, above everything, on every
          page. A dashboard that renders a synthetic equity curve exactly
          like a real one is the single failure this project cannot have. */}
      {onFixture && (
        <div className="banner banner-synthetic" role="status">
          <strong>Synthetic data.</strong> These bars are generated, not
          traded. Every number on every page below describes a random walk,
          not a market. Point the backend at real bars to measure anything.
        </div>
      )}
      <header className="topbar">
        <div className="brand">
          backtest-engine <span>· local</span>
        </div>
        <nav className="nav">
          <a href="#/" aria-current={route.name === "run" ? "page" : undefined}>
            Run
          </a>
          <a href="#/result" aria-current={route.name === "result" ? "page" : undefined}>
            Result
          </a>
          <a href="#/compare" aria-current={route.name === "compare" ? "page" : undefined}>
            Compare
          </a>
          <a
            href="#/validation"
            aria-current={route.name === "validation" ? "page" : undefined}
          >
            Validation
          </a>
          <a
            href="#/strategy"
            aria-current={route.name === "strategy" ? "page" : undefined}
          >
            Strategy
          </a>
          <a href="#/batch" aria-current={route.name === "batch" ? "page" : undefined}>
            Batch
          </a>
          <a href="#/screen" aria-current={route.name === "screen" ? "page" : undefined}>
            Screen
          </a>
          <a href="#/live" aria-current={route.name === "live" ? "page" : undefined}>
            Live
          </a>
        </nav>
        <div className="topbar-meta">
          {/* MT5 server time != local time != UTC: always say which. */}
          <div>All timestamps in UTC</div>
          <div>
            {environment
              ? `symbols: ${
                  environment.source === "terminal"
                    ? "MT5 terminal"
                    : environment.source === "fixture"
                      ? "synthetic fixture"
                      : "local cache"
                }` +
                (environment.server_timezone ? ` · server ${environment.server_timezone}` : "")
              : "backend not reachable"}
          </div>
        </div>
      </header>

      <main>
        {route.name === "run" && <RunPage />}
        {route.name === "result" && <ResultPage runId={route.runId} />}
        {route.name === "compare" && <ComparePage initialRunIds={route.runIds} />}
        {route.name === "validation" && <ValidationPage runId={route.runId} />}
        {route.name === "strategy" && <StrategyPage strategyId={route.strategyId} />}
        {route.name === "batch" && <BatchPage />}
        {route.name === "screen" && <ScreenPage jobId={route.jobId} />}
        {route.name === "live" && <LivePage sessionId={route.sessionId} />}
      </main>
    </div>
  );
}
