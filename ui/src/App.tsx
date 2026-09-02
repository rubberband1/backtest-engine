import { useEffect, useState } from "react";
import { api, type SymbolList } from "./api/client";
import { BatchPage } from "./pages/BatchPage";
import { ComparePage } from "./pages/ComparePage";
import { ResultPage } from "./pages/ResultPage";
import { RunPage } from "./pages/RunPage";
import { ValidationPage } from "./pages/ValidationPage";
import { useRoute } from "./router";

export function App() {
  const route = useRoute();
  const [environment, setEnvironment] = useState<SymbolList | null>(null);

  useEffect(() => {
    api.symbols().then(setEnvironment).catch(() => setEnvironment(null));
  }, []);

  return (
    <div className="app">
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
          <a href="#/batch" aria-current={route.name === "batch" ? "page" : undefined}>
            Batch
          </a>
        </nav>
        <div className="topbar-meta">
          {/* MT5 server time != local time != UTC: always say which. */}
          <div>All timestamps in UTC</div>
          <div>
            {environment
              ? `symbols: ${
                  environment.source === "terminal" ? "MT5 terminal" : "local cache"
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
        {route.name === "batch" && <BatchPage />}
      </main>
    </div>
  );
}
