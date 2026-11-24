import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import type { LibraryStats, ResultSummary, ResultView } from "./api/types";
import { ExportPanel } from "./features/export/ExportPanel";
import { ImportPanel } from "./features/library/ImportPanel";
import { OrganizePanel } from "./features/organize/OrganizePanel";
import { ResultPanel } from "./features/results/ResultPanel";

export function App() {
  const [stats, setStats] = useState<LibraryStats | null>(null);
  const [results, setResults] = useState<ResultSummary[]>([]);
  const [selected, setSelected] = useState<ResultView | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setStats(await api.library());
      setResults(await api.results());
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  }, []);

  const select = useCallback(async (id: string) => {
    try {
      setSelected(await api.result(id));
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  }, []);

  const organized = useCallback(async (id: string) => {
    await refresh();
    await select(id);
  }, [refresh, select]);

  useEffect(() => void refresh(), [refresh]);

  return (
    <main>
      <h1>Proteus</h1>
      {error && <p className="error">{error}</p>}
      <ImportPanel stats={stats} onImported={refresh} />
      <OrganizePanel onOrganized={organized} />
      <ResultPanel results={results} selected={selected} onSelect={select} />
      <ExportPanel resultId={selected?.id ?? null} />
    </main>
  );
}
