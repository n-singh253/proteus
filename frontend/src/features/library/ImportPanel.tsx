import { useCallback, useState } from "react";
import { ApiError, api } from "../../api/client";
import type { ImportRequest, Job, LibraryStats } from "../../api/types";
import { JobStatus } from "../../shared/JobStatus";
import { useJob } from "../../shared/useJob";

interface Props {
  stats: LibraryStats | null;
  onImported: () => void;
}

export function ImportPanel({ stats, onImported }: Props) {
  const [kind, setKind] = useState<ImportRequest["kind"]>("csv");
  const [path, setPath] = useState("");
  const [restricted, setRestricted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [started, setStarted] = useState<Job | null>(null);
  const job = useJob(started, useCallback(() => onImported(), [onImported]));

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      setStarted(await api.startImport({ kind, path, identification_only: restricted }));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }

  return (
    <section>
      <h2>1. Library</h2>
      {stats && (
        <p>
          {stats.tracks} tracks from {stats.source_records} source records
          {stats.tracks_without_analysis_data > 0 &&
            ` · ${stats.tracks_without_analysis_data} identification-only`}
        </p>
      )}
      <form onSubmit={submit}>
        <select value={kind} onChange={(e) => setKind(e.target.value as ImportRequest["kind"])}>
          <option value="csv">CSV file</option>
          <option value="json">JSON file</option>
          <option value="m3u">M3U/M3U8 playlist</option>
          <option value="local">Local audio folder</option>
        </select>
        <input
          aria-label="Absolute path"
          placeholder="/Users/you/Music or /Users/you/library.csv"
          value={path}
          onChange={(e) => setPath(e.target.value)}
        />
        {kind !== "local" && (
          <label>
            <input type="checkbox" checked={restricted} onChange={(e) => setRestricted(e.target.checked)} />
            Identification only (provider terms restrict analysis)
          </label>
        )}
        <button type="submit">Import</button>
      </form>
      {error && <p className="error">{error}</p>}
      <JobStatus job={job} />
      {job?.state === "succeeded" && job.result && (
        <p>
          {String(job.result.new_tracks)} new, {String(job.result.matched_existing)} matched,{" "}
          {String(job.result.refreshed_records)} refreshed · {String(job.result.failures)} skipped
        </p>
      )}
    </section>
  );
}
