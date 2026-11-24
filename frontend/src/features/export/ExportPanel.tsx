import { useState } from "react";
import { ApiError, api } from "../../api/client";
import type { ExportRequest, Job } from "../../api/types";
import { JobStatus } from "../../shared/JobStatus";
import { useJob } from "../../shared/useJob";

const noop = () => undefined;

export function ExportPanel({ resultId }: { resultId: string | null }) {
  const [request, setRequest] = useState<ExportRequest>({
    target: "m3u8", out_dir: "", relative_paths: false, overwrite: false,
  });
  const [error, setError] = useState<string | null>(null);
  const [started, setStarted] = useState<Job | null>(null);
  const job = useJob(started, noop);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!resultId) return;
    setError(null);
    try {
      setStarted(await api.startExport(resultId, request));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }

  const files = (job?.result?.files as string[] | undefined) ?? [];
  const unexportable = (job?.result?.unexportable as string[] | undefined) ?? [];
  return (
    <section>
      <h2>4. Export</h2>
      <form onSubmit={submit}>
        <select value={request.target}
          onChange={(e) => setRequest({ ...request, target: e.target.value as ExportRequest["target"] })}>
          <option value="m3u8">M3U8 playlists</option>
          <option value="csv">CSV</option>
          <option value="json">JSON</option>
        </select>
        <input aria-label="Output folder" placeholder="/Users/you/Playlists" value={request.out_dir}
          onChange={(e) => setRequest({ ...request, out_dir: e.target.value })} />
        <label>
          <input type="checkbox" checked={request.relative_paths}
            onChange={(e) => setRequest({ ...request, relative_paths: e.target.checked })} />
          Relative paths
        </label>
        <label>
          <input type="checkbox" checked={request.overwrite}
            onChange={(e) => setRequest({ ...request, overwrite: e.target.checked })} />
          Overwrite existing files
        </label>
        <button type="submit" disabled={!resultId}>Export selected result</button>
      </form>
      {error && <p className="error">{error}</p>}
      <JobStatus job={job} />
      {job?.state === "succeeded" && (
        <>
          <p>{String(job.result?.exported_tracks)} tracks exported to {files.length} file(s).</p>
          {unexportable.length > 0 && (
            <p className="warning">
              {unexportable.length} track(s) could not be exported (e.g. {unexportable[0]}); the saved
              result still contains them.
            </p>
          )}
        </>
      )}
    </section>
  );
}
