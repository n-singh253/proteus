import { api } from "../api/client";
import type { Job } from "../api/types";

export function JobStatus({ job }: { job: Job | null }) {
  if (!job) return null;
  const active = job.state === "queued" || job.state === "running";
  return (
    <div className={`job job-${job.state}`} role="status">
      <strong>{job.kind}</strong>: {job.state} — {job.stage}
      {active && <progress value={job.progress ?? undefined} max={1} />}
      {active && (
        <button type="button" onClick={() => void api.cancel(job.id)}>
          Cancel
        </button>
      )}
      {job.error && <p className="error">{job.error}</p>}
      {job.reasons.length > 0 && (
        <ul className="error">
          {job.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
