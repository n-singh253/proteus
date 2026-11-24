import { useCallback, useState } from "react";
import { ApiError, api } from "../../api/client";
import type { Job, OrganizeRequest } from "../../api/types";
import { JobStatus } from "../../shared/JobStatus";
import { useJob } from "../../shared/useJob";

type Mode = "auto" | "exact" | "range" | "categories";

function number(value: string): number | undefined {
  return value.trim() ? Number(value) : undefined;
}

function names(value: string, required: boolean) {
  return value.split(",").map((n) => n.trim()).filter(Boolean).map((name) => ({ name, required }));
}

export function OrganizePanel({ onOrganized }: { onOrganized: (resultId: string) => void }) {
  const [mode, setMode] = useState<Mode>("auto");
  const [fields, setFields] = useState({ count: "8", min: "4", max: "12", required: "", preferred: "",
    minSize: "", maxSize: "" });
  const [features, setFeatures] = useState<OrganizeRequest["features"]>("fused");
  const [error, setError] = useState<string | null>(null);
  const [started, setStarted] = useState<Job | null>(null);
  const finished = useCallback((job: Job) => {
    if (job.state === "succeeded" && job.result) onOrganized(String(job.result.result_id));
  }, [onOrganized]);
  const job = useJob(started, finished);
  const set = (key: keyof typeof fields) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setFields({ ...fields, [key]: e.target.value });

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    const request: OrganizeRequest = {
      mode: mode === "auto" ? "auto" : "guided",
      playlists: mode === "exact" ? number(fields.count) : undefined,
      min_playlists: mode === "range" ? number(fields.min) : undefined,
      max_playlists: mode === "range" ? number(fields.max) : undefined,
      categories: [...(mode === "auto" ? [] : names(fields.required, true)), ...names(fields.preferred, false)],
      min_size: number(fields.minSize),
      max_size: number(fields.maxSize),
      features,
    };
    try {
      setStarted(await api.startOrganize(request));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }

  return (
    <section>
      <h2>2. Organize</h2>
      <form onSubmit={submit} className="organize">
        <fieldset>
          {(["auto", "exact", "range", "categories"] as Mode[]).map((m) => (
            <label key={m}>
              <input type="radio" name="mode" checked={mode === m} onChange={() => setMode(m)} />
              {{ auto: "Autonomous", exact: "Exact count", range: "Count range", categories: "Categories" }[m]}
            </label>
          ))}
        </fieldset>
        {mode === "exact" && <input aria-label="Playlists" type="number" min={1} value={fields.count} onChange={set("count")} />}
        {mode === "range" && (
          <>
            <input aria-label="Fewest playlists" type="number" min={1} value={fields.min} onChange={set("min")} />
            <input aria-label="Most playlists" type="number" min={1} value={fields.max} onChange={set("max")} />
          </>
        )}
        {mode !== "auto" && (
          <input aria-label="Required categories" placeholder="Required categories: Hip-Hop, House"
            value={fields.required} onChange={set("required")} />
        )}
        <input aria-label="Preferred categories" placeholder="Preferred categories (optional)"
          value={fields.preferred} onChange={set("preferred")} />
        <input aria-label="Minimum size" type="number" min={1} placeholder="Min size" value={fields.minSize} onChange={set("minSize")} />
        <input aria-label="Maximum size" type="number" min={1} placeholder="Max size" value={fields.maxSize} onChange={set("maxSize")} />
        <select value={features} onChange={(e) => setFeatures(e.target.value as OrganizeRequest["features"])}>
          <option value="fused">Metadata + audio</option>
          <option value="metadata">Metadata only</option>
          <option value="audio">Audio only</option>
        </select>
        <button type="submit">Organize</button>
      </form>
      {error && <p className="error">{error}</p>}
      <JobStatus job={job} />
    </section>
  );
}
