import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Job } from "../api/types";

const POLL_MS = 500;
const FINISHED = new Set(["succeeded", "failed", "cancelled"]);

/** Follows a job until it finishes; `onFinished` runs once with the final state. */
export function useJob(initial: Job | null, onFinished: (job: Job) => void) {
  const [job, setJob] = useState<Job | null>(initial);

  useEffect(() => setJob(initial), [initial]);

  useEffect(() => {
    if (!job || FINISHED.has(job.state)) return;
    const timer = window.setTimeout(async () => {
      try {
        const next = await api.job(job.id);
        setJob(next);
        if (FINISHED.has(next.state)) onFinished(next);
      } catch {
        setJob({ ...job, state: "failed", error: "lost contact with the Proteus server" });
      }
    }, POLL_MS);
    return () => window.clearTimeout(timer);
  }, [job, onFinished]);

  return job;
}
