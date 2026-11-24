import type {
  ExportRequest,
  ImportRequest,
  Job,
  LibraryStats,
  OrganizeRequest,
  ResultSummary,
  ResultView,
} from "./types";

const token =
  document.querySelector<HTMLMetaElement>('meta[name="proteus-token"]')?.content ?? "";

export class ApiError extends Error {}

interface ErrorBody {
  error?: string;
  detail?: string | { msg: string }[];
}

async function request<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", "X-Proteus-Token": token },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const data = (await response.json().catch(() => ({}))) as ErrorBody;
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new ApiError(data.error ?? detail ?? `request failed (HTTP ${response.status})`);
  }
  return (await response.json()) as T;
}

export const api = {
  library: () => request<LibraryStats>("GET", "/api/library"),
  startImport: (body: ImportRequest) => request<Job>("POST", "/api/imports", body),
  startOrganize: (body: OrganizeRequest) => request<Job>("POST", "/api/organize", body),
  job: (id: string) => request<Job>("GET", `/api/jobs/${id}`),
  cancel: (id: string) => request<Job>("POST", `/api/jobs/${id}/cancel`),
  results: () => request<ResultSummary[]>("GET", "/api/results"),
  result: (id: string) => request<ResultView>("GET", `/api/results/${id}`),
  startExport: (id: string, body: ExportRequest) =>
    request<Job>("POST", `/api/results/${id}/export`, body),
};
