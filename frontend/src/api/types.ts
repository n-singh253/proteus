// Mirrors proteus_music.api.views; keep in sync with the Python models.
export type JobState = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface Job {
  id: string;
  kind: string;
  state: JobState;
  stage: string;
  progress: number | null;
  result: Record<string, unknown> | null;
  error: string | null;
  reasons: string[];
}

export interface LibraryStats {
  tracks: number;
  source_records: number;
  records_by_source: Record<string, number>;
  tracks_with_local_file: number;
  tracks_without_analysis_data: number;
}

export interface ImportRequest {
  kind: "csv" | "json" | "m3u" | "local";
  path: string;
  identification_only: boolean;
}

export interface OrganizeRequest {
  mode: "guided" | "auto";
  playlists?: number;
  min_playlists?: number;
  max_playlists?: number;
  categories?: { name: string; required: boolean }[];
  min_size?: number;
  max_size?: number;
  features: "fused" | "metadata" | "audio";
}

export interface ExportRequest {
  target: "m3u8" | "csv" | "json";
  out_dir: string;
  relative_paths: boolean;
  overwrite: boolean;
}

export interface ResultSummary {
  id: string;
  created_at: string;
  playlists: number;
  tracks: number;
}

export interface TrackView {
  id: string;
  title: string;
  artists: string[];
  flags: string[];
  similarity: number;
}

export interface PlaylistView {
  index: number;
  name: string;
  description: string;
  origin: string;
  size: number;
  cohesion: number;
  flagged: number;
  tracks: TrackView[];
  flagged_tracks: TrackView[];
}

export interface QualityReport {
  silhouette: number | null;
  fragmentation: number;
  low_confidence_rate: number;
  semantic_coherence: number;
  score: number;
}

export interface ResultView {
  id: string;
  created_at: string;
  track_count: number;
  config: Record<string, unknown>;
  warnings: string[];
  quality: QualityReport | null;
  selection: { chosen: number; reason: string } | null;
  categories: { name: string; evidence: string; kept: boolean; note: string }[];
  playlists: PlaylistView[];
}
