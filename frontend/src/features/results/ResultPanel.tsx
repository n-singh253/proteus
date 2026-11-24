import { useState } from "react";
import type { PlaylistView, ResultSummary, ResultView } from "../../api/types";

interface Props {
  results: ResultSummary[];
  selected: ResultView | null;
  onSelect: (id: string) => void;
}

function Playlist({ playlist }: { playlist: PlaylistView }) {
  const [open, setOpen] = useState(false);
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const tracks = flaggedOnly ? playlist.flagged_tracks : playlist.tracks;
  return (
    <li className="playlist">
      <button type="button" className="link" onClick={() => setOpen(!open)}>
        {playlist.name} · {playlist.size} tracks{playlist.flagged > 0 && ` · ${playlist.flagged} flagged`}
      </button>
      <span className="origin">{playlist.origin}</span>
      {open && (
        <div>
          <p className="muted">{playlist.description}</p>
          {playlist.flagged > 0 && (
            <label>
              <input type="checkbox" checked={flaggedOnly} onChange={(e) => setFlaggedOnly(e.target.checked)} />
              Show flagged tracks only
            </label>
          )}
          <ol>
            {tracks.map((t) => (
              <li key={t.id}>
                {t.artists.join(", ") || "?"} — {t.title}
                {t.flags.length > 0 && <span className="flag"> {t.flags.join(", ")}</span>}
              </li>
            ))}
          </ol>
          {!flaggedOnly && playlist.size > tracks.length && (
            <p className="muted">Showing the {tracks.length} best-fitting tracks; export for the full list.</p>
          )}
        </div>
      )}
    </li>
  );
}

export function ResultPanel({ results, selected, onSelect }: Props) {
  return (
    <section>
      <h2>3. Playlists</h2>
      {results.length === 0 ? (
        <p className="muted">No saved results yet.</p>
      ) : (
        <select value={selected?.id ?? ""} onChange={(e) => onSelect(e.target.value)}>
          <option value="" disabled>Choose a saved result</option>
          {results.map((r) => (
            <option key={r.id} value={r.id}>
              {r.id} · {new Date(r.created_at).toLocaleString()} · {r.playlists} playlists
            </option>
          ))}
        </select>
      )}
      {selected && (
        <>
          <p>
            {selected.playlists.length} playlists for {selected.track_count} tracks
            {selected.selection && ` — ${selected.selection.reason}`}
          </p>
          {selected.quality && (
            <p className="muted">
              Silhouette {selected.quality.silhouette?.toFixed(3) ?? "n/a"} · semantic coherence{" "}
              {(selected.quality.semantic_coherence * 100).toFixed(0)}% · low-confidence{" "}
              {(selected.quality.low_confidence_rate * 100).toFixed(0)}% · validated: every track
              assigned exactly once
            </p>
          )}
          {selected.warnings.map((w) => (
            <p key={w} className="warning">{w}</p>
          ))}
          <ul>
            {selected.playlists.map((p) => (
              <Playlist key={p.index} playlist={p} />
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
