import { Link2 } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import styles from './RelatedPapers.module.css';
import type {
  ConnectReasonMode,
  ConnectResponse,
  LibraryNote,
  NotePaperLink,
  Paper,
} from '../types';
import { Spinner } from './Spinner';

function networkErrorMessage(err: unknown): string {
  if (err instanceof TypeError && err.message === 'Failed to fetch') {
    return 'Cannot reach the backend. Make sure the API server is running on port 8000.';
  }
  if (err instanceof Error && err.message) {
    return err.message;
  }
  return 'Unknown error';
}

async function readErrorDetail(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const data = JSON.parse(text) as { detail?: unknown };
    if (typeof data.detail === 'string' && data.detail.trim()) {
      return data.detail;
    }
  } catch {
    // Response was not JSON.
  }
  return text || `Request failed (${response.status})`;
}

interface RelatedPapersPanelProps {
  noteId?: string;
  isReady: boolean;
  relatedGeneratedAt?: string | null;
  updatedAt?: string | null;
  paperLinks: NotePaperLink[];
  papers: Paper[];
  onNoteUpdate?: (note: LibraryNote) => void;
}

export function RelatedPapersPanel({
  noteId,
  isReady,
  relatedGeneratedAt,
  updatedAt,
  paperLinks,
  papers,
  onNoteUpdate,
}: RelatedPapersPanelProps) {
  const [reasonMode, setReasonMode] = useState<ConnectReasonMode>('llm');
  const [result, setResult] = useState<ConnectResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStored = useCallback(async (id: string) => {
    const response = await fetch(`/api/notes/${id}/related`);
    if (response.status === 404) {
      return null;
    }
    if (!response.ok) {
      throw new Error(await readErrorDetail(response));
    }
    return (await response.json()) as ConnectResponse;
  }, []);

  useEffect(() => {
    if (!noteId || !relatedGeneratedAt) {
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const stored = await loadStored(noteId);
        if (!cancelled) {
          setResult(stored);
        }
      } catch (err) {
        if (!cancelled) {
          setError(networkErrorMessage(err));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [noteId, relatedGeneratedAt, loadStored]);

  const runConnect = useCallback(async () => {
    if (!noteId || !isReady) {
      return;
    }
    setIsRunning(true);
    setError(null);
    try {
      const response = await fetch(`/api/notes/${noteId}/related`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason_mode: reasonMode }),
      });
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      const body = (await response.json()) as ConnectResponse;
      setResult(body);
      const noteRes = await fetch(`/api/notes/${noteId}`);
      if (noteRes.ok && onNoteUpdate) {
        onNoteUpdate((await noteRes.json()) as LibraryNote);
      }
    } catch (err) {
      setError(networkErrorMessage(err));
    } finally {
      setIsRunning(false);
    }
  }, [noteId, isReady, reasonMode, onNoteUpdate]);

  const waitingForAuto =
    Boolean(noteId) &&
    isReady &&
    !relatedGeneratedAt &&
    result == null &&
    !isRunning &&
    updatedAt != null &&
    Date.now() - new Date(updatedAt).getTime() < 90_000;

  const titleById = new Map(papers.map((paper) => [paper.id, paper.title]));
  const sourceById = new Map(paperLinks.map((link) => [link.paper_id, link.source]));

  return (
    <div className={styles.panel}>
      <label className={styles.fieldLabel} id="related-papers-label">
        Related papers
        <span className={styles.hint}>local library · AI links can be unchecked</span>
      </label>

      <div className={styles.modeRow} role="group" aria-label="Reason mode">
        <label className={styles.mode}>
          <input
            type="radio"
            name={`reason-mode-${noteId ?? 'draft'}`}
            checked={reasonMode === 'llm'}
            onChange={() => setReasonMode('llm')}
          />
          LLM reason
        </label>
        <label className={styles.mode}>
          <input
            type="radio"
            name={`reason-mode-${noteId ?? 'draft'}`}
            checked={reasonMode === 'snippet'}
            onChange={() => setReasonMode('snippet')}
          />
          Snippet
        </label>
      </div>

      <button
        type="button"
        className={styles.secondaryButton}
        onClick={() => void runConnect()}
        disabled={!noteId || !isReady || isRunning}
      >
        <Link2 className={styles.icon} />
        {isRunning ? 'Finding…' : 'Find related papers'}
      </button>

      {!noteId && (
        <p className={styles.hint}>Save the note first to search the paper library.</p>
      )}
      {noteId && !isReady && (
        <p className={styles.hint}>Wait until the note finishes processing.</p>
      )}

      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}

      {(isRunning || waitingForAuto) && (
        <div className={styles.loading} aria-live="polite">
          <Spinner />
          <p>Searching the paper library…</p>
        </div>
      )}

      {result && result.papers.length === 0 && !isRunning && (
        <p className={styles.empty}>No related papers above the similarity threshold.</p>
      )}

      {result && result.papers.length > 0 && (
        <ol className={styles.list} aria-labelledby="related-papers-label">
          {result.papers.map((paper) => {
            const linked = paper.linked;
            const source = paper.source ?? sourceById.get(paper.paper_id) ?? null;
            return (
              <li key={paper.paper_id} className={styles.item}>
                <div className={styles.itemHeader}>
                  <span className={styles.itemTitle}>
                    {paper.title || titleById.get(paper.paper_id) || 'Untitled paper'}
                  </span>
                  {linked && source === 'ai' && (
                    <span className={styles.badge}>AI</span>
                  )}
                  {linked && source === 'researcher' && (
                    <span className={styles.badgeMuted}>linked</span>
                  )}
                </div>
                <p className={styles.meta}>
                  {paper.year != null ? `${paper.year} · ` : ''}
                  similarity {paper.similarity.toFixed(2)}
                  {paper.page != null ? ` · p. ${paper.page}` : ''}
                  {paper.section ? ` · ${paper.section}` : ''}
                </p>
                <p className={styles.reason}>{paper.reason}</p>
                {paper.reason_mode === 'llm' && paper.snippet && paper.reason !== paper.snippet && (
                  <p className={styles.snippet}>{paper.snippet}</p>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
