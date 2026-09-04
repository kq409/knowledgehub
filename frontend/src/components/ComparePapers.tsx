import { Columns2, Plus, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import styles from './ComparePapers.module.css';
import type {
  CitationSourceType,
  CompareCitation,
  CompareResponse,
  CompareSummary,
  DefaultCompareDimension,
  LibraryNote,
  Paper,
} from '../types';
import {
  DEFAULT_COMPARE_DIMENSIONS,
  dimensionLabel,
} from '../types';
import { Box } from './Box';
import { Spinner } from './Spinner';

const MAX_PAPERS = 4;
const MAX_CUSTOM_DIMENSIONS = 4;

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return '';
  }
  return date.toLocaleString();
}

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

function sourceLabel(sourceType: CitationSourceType): string {
  if (sourceType === 'paper') {
    return 'Paper';
  }
  if (sourceType === 'voice') {
    return 'Voice note';
  }
  return 'Handwritten note';
}

function badgeClass(sourceType: CitationSourceType): string {
  if (sourceType === 'paper') {
    return styles.paper ?? '';
  }
  if (sourceType === 'voice') {
    return styles.voice ?? '';
  }
  return styles.handwritten ?? '';
}

function citationMeta(citation: CompareCitation): string {
  const parts: string[] = [];
  if (citation.year != null) {
    parts.push(String(citation.year));
  }
  if (citation.page != null) {
    parts.push(`p. ${citation.page}`);
  }
  if (citation.section) {
    parts.push(citation.section);
  }
  return parts.join(' · ');
}

const ALL_DEFAULTS_ON = Object.fromEntries(
  DEFAULT_COMPARE_DIMENSIONS.map((dim) => [dim, true])
) as Record<DefaultCompareDimension, boolean>;

export function ComparePapers() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [notes, setNotes] = useState<LibraryNote[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [enabledDefaults, setEnabledDefaults] =
    useState<Record<DefaultCompareDimension, boolean>>(ALL_DEFAULTS_ON);
  const [customDimensions, setCustomDimensions] = useState<string[]>([]);
  const [customDraft, setCustomDraft] = useState('');
  const [isComparing, setIsComparing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [history, setHistory] = useState<CompareSummary[]>([]);

  const loadLibrary = useCallback(async () => {
    try {
      const [paperRes, noteRes] = await Promise.all([
        fetch('/api/papers'),
        fetch('/api/notes'),
      ]);
      if (!paperRes.ok) {
        throw new Error(await readErrorDetail(paperRes));
      }
      if (!noteRes.ok) {
        throw new Error(await readErrorDetail(noteRes));
      }
      setPapers((await paperRes.json()) as Paper[]);
      setNotes((await noteRes.json()) as LibraryNote[]);
    } catch (err) {
      setError('Could not load library: ' + networkErrorMessage(err));
    }
  }, []);

  const loadHistory = useCallback(async () => {
    try {
      const response = await fetch('/api/compare');
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      setHistory((await response.json()) as CompareSummary[]);
    } catch (err) {
      console.error('Failed to load comparison history:', err);
    }
  }, []);

  useEffect(() => {
    void loadLibrary();
    void loadHistory();
  }, [loadLibrary, loadHistory]);

  const dimensions = useMemo(() => {
    const defaults = DEFAULT_COMPARE_DIMENSIONS.filter(
      (dim) => enabledDefaults[dim]
    );
    return [...defaults, ...customDimensions];
  }, [enabledDefaults, customDimensions]);

  const notesByPaper = useMemo(() => {
    const map = new Map<string, LibraryNote[]>();
    for (const note of notes) {
      if (!note.paper_id || note.processing_status !== 'ready') {
        continue;
      }
      const list = map.get(note.paper_id) ?? [];
      list.push(note);
      map.set(note.paper_id, list);
    }
    return map;
  }, [notes]);

  const togglePaper = (paper: Paper) => {
    if (paper.processing_status !== 'ready') {
      return;
    }
    setSelectedIds((current) => {
      if (current.includes(paper.id)) {
        return current.filter((id) => id !== paper.id);
      }
      if (current.length >= MAX_PAPERS) {
        return current;
      }
      return [...current, paper.id];
    });
  };

  const addCustomDimension = () => {
    const label = customDraft.trim();
    if (!label) {
      return;
    }
    const key = label.toLowerCase();
    const exists =
      dimensions.some((dim) => dim.toLowerCase() === key) ||
      customDimensions.some((dim) => dim.toLowerCase() === key);
    if (exists || customDimensions.length >= MAX_CUSTOM_DIMENSIONS) {
      setCustomDraft('');
      return;
    }
    setCustomDimensions((current) => [...current, label]);
    setCustomDraft('');
  };

  const canCompare =
    selectedIds.length >= 2 &&
    selectedIds.length <= MAX_PAPERS &&
    dimensions.length > 0 &&
    !isComparing;

  const runCompare = useCallback(async () => {
    if (!canCompare) {
      return;
    }
    setIsComparing(true);
    setError(null);
    try {
      const response = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paper_ids: selectedIds,
          dimensions,
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      const data = (await response.json()) as CompareResponse;
      setResult(data);
      await loadHistory();
    } catch (err) {
      setError('Compare failed: ' + networkErrorMessage(err));
    } finally {
      setIsComparing(false);
    }
  }, [canCompare, selectedIds, dimensions, loadHistory]);

  const openHistoryItem = useCallback(async (id: string) => {
    setError(null);
    try {
      const response = await fetch(`/api/compare/${id}`);
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      setResult((await response.json()) as CompareResponse);
    } catch (err) {
      setError('Could not load comparison: ' + networkErrorMessage(err));
    }
  }, []);

  const deleteHistoryItem = useCallback(
    async (id: string) => {
      try {
        const response = await fetch(`/api/compare/${id}`, { method: 'DELETE' });
        if (!response.ok) {
          throw new Error(await readErrorDetail(response));
        }
        if (result?.id === id) {
          setResult(null);
        }
        await loadHistory();
      } catch (err) {
        setError('Delete failed: ' + networkErrorMessage(err));
      }
    },
    [result, loadHistory]
  );

  const citationsByPaper = useMemo(() => {
    if (!result) {
      return new Map<string, CompareCitation[]>();
    }
    const map = new Map<string, CompareCitation[]>();
    for (const citation of result.citations) {
      const list = map.get(citation.paper_id) ?? [];
      list.push(citation);
      map.set(citation.paper_id, list);
    }
    return map;
  }, [result]);

  return (
    <div className={styles.compare}>
      <Box header="Compare Papers" icon={Columns2}>
        <p className={styles.intro}>
          Select 2–4 ready papers. Comparison uses paper text first; linked
          notes are researcher commentary, not paper claims.
        </p>

        <fieldset className={styles.sources} disabled={isComparing}>
          <legend className={styles.legend}>Papers</legend>
          {papers.length === 0 ? (
            <p className={styles.empty}>No papers yet. Upload PDFs in Paper Library.</p>
          ) : (
            <ul className={styles.paperList}>
              {papers.map((paper) => {
                const ready = paper.processing_status === 'ready';
                const checked = selectedIds.includes(paper.id);
                const atCap =
                  !checked && selectedIds.length >= MAX_PAPERS;
                const linked = notesByPaper.get(paper.id) ?? [];
                return (
                  <li key={paper.id}>
                    <label
                      className={`${styles.paperRow} ${
                        !ready ? styles.disabledRow : ''
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={!ready || atCap || isComparing}
                        onChange={() => togglePaper(paper)}
                      />
                      <span>
                        <span className={styles.paperTitle}>{paper.title}</span>
                        <span className={styles.paperMeta}>
                          {paper.processing_status}
                          {paper.year != null ? ` · ${paper.year}` : ''}
                          {linked.length > 0
                            ? ` · ${linked.length} linked note${
                                linked.length === 1 ? '' : 's'
                              }`
                            : ''}
                        </span>
                        {checked && linked.length > 0 && (
                          <span className={styles.linkedNotes}>
                            Notes used:{' '}
                            {linked.map((note) => note.title).join(', ')}
                          </span>
                        )}
                      </span>
                    </label>
                  </li>
                );
              })}
            </ul>
          )}
        </fieldset>

        <fieldset className={styles.sources} disabled={isComparing}>
          <legend className={styles.legend}>Dimensions</legend>
          <div className={styles.dimensionGrid}>
            {DEFAULT_COMPARE_DIMENSIONS.map((dim) => (
              <label key={dim} className={styles.check}>
                <input
                  type="checkbox"
                  checked={enabledDefaults[dim]}
                  onChange={(e) =>
                    setEnabledDefaults((current) => ({
                      ...current,
                      [dim]: e.target.checked,
                    }))
                  }
                />
                {dimensionLabel(dim)}
              </label>
            ))}
          </div>
          <div className={styles.customRow}>
            <input
              className={styles.customInput}
              value={customDraft}
              onChange={(e) => setCustomDraft(e.target.value)}
              placeholder="Add a custom dimension"
              aria-label="Custom comparison dimension"
              disabled={customDimensions.length >= MAX_CUSTOM_DIMENSIONS}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  addCustomDimension();
                }
              }}
            />
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={addCustomDimension}
              disabled={
                !customDraft.trim() ||
                customDimensions.length >= MAX_CUSTOM_DIMENSIONS
              }
            >
              <Plus className={styles.icon} />
              Add
            </button>
          </div>
          {customDimensions.length > 0 && (
            <ul className={styles.customList}>
              {customDimensions.map((dim) => (
                <li key={dim} className={styles.customItem}>
                  <span>{dim}</span>
                  <button
                    type="button"
                    className={styles.iconButton}
                    aria-label={`Remove ${dim}`}
                    onClick={() =>
                      setCustomDimensions((current) =>
                        current.filter((item) => item !== dim)
                      )
                    }
                  >
                    <Trash2 className={styles.icon} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </fieldset>

        <div className={styles.actions}>
          <button
            type="button"
            className={styles.primaryButton}
            onClick={() => void runCompare()}
            disabled={!canCompare}
          >
            {isComparing ? 'Comparing…' : 'Compare'}
          </button>
        </div>
      </Box>

      {error && (
        <div className={styles.error} role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {isComparing && (
        <Box header="Comparison">
          <div className={styles.loading}>
            <Spinner />
            <p>Retrieving evidence and building the comparison table…</p>
          </div>
        </Box>
      )}

      {!isComparing && result && (
        <>
          <Box header="Comparison table">
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Dimension</th>
                    {result.papers.map((paper) => (
                      <th key={paper.paper_id}>
                        {paper.title}
                        {paper.year != null ? ` (${paper.year})` : ''}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.dimensions.map((dim) => (
                    <tr key={dim}>
                      <th>{dimensionLabel(dim)}</th>
                      {result.papers.map((paper) => (
                        <td key={`${paper.paper_id}-${dim}`}>
                          {paper.values[dim] || '—'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className={styles.modelMeta}>
              {result.model} · {result.prompt_version}
            </p>
          </Box>

          <Box header="Synthesis">
            <h3 className={styles.synthTitle}>Agreements</h3>
            <p className={styles.answer}>
              {result.synthesis.agreements || '—'}
            </p>
            <h3 className={styles.synthTitle}>Disagreements</h3>
            <p className={styles.answer}>
              {result.synthesis.disagreements || '—'}
            </p>
            <h3 className={styles.synthTitle}>Research gap</h3>
            <p className={styles.answer}>
              {result.synthesis.research_gap || '—'}
            </p>
          </Box>

          <Box header="Citations">
            {result.citations.length === 0 ? (
              <p className={styles.empty}>No matching chunks were retrieved.</p>
            ) : (
              result.papers.map((paper) => {
                const group = citationsByPaper.get(paper.paper_id) ?? [];
                if (group.length === 0) {
                  return null;
                }
                return (
                  <div key={paper.paper_id} className={styles.citationGroup}>
                    <h3 className={styles.synthTitle}>{paper.title}</h3>
                    <ol className={styles.citations}>
                      {group.map((citation) => (
                        <li key={citation.chunk_id} className={styles.citation}>
                          <div className={styles.citationHeader}>
                            <span className={styles.index}>
                              [{citation.index}]
                            </span>
                            <span
                              className={`${styles.badge} ${badgeClass(
                                citation.source_type
                              )}`}
                            >
                              {sourceLabel(citation.source_type)}
                            </span>
                            <span className={styles.citationTitle}>
                              {citation.title}
                            </span>
                          </div>
                          {citationMeta(citation) && (
                            <p className={styles.citationMeta}>
                              {citationMeta(citation)}
                            </p>
                          )}
                          <p className={styles.snippet}>{citation.snippet}</p>
                        </li>
                      ))}
                    </ol>
                  </div>
                );
              })
            )}
          </Box>
        </>
      )}

      <Box header="History">
        {history.length === 0 ? (
          <p className={styles.empty}>No saved comparisons yet.</p>
        ) : (
          <ul className={styles.historyList}>
            {history.map((item) => (
              <li key={item.id} className={styles.historyItem}>
                <button
                  type="button"
                  className={styles.historyButton}
                  onClick={() => void openHistoryItem(item.id)}
                >
                  <span className={styles.paperTitle}>
                    {item.paper_titles.join(' · ') || 'Comparison'}
                  </span>
                  <span className={styles.paperMeta}>
                    {formatDate(item.created_at)}
                  </span>
                </button>
                <button
                  type="button"
                  className={styles.iconButton}
                  aria-label="Delete comparison"
                  onClick={() => void deleteHistoryItem(item.id)}
                >
                  <Trash2 className={styles.icon} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </Box>
    </div>
  );
}
