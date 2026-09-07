import { BookOpen, Trash2, Upload, FileText } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styles from './Library.module.css';
import type {
  LibraryNote,
  Paper,
  ProcessingStatus,
} from '../types';
import { Box } from './Box';
import { TextBox } from './TextBox';
import { Spinner } from './Spinner';
import { RelatedPapersPanel } from './RelatedPapers';

type LibraryFilter = 'all' | 'papers' | 'notes';
type LibraryKind = 'paper' | 'note';

interface LibraryUploadResponse {
  kind: LibraryKind;
  paper: Paper | null;
  note: LibraryNote | null;
}

interface SelectedRef {
  kind: LibraryKind;
  id: string;
}

interface LibraryItem {
  kind: LibraryKind;
  id: string;
  title: string;
  updatedAt: string;
  status: ProcessingStatus;
  chunkCount: number;
  paper?: Paper;
  note?: LibraryNote;
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return '';
  }
  return date.toLocaleString();
}

function listToLines(items: string[]): string {
  return items.join('\n');
}

function linesToList(value: string): string[] {
  return value
    .split('\n')
    .map((item) => item.trim())
    .filter(Boolean);
}

function statusClass(status: ProcessingStatus): string {
  return `${styles.status} ${styles[status]}`;
}

function kindLabel(item: LibraryItem): string {
  if (item.kind === 'paper') {
    return 'Paper';
  }
  return item.note?.source_type === 'handwritten' ? 'Handwritten' : 'Voice';
}

function kindClass(item: LibraryItem): string {
  if (item.kind === 'paper') {
    return `${styles.source} ${styles.paper}`;
  }
  return `${styles.source} ${styles[item.note?.source_type ?? 'handwritten']}`;
}

function linkedPaperLabel(note: LibraryNote, papers: Paper[]): string {
  if (!note.paper_ids.length) {
    return '';
  }
  const titles = new Map(papers.map((paper) => [paper.id, paper.title]));
  return note.paper_ids.map((id) => titles.get(id) ?? 'Unknown paper').join(', ');
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

function toItems(papers: Paper[], notes: LibraryNote[]): LibraryItem[] {
  const items: LibraryItem[] = [
    ...papers.map((paper) => ({
      kind: 'paper' as const,
      id: paper.id,
      title: paper.title,
      updatedAt: paper.updated_at,
      status: paper.processing_status,
      chunkCount: paper.chunk_count,
      paper,
    })),
    ...notes.map((note) => ({
      kind: 'note' as const,
      id: note.id,
      title: note.title,
      updatedAt: note.updated_at,
      status: note.processing_status,
      chunkCount: note.chunk_count,
      note,
    })),
  ];
  items.sort(
    (a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt)
  );
  return items;
}

export function Library() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [notes, setNotes] = useState<LibraryNote[]>([]);
  const [selected, setSelected] = useState<SelectedRef | null>(null);
  const [filter, setFilter] = useState<LibraryFilter>('all');
  const [isUploading, setIsUploading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [queuedHint, setQueuedHint] = useState<string | null>(null);
  const [linkedNotes, setLinkedNotes] = useState<LibraryNote[]>([]);
  const [draftTitle, setDraftTitle] = useState('');
  const [draftAuthors, setDraftAuthors] = useState('');
  const [draftYear, setDraftYear] = useState('');
  const [draftAbstract, setDraftAbstract] = useState('');
  const [draftTags, setDraftTags] = useState('');
  const [draftSummary, setDraftSummary] = useState('');
  const [draftObservations, setDraftObservations] = useState('');
  const [draftHypotheses, setDraftHypotheses] = useState('');
  const [draftQuestions, setDraftQuestions] = useState('');
  const [draftNextSteps, setDraftNextSteps] = useState('');
  const [draftExtractedText, setDraftExtractedText] = useState('');
  const [draftPaperIds, setDraftPaperIds] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const items = useMemo(() => toItems(papers, notes), [papers, notes]);
  const visibleItems = useMemo(() => {
    if (filter === 'papers') {
      return items.filter((item) => item.kind === 'paper');
    }
    if (filter === 'notes') {
      return items.filter((item) => item.kind === 'note');
    }
    return items;
  }, [items, filter]);

  const selectedItem =
    items.find(
      (item) => selected != null && item.kind === selected.kind && item.id === selected.id
    ) ?? null;
  const selectedPaper = selectedItem?.paper ?? null;
  const selectedNote = selectedItem?.note ?? null;

  const hasPending = items.some(
    (item) => item.status === 'pending' || item.status === 'processing'
  );
  const waitingForRelated = notes.some(
    (note) =>
      note.processing_status === 'ready' &&
      note.related_generated_at == null &&
      Date.now() - new Date(note.updated_at).getTime() < 90_000
  );

  const loadPapers = useCallback(async () => {
    try {
      const response = await fetch('/api/papers');
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      setPapers((await response.json()) as Paper[]);
    } catch (err) {
      console.error('Failed to load papers:', err);
      setError('Could not load papers: ' + networkErrorMessage(err));
    }
  }, []);

  const loadNotes = useCallback(async () => {
    try {
      const response = await fetch('/api/notes');
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      setNotes((await response.json()) as LibraryNote[]);
    } catch (err) {
      console.error('Failed to load notes:', err);
      setError('Could not load notes: ' + networkErrorMessage(err));
    }
  }, []);

  const loadLibrary = useCallback(async () => {
    await Promise.all([loadPapers(), loadNotes()]);
  }, [loadPapers, loadNotes]);

  useEffect(() => {
    void loadLibrary();
  }, [loadLibrary]);

  useEffect(() => {
    if (!hasPending && !waitingForRelated) {
      return;
    }
    const timer = window.setInterval(() => {
      void loadLibrary();
    }, 2500);
    return () => window.clearInterval(timer);
  }, [hasPending, waitingForRelated, loadLibrary]);

  useEffect(() => {
    if (selectedPaper) {
      setDraftTitle(selectedPaper.title);
      setDraftAuthors(listToLines(selectedPaper.authors));
      setDraftYear(selectedPaper.year != null ? String(selectedPaper.year) : '');
      setDraftAbstract(selectedPaper.abstract ?? '');
      setDraftTags(listToLines(selectedPaper.tags));
      const loadLinked = async () => {
        try {
          const response = await fetch(`/api/papers/${selectedPaper.id}/notes`);
          if (!response.ok) {
            setLinkedNotes([]);
            return;
          }
          setLinkedNotes((await response.json()) as LibraryNote[]);
        } catch {
          setLinkedNotes([]);
        }
      };
      void loadLinked();
      return;
    }
    setLinkedNotes([]);
    if (!selectedNote) {
      return;
    }
    setDraftTitle(selectedNote.title);
    setDraftSummary(selectedNote.summary);
    setDraftObservations(listToLines(selectedNote.observations));
    setDraftHypotheses(listToLines(selectedNote.hypotheses));
    setDraftQuestions(listToLines(selectedNote.questions));
    setDraftNextSteps(listToLines(selectedNote.next_steps));
    setDraftTags(listToLines(selectedNote.tags));
    setDraftExtractedText(selectedNote.extracted_text ?? '');
    setDraftPaperIds(selectedNote.paper_ids ?? []);
  }, [selectedPaper, selectedNote]);

  const uploadPdf = useCallback(
    async (file: File) => {
      if (!file.type.includes('pdf') && !file.name.toLowerCase().endsWith('.pdf')) {
        setError('Please upload a PDF file');
        return;
      }

      setIsUploading(true);
      setError(null);
      setQueuedHint(null);
      try {
        const formData = new FormData();
        formData.append('file', file, file.name);
        const response = await fetch('/api/library/upload', {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) {
          throw new Error(await readErrorDetail(response));
        }
        const data = (await response.json()) as LibraryUploadResponse;
        setFilter('all');
        if (data.kind === 'paper' && data.paper) {
          setSelected({ kind: 'paper', id: data.paper.id });
          setQueuedHint('Queued as paper from document content');
        } else if (data.kind === 'note' && data.note) {
          setSelected({ kind: 'note', id: data.note.id });
          setQueuedHint('Queued as note from document content');
        }
        await loadLibrary();
      } catch (err) {
        setError('Upload failed: ' + networkErrorMessage(err));
      } finally {
        setIsUploading(false);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    },
    [loadLibrary]
  );

  const savePaper = useCallback(async () => {
    if (!selectedPaper) {
      return;
    }
    setIsSaving(true);
    setError(null);
    try {
      const yearValue = draftYear.trim();
      const year = yearValue ? Number.parseInt(yearValue, 10) : null;
      if (yearValue && Number.isNaN(year)) {
        throw new Error('Year must be a number');
      }
      const response = await fetch(`/api/papers/${selectedPaper.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: draftTitle.trim() || selectedPaper.title,
          authors: linesToList(draftAuthors),
          year,
          abstract: draftAbstract.trim() || null,
          tags: linesToList(draftTags),
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      await loadPapers();
    } catch (err) {
      setError('Save failed: ' + networkErrorMessage(err));
    } finally {
      setIsSaving(false);
    }
  }, [
    selectedPaper,
    draftTitle,
    draftAuthors,
    draftYear,
    draftAbstract,
    draftTags,
    loadPapers,
  ]);

  const saveNote = useCallback(async () => {
    if (!selectedNote) {
      return;
    }
    setIsSaving(true);
    setError(null);
    try {
      const payload =
        selectedNote.source_type === 'handwritten'
          ? {
              title: draftTitle.trim() || selectedNote.title,
              tags: linesToList(draftTags),
              extracted_text: draftExtractedText.trim() || null,
              paper_ids: draftPaperIds,
            }
          : {
              title: draftTitle.trim() || selectedNote.title,
              summary: draftSummary,
              observations: linesToList(draftObservations),
              hypotheses: linesToList(draftHypotheses),
              questions: linesToList(draftQuestions),
              next_steps: linesToList(draftNextSteps),
              tags: linesToList(draftTags),
              paper_ids: draftPaperIds,
            };
      const response = await fetch(`/api/notes/${selectedNote.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      await loadNotes();
    } catch (err) {
      setError('Save failed: ' + networkErrorMessage(err));
    } finally {
      setIsSaving(false);
    }
  }, [
    selectedNote,
    draftTitle,
    draftSummary,
    draftObservations,
    draftHypotheses,
    draftQuestions,
    draftNextSteps,
    draftTags,
    draftExtractedText,
    draftPaperIds,
    loadNotes,
  ]);

  const deleteItem = useCallback(
    async (item: LibraryItem) => {
      try {
        const path =
          item.kind === 'paper'
            ? `/api/papers/${item.id}`
            : `/api/notes/${item.id}`;
        const response = await fetch(path, { method: 'DELETE' });
        if (!response.ok) {
          throw new Error(await readErrorDetail(response));
        }
        if (selected?.kind === item.kind && selected.id === item.id) {
          setSelected(null);
        }
        await loadLibrary();
      } catch (err) {
        setError('Delete failed: ' + networkErrorMessage(err));
      }
    },
    [selected, loadLibrary]
  );

  const togglePaper = (paperId: string) => {
    setDraftPaperIds((current) =>
      current.includes(paperId)
        ? current.filter((id) => id !== paperId)
        : [...current, paperId]
    );
  };

  return (
    <div className={`${styles.library} ${styles.compact}`}>
      <Box header="Upload PDF" icon={Upload} compact>
        <div
          className={`${styles.dropzone} ${isDragging ? styles.dragging : ''} ${
            isUploading ? styles.uploading : ''
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            setIsDragging(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            const file = e.dataTransfer.files?.[0];
            if (file) {
              void uploadPdf(file);
            }
          }}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              fileInputRef.current?.click();
            }
          }}
          aria-label="Upload PDF"
        >
          {isUploading ? (
            <>
              <Spinner />
              <p>Uploading — classifying from document content…</p>
            </>
          ) : (
            <>
              <Upload className={styles.uploadIcon} />
              <p className={styles.dropTitle}>
                {isDragging ? 'Drop PDF here' : 'Upload PDF'}
              </p>
              <p className={styles.dropSubtitle}>
                Drag and drop or click to browse. The file is read after upload
                and filed as a paper or a note automatically.
              </p>
            </>
          )}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf,.pdf"
          className={styles.hiddenInput}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) {
              void uploadPdf(file);
            }
          }}
        />
      </Box>

      {queuedHint && (
        <div className={styles.notice} role="status">
          <span>{queuedHint}</span>
          <button type="button" onClick={() => setQueuedHint(null)}>
            Dismiss
          </button>
        </div>
      )}

      {error && (
        <div className={styles.error} role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      <Box header="Library" icon={BookOpen} compact>
        <div className={styles.filters} role="tablist" aria-label="Library filter">
          {(
            [
              ['all', 'All'],
              ['papers', 'Papers'],
              ['notes', 'Notes'],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              role="tab"
              aria-selected={filter === value}
              className={`${styles.filterChip} ${
                filter === value ? styles.active : ''
              }`}
              onClick={() => setFilter(value)}
            >
              {label}
            </button>
          ))}
        </div>
        {visibleItems.length === 0 ? (
          <p className={styles.empty}>
            Nothing here yet. Upload a PDF to start — or save a voice note.
          </p>
        ) : (
          <ul className={styles.list}>
            {visibleItems.map((item) => (
              <li key={`${item.kind}-${item.id}`}>
                <div
                  className={`${styles.item} ${
                    selected?.kind === item.kind && selected.id === item.id
                      ? styles.selected
                      : ''
                  }`}
                >
                  <button
                    type="button"
                    className={styles.selectButton}
                    onClick={() => setSelected({ kind: item.kind, id: item.id })}
                  >
                    <span className={styles.itemTitle}>{item.title}</span>
                    <span className={styles.itemMeta}>
                      <span className={kindClass(item)}>{kindLabel(item)}</span>
                      <span className={statusClass(item.status)}>{item.status}</span>
                      <span>{item.chunkCount} chunks</span>
                      {item.note && linkedPaperLabel(item.note, papers) ? (
                        <span>Linked: {linkedPaperLabel(item.note, papers)}</span>
                      ) : null}
                      <span>{formatDate(item.updatedAt)}</span>
                    </span>
                  </button>
                  <button
                    type="button"
                    className={styles.deleteButton}
                    aria-label={`Delete ${item.title}`}
                    onClick={() => void deleteItem(item)}
                  >
                    <Trash2 className={styles.icon} />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Box>

      {selectedPaper && (
        <Box header="Paper Details" icon={FileText} compact>
          <div className={styles.metaRow}>
            <span className={`${styles.source} ${styles.paper}`}>Paper</span>
            <span className={statusClass(selectedPaper.processing_status)}>
              {selectedPaper.processing_status}
            </span>
            <span className={styles.metaText}>
              {selectedPaper.page_count != null
                ? `${selectedPaper.page_count} pages`
                : 'Pages unknown'}
              {' · '}
              {selectedPaper.chunk_count} chunks
              {' · '}
              {selectedPaper.original_filename}
            </span>
          </div>

          {selectedPaper.processing_error && (
            <p className={styles.processingError}>
              {selectedPaper.processing_error}
            </p>
          )}

          <label className={styles.fieldLabel} htmlFor="library-paper-title">
            Title
          </label>
          <TextBox
            id="library-paper-title"
            mode="input"
            value={draftTitle}
            onChange={setDraftTitle}
            rows={2}
            ariaLabel="Paper title"
          />

          <label className={styles.fieldLabel} htmlFor="library-paper-authors">
            Authors
            <span className={styles.hint}>one per line</span>
          </label>
          <TextBox
            id="library-paper-authors"
            mode="input"
            value={draftAuthors}
            onChange={setDraftAuthors}
            rows={3}
            ariaLabel="Paper authors"
          />

          <label className={styles.fieldLabel} htmlFor="library-paper-year">
            Year
          </label>
          <TextBox
            id="library-paper-year"
            mode="input"
            value={draftYear}
            onChange={setDraftYear}
            rows={1}
            ariaLabel="Paper year"
          />

          <label className={styles.fieldLabel} htmlFor="library-paper-abstract">
            Abstract
          </label>
          <TextBox
            id="library-paper-abstract"
            mode="input"
            value={draftAbstract}
            onChange={setDraftAbstract}
            rows={5}
            ariaLabel="Paper abstract"
          />

          <label className={styles.fieldLabel} htmlFor="library-paper-tags">
            Tags
            <span className={styles.hint}>one per line</span>
          </label>
          <TextBox
            id="library-paper-tags"
            mode="input"
            value={draftTags}
            onChange={setDraftTags}
            rows={2}
            ariaLabel="Paper tags"
          />

          <p className={styles.fieldLabel}>Linked notes</p>
          {linkedNotes.length === 0 ? (
            <p className={styles.empty}>No notes linked to this paper.</p>
          ) : (
            <ul className={styles.linkedList}>
              {linkedNotes.map((note) => (
                <li key={note.id} className={styles.linkedItem}>
                  {note.title}
                </li>
              ))}
            </ul>
          )}

          <div className={styles.actions}>
            <button
              type="button"
              className={styles.primaryButton}
              onClick={() => void savePaper()}
              disabled={isSaving || !draftTitle.trim()}
            >
              {isSaving ? 'Saving…' : 'Save metadata'}
            </button>
            <a
              className={styles.secondaryButton}
              href={`/api/papers/${selectedPaper.id}/file`}
              target="_blank"
              rel="noreferrer"
            >
              Open PDF
            </a>
          </div>
        </Box>
      )}

      {selectedNote && (
        <Box header="Note Details" icon={FileText} compact>
          <div className={styles.metaRow}>
            <span
              className={`${styles.source} ${styles[selectedNote.source_type]}`}
            >
              {selectedNote.source_type === 'handwritten'
                ? 'Handwritten'
                : 'Voice'}
            </span>
            <span className={statusClass(selectedNote.processing_status)}>
              {selectedNote.processing_status}
            </span>
            <span className={styles.metaText}>
              {selectedNote.source_type === 'handwritten'
                ? selectedNote.page_count != null
                  ? `${selectedNote.page_count} pages`
                  : 'Pages unknown'
                : selectedNote.review_status}
              {' · '}
              {selectedNote.chunk_count} chunks
              {selectedNote.original_filename
                ? ` · ${selectedNote.original_filename}`
                : ''}
            </span>
          </div>

          {selectedNote.processing_error && (
            <p className={styles.processingError}>
              {selectedNote.processing_error}
            </p>
          )}

          <label className={styles.fieldLabel} htmlFor="library-note-title">
            Title
          </label>
          <TextBox
            id="library-note-title"
            mode="input"
            value={draftTitle}
            onChange={setDraftTitle}
            rows={2}
            ariaLabel="Note title"
          />

          {selectedNote.source_type === 'voice' && (
            <>
              <label className={styles.fieldLabel} htmlFor="library-note-summary">
                Summary
              </label>
              <TextBox
                id="library-note-summary"
                mode="input"
                value={draftSummary}
                onChange={setDraftSummary}
                rows={3}
                ariaLabel="Note summary"
              />

              <label
                className={styles.fieldLabel}
                htmlFor="library-note-observations"
              >
                Observations
                <span className={styles.hint}>one per line</span>
              </label>
              <TextBox
                id="library-note-observations"
                mode="input"
                value={draftObservations}
                onChange={setDraftObservations}
                rows={3}
                ariaLabel="Note observations"
              />

              <label
                className={styles.fieldLabel}
                htmlFor="library-note-hypotheses"
              >
                Hypotheses
                <span className={styles.hint}>one per line</span>
              </label>
              <TextBox
                id="library-note-hypotheses"
                mode="input"
                value={draftHypotheses}
                onChange={setDraftHypotheses}
                rows={3}
                ariaLabel="Note hypotheses"
              />

              <label
                className={styles.fieldLabel}
                htmlFor="library-note-questions"
              >
                Questions
                <span className={styles.hint}>one per line</span>
              </label>
              <TextBox
                id="library-note-questions"
                mode="input"
                value={draftQuestions}
                onChange={setDraftQuestions}
                rows={3}
                ariaLabel="Note questions"
              />

              <label
                className={styles.fieldLabel}
                htmlFor="library-note-next-steps"
              >
                Next steps
                <span className={styles.hint}>one per line</span>
              </label>
              <TextBox
                id="library-note-next-steps"
                mode="input"
                value={draftNextSteps}
                onChange={setDraftNextSteps}
                rows={3}
                ariaLabel="Note next steps"
              />
            </>
          )}

          {selectedNote.source_type === 'handwritten' && (
            <>
              <label className={styles.fieldLabel} htmlFor="library-note-text">
                Extracted text
                <span className={styles.hint}>editable; saving re-embeds</span>
              </label>
              <TextBox
                id="library-note-text"
                mode="input"
                value={draftExtractedText}
                onChange={setDraftExtractedText}
                rows={10}
                ariaLabel="Extracted note text"
              />
            </>
          )}

          <label className={styles.fieldLabel} id="library-note-papers-label">
            Linked papers
            <span className={styles.hint}>used when comparing those papers</span>
          </label>
          {papers.length === 0 ? (
            <p className={styles.empty}>No papers in the library yet.</p>
          ) : (
            <ul
              className={styles.checkList}
              aria-labelledby="library-note-papers-label"
            >
              {papers.map((paper) => (
                <li key={paper.id}>
                  <label className={styles.checkItem}>
                    <input
                      type="checkbox"
                      checked={draftPaperIds.includes(paper.id)}
                      onChange={() => togglePaper(paper.id)}
                    />
                    <span>
                      {paper.title}
                      {selectedNote.paper_links?.find(
                        (link) => link.paper_id === paper.id
                      )?.source === 'ai' && (
                        <span className={styles.aiBadge}> AI</span>
                      )}
                    </span>
                  </label>
                </li>
              ))}
            </ul>
          )}

          <RelatedPapersPanel
            noteId={selectedNote.id}
            isReady={selectedNote.processing_status === 'ready'}
            relatedGeneratedAt={selectedNote.related_generated_at}
            updatedAt={selectedNote.updated_at}
            paperLinks={selectedNote.paper_links ?? []}
            papers={papers}
            onNoteUpdate={(updated) => {
              setNotes((current) =>
                current.map((item) => (item.id === updated.id ? updated : item))
              );
              setDraftPaperIds(updated.paper_ids ?? []);
            }}
          />

          <label className={styles.fieldLabel} htmlFor="library-note-tags">
            Tags
            <span className={styles.hint}>one per line</span>
          </label>
          <TextBox
            id="library-note-tags"
            mode="input"
            value={draftTags}
            onChange={setDraftTags}
            rows={2}
            ariaLabel="Note tags"
          />

          <div className={styles.actions}>
            <button
              type="button"
              className={styles.primaryButton}
              onClick={() => void saveNote()}
              disabled={isSaving || !draftTitle.trim()}
            >
              {isSaving ? 'Saving…' : 'Save'}
            </button>
            {selectedNote.source_type === 'handwritten' && (
              <a
                className={styles.secondaryButton}
                href={`/api/notes/${selectedNote.id}/file`}
                target="_blank"
                rel="noreferrer"
              >
                Open PDF
              </a>
            )}
          </div>
        </Box>
      )}
    </div>
  );
}
