import { BookOpen, Trash2, Upload, FileText } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './NotesLibrary.module.css';
import type { LibraryNote, Paper, ProcessingStatus } from '../types';
import { Box } from './Box';
import { TextBox } from './TextBox';
import { Spinner } from './Spinner';

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

function sourceLabel(sourceType: LibraryNote['source_type']): string {
  return sourceType === 'handwritten' ? 'Handwritten' : 'Voice';
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

export function NotesLibrary() {
  const [notes, setNotes] = useState<LibraryNote[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState('');
  const [draftSummary, setDraftSummary] = useState('');
  const [draftObservations, setDraftObservations] = useState('');
  const [draftHypotheses, setDraftHypotheses] = useState('');
  const [draftQuestions, setDraftQuestions] = useState('');
  const [draftNextSteps, setDraftNextSteps] = useState('');
  const [draftTags, setDraftTags] = useState('');
  const [draftExtractedText, setDraftExtractedText] = useState('');
  const [draftPaperId, setDraftPaperId] = useState('');
  const [papers, setPapers] = useState<Paper[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const selected = notes.find((note) => note.id === selectedId) ?? null;
  const hasPending = notes.some(
    (note) =>
      note.processing_status === 'pending' ||
      note.processing_status === 'processing'
  );

  const loadNotes = useCallback(async () => {
    try {
      const response = await fetch('/api/notes');
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      const data = (await response.json()) as LibraryNote[];
      setNotes(data);
    } catch (err) {
      console.error('Failed to load notes:', err);
      setError('Could not load notes: ' + networkErrorMessage(err));
    }
  }, []);

  const loadPapers = useCallback(async () => {
    try {
      const response = await fetch('/api/papers');
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      const data = (await response.json()) as Paper[];
      setPapers(data);
    } catch (err) {
      console.error('Failed to load papers:', err);
    }
  }, []);

  useEffect(() => {
    void loadNotes();
    void loadPapers();
  }, [loadNotes, loadPapers]);

  useEffect(() => {
    if (!hasPending) {
      return;
    }
    const timer = window.setInterval(() => {
      void loadNotes();
    }, 2500);
    return () => window.clearInterval(timer);
  }, [hasPending, loadNotes]);

  useEffect(() => {
    if (!selected) {
      return;
    }
    setDraftTitle(selected.title);
    setDraftSummary(selected.summary);
    setDraftObservations(listToLines(selected.observations));
    setDraftHypotheses(listToLines(selected.hypotheses));
    setDraftQuestions(listToLines(selected.questions));
    setDraftNextSteps(listToLines(selected.next_steps));
    setDraftTags(listToLines(selected.tags));
    setDraftExtractedText(selected.extracted_text ?? '');
    setDraftPaperId(selected.paper_id ?? '');
  }, [selected]);

  const uploadPdf = useCallback(
    async (file: File) => {
      if (!file.type.includes('pdf') && !file.name.toLowerCase().endsWith('.pdf')) {
        setError('Please upload a PDF file');
        return;
      }

      setIsUploading(true);
      setError(null);
      try {
        const formData = new FormData();
        formData.append('file', file, file.name);
        const response = await fetch('/api/notes/upload', {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) {
          throw new Error(await readErrorDetail(response));
        }
        const note = (await response.json()) as LibraryNote;
        setSelectedId(note.id);
        await loadNotes();
      } catch (err) {
        setError('Note upload failed: ' + networkErrorMessage(err));
      } finally {
        setIsUploading(false);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    },
    [loadNotes]
  );

  const saveNote = useCallback(async () => {
    if (!selected) {
      return;
    }
    setIsSaving(true);
    setError(null);
    try {
      const payload =
        selected.source_type === 'handwritten'
          ? {
              title: draftTitle.trim() || selected.title,
              tags: linesToList(draftTags),
              extracted_text: draftExtractedText.trim() || null,
              paper_id: draftPaperId || null,
            }
          : {
              title: draftTitle.trim() || selected.title,
              summary: draftSummary,
              observations: linesToList(draftObservations),
              hypotheses: linesToList(draftHypotheses),
              questions: linesToList(draftQuestions),
              next_steps: linesToList(draftNextSteps),
              tags: linesToList(draftTags),
              paper_id: draftPaperId || null,
            };
      const response = await fetch(`/api/notes/${selected.id}`, {
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
    selected,
    draftTitle,
    draftSummary,
    draftObservations,
    draftHypotheses,
    draftQuestions,
    draftNextSteps,
    draftTags,
    draftExtractedText,
    draftPaperId,
    loadNotes,
  ]);

  const deleteNote = useCallback(
    async (id: string) => {
      try {
        const response = await fetch(`/api/notes/${id}`, { method: 'DELETE' });
        if (!response.ok) {
          throw new Error(await readErrorDetail(response));
        }
        if (selectedId === id) {
          setSelectedId(null);
        }
        await loadNotes();
      } catch (err) {
        setError('Delete failed: ' + networkErrorMessage(err));
      }
    },
    [selectedId, loadNotes]
  );

  return (
    <div className={styles.library}>
      <Box header="Upload Handwritten Note" icon={Upload}>
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
          aria-label="Upload PDF note"
        >
          {isUploading ? (
            <>
              <Spinner />
              <p>Uploading and queuing for parsing…</p>
            </>
          ) : (
            <>
              <Upload className={styles.uploadIcon} />
              <p className={styles.dropTitle}>
                {isDragging ? 'Drop PDF here' : 'Upload Word-exported PDF note'}
              </p>
              <p className={styles.dropSubtitle}>
                Drag and drop or click to browse. Text extraction + embedding
                runs in the background.
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

      {error && (
        <div className={styles.error} role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      <Box header="Notes Library" icon={BookOpen}>
        {notes.length === 0 ? (
          <p className={styles.empty}>
            No notes yet. Save a voice note or upload a PDF to start.
          </p>
        ) : (
          <ul className={styles.list}>
            {notes.map((note) => (
              <li key={note.id}>
                <div
                  className={`${styles.item} ${
                    selectedId === note.id ? styles.selected : ''
                  }`}
                >
                  <button
                    type="button"
                    className={styles.selectButton}
                    onClick={() => setSelectedId(note.id)}
                  >
                    <span className={styles.itemTitle}>{note.title}</span>
                    <span className={styles.itemMeta}>
                      <span className={`${styles.source} ${styles[note.source_type]}`}>
                        {sourceLabel(note.source_type)}
                      </span>
                      <span className={statusClass(note.processing_status)}>
                        {note.processing_status}
                      </span>
                      <span>{note.chunk_count} chunks</span>
                      <span>{formatDate(note.updated_at)}</span>
                    </span>
                  </button>
                  <button
                    type="button"
                    className={styles.deleteButton}
                    aria-label={`Delete ${note.title}`}
                    onClick={() => void deleteNote(note.id)}
                  >
                    <Trash2 className={styles.icon} />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Box>

      {selected && (
        <Box header="Note Details" icon={FileText}>
          <div className={styles.metaRow}>
            <span className={`${styles.source} ${styles[selected.source_type]}`}>
              {sourceLabel(selected.source_type)}
            </span>
            <span className={statusClass(selected.processing_status)}>
              {selected.processing_status}
            </span>
            <span className={styles.metaText}>
              {selected.source_type === 'handwritten'
                ? selected.page_count != null
                  ? `${selected.page_count} pages`
                  : 'Pages unknown'
                : selected.review_status}
              {' · '}
              {selected.chunk_count} chunks
              {selected.original_filename ? ` · ${selected.original_filename}` : ''}
            </span>
          </div>

          {selected.processing_error && (
            <p className={styles.processingError}>{selected.processing_error}</p>
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

          {selected.source_type === 'voice' && (
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

              <label className={styles.fieldLabel} htmlFor="library-note-observations">
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

              <label className={styles.fieldLabel} htmlFor="library-note-hypotheses">
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

              <label className={styles.fieldLabel} htmlFor="library-note-questions">
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

              <label className={styles.fieldLabel} htmlFor="library-note-next-steps">
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

          {selected.source_type === 'handwritten' && (
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

          <label className={styles.fieldLabel} htmlFor="library-note-paper">
            Linked paper
            <span className={styles.hint}>used when comparing that paper</span>
          </label>
          <select
            id="library-note-paper"
            className={styles.select}
            value={draftPaperId}
            onChange={(e) => setDraftPaperId(e.target.value)}
            aria-label="Linked paper"
          >
            <option value="">None</option>
            {papers.map((paper) => (
              <option key={paper.id} value={paper.id}>
                {paper.title}
              </option>
            ))}
          </select>

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
            {selected.source_type === 'handwritten' && (
              <a
                className={styles.secondaryButton}
                href={`/api/notes/${selected.id}/file`}
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
