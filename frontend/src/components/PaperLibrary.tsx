import { BookOpen, Trash2, Upload, FileText } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './PaperLibrary.module.css';
import type { Paper, PaperStatus } from '../types';
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

function authorsToLines(authors: string[]): string {
  return authors.join('\n');
}

function linesToAuthors(value: string): string[] {
  return value
    .split('\n')
    .map((item) => item.trim())
    .filter(Boolean);
}

function tagsToLines(tags: string[]): string {
  return tags.join('\n');
}

function linesToTags(value: string): string[] {
  return value
    .split('\n')
    .map((item) => item.trim())
    .filter(Boolean);
}

function statusClass(status: PaperStatus): string {
  return `${styles.status} ${styles[status]}`;
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

export function PaperLibrary() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState('');
  const [draftAuthors, setDraftAuthors] = useState('');
  const [draftYear, setDraftYear] = useState('');
  const [draftAbstract, setDraftAbstract] = useState('');
  const [draftTags, setDraftTags] = useState('');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const selected = papers.find((paper) => paper.id === selectedId) ?? null;
  const hasPending = papers.some(
    (paper) =>
      paper.processing_status === 'pending' ||
      paper.processing_status === 'processing'
  );

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
      setError('Could not load papers: ' + networkErrorMessage(err));
    }
  }, []);

  useEffect(() => {
    void loadPapers();
  }, [loadPapers]);

  useEffect(() => {
    if (!hasPending) {
      return;
    }
    const timer = window.setInterval(() => {
      void loadPapers();
    }, 2500);
    return () => window.clearInterval(timer);
  }, [hasPending, loadPapers]);

  useEffect(() => {
    if (!selected) {
      return;
    }
    setDraftTitle(selected.title);
    setDraftAuthors(authorsToLines(selected.authors));
    setDraftYear(selected.year != null ? String(selected.year) : '');
    setDraftAbstract(selected.abstract ?? '');
    setDraftTags(tagsToLines(selected.tags));
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
        const response = await fetch('/api/papers', {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) {
          throw new Error(await readErrorDetail(response));
        }
        const paper = (await response.json()) as Paper;
        setSelectedId(paper.id);
        await loadPapers();
      } catch (err) {
        setError('Paper upload failed: ' + networkErrorMessage(err));
      } finally {
        setIsUploading(false);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    },
    [loadPapers]
  );

  const saveMetadata = useCallback(async () => {
    if (!selected) {
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
      const response = await fetch(`/api/papers/${selected.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: draftTitle.trim() || selected.title,
          authors: linesToAuthors(draftAuthors),
          year,
          abstract: draftAbstract.trim() || null,
          tags: linesToTags(draftTags),
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
    selected,
    draftTitle,
    draftAuthors,
    draftYear,
    draftAbstract,
    draftTags,
    loadPapers,
  ]);

  const deletePaper = useCallback(
    async (id: string) => {
      try {
        const response = await fetch(`/api/papers/${id}`, { method: 'DELETE' });
        if (!response.ok) {
          throw new Error(await readErrorDetail(response));
        }
        if (selectedId === id) {
          setSelectedId(null);
        }
        await loadPapers();
      } catch (err) {
        setError('Delete failed: ' + networkErrorMessage(err));
      }
    },
    [selectedId, loadPapers]
  );

  return (
    <div className={styles.library}>
      <Box header="Upload Paper" icon={Upload}>
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
          aria-label="Upload PDF paper"
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
                {isDragging ? 'Drop PDF here' : 'Upload research PDF'}
              </p>
              <p className={styles.dropSubtitle}>
                Drag and drop or click to browse. Parsing + embedding runs in the
                background.
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

      <Box header="Paper Library" icon={BookOpen}>
        {papers.length === 0 ? (
          <p className={styles.empty}>No papers yet. Upload a PDF to start.</p>
        ) : (
          <ul className={styles.list}>
            {papers.map((paper) => (
              <li key={paper.id}>
                <div
                  className={`${styles.item} ${
                    selectedId === paper.id ? styles.selected : ''
                  }`}
                >
                  <button
                    type="button"
                    className={styles.selectButton}
                    onClick={() => setSelectedId(paper.id)}
                  >
                    <span className={styles.itemTitle}>{paper.title}</span>
                    <span className={styles.itemMeta}>
                      <span className={statusClass(paper.processing_status)}>
                        {paper.processing_status}
                      </span>
                      <span>{paper.chunk_count} chunks</span>
                      <span>{formatDate(paper.updated_at)}</span>
                    </span>
                  </button>
                  <button
                    type="button"
                    className={styles.deleteButton}
                    aria-label={`Delete ${paper.title}`}
                    onClick={() => void deletePaper(paper.id)}
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
        <Box header="Paper Details" icon={FileText}>
          <div className={styles.metaRow}>
            <span className={statusClass(selected.processing_status)}>
              {selected.processing_status}
            </span>
            <span className={styles.metaText}>
              {selected.page_count != null
                ? `${selected.page_count} pages`
                : 'Pages unknown'}
              {' · '}
              {selected.chunk_count} chunks
              {' · '}
              {selected.original_filename}
            </span>
          </div>

          {selected.processing_error && (
            <p className={styles.processingError}>{selected.processing_error}</p>
          )}

          <label className={styles.fieldLabel} htmlFor="paper-title">
            Title
          </label>
          <TextBox
            id="paper-title"
            mode="input"
            value={draftTitle}
            onChange={setDraftTitle}
            rows={2}
            ariaLabel="Paper title"
          />

          <label className={styles.fieldLabel} htmlFor="paper-authors">
            Authors
            <span className={styles.hint}>one per line</span>
          </label>
          <TextBox
            id="paper-authors"
            mode="input"
            value={draftAuthors}
            onChange={setDraftAuthors}
            rows={3}
            ariaLabel="Paper authors"
          />

          <label className={styles.fieldLabel} htmlFor="paper-year">
            Year
          </label>
          <TextBox
            id="paper-year"
            mode="input"
            value={draftYear}
            onChange={setDraftYear}
            rows={1}
            ariaLabel="Paper year"
          />

          <label className={styles.fieldLabel} htmlFor="paper-abstract">
            Abstract
          </label>
          <TextBox
            id="paper-abstract"
            mode="input"
            value={draftAbstract}
            onChange={setDraftAbstract}
            rows={5}
            ariaLabel="Paper abstract"
          />

          <label className={styles.fieldLabel} htmlFor="paper-tags">
            Tags
            <span className={styles.hint}>one per line</span>
          </label>
          <TextBox
            id="paper-tags"
            mode="input"
            value={draftTags}
            onChange={setDraftTags}
            rows={2}
            ariaLabel="Paper tags"
          />

          <div className={styles.actions}>
            <button
              type="button"
              className={styles.primaryButton}
              onClick={() => void saveMetadata()}
              disabled={isSaving || !draftTitle.trim()}
            >
              {isSaving ? 'Saving…' : 'Save metadata'}
            </button>
            <a
              className={styles.secondaryButton}
              href={`/api/papers/${selected.id}/file`}
              target="_blank"
              rel="noreferrer"
            >
              Open PDF
            </a>
          </div>
        </Box>
      )}
    </div>
  );
}
