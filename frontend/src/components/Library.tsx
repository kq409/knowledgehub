import { BookOpen, FileText, Trash2, Upload } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styles from './Library.module.css';
import type {
  AccessionStatus,
  LibraryDocument,
  LibraryNote,
  LibraryRecord,
  Paper,
  ProcessingStatus,
  RecordPage,
  SpaceInfo,
} from '../types';
import { Box } from './Box';
import { TextBox } from './TextBox';
import { RelatedPapersPanel } from './RelatedPapers';
import { useAppStatus } from '../hooks/appStatus';

type LibraryFilter = 'all' | 'papers' | 'notes' | 'documents';
type LibraryKind = 'paper' | 'note' | 'document';

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
  accessionStatus: AccessionStatus;
  revision: number;
  checksum?: string | null;
  chunkCount: number;
  paper?: Paper;
  note?: LibraryNote;
  document?: LibraryDocument;
  highlight?: string | null;
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

function digestLines(paper: Paper): { label: string; value: string }[] {
  const digest = paper.digest ?? {
    problem: '',
    method: '',
    key_results: '',
    limitations: '',
  };
  return [
    { label: 'Problem', value: digest.problem },
    { label: 'Method', value: digest.method },
    { label: 'Results', value: digest.key_results },
    { label: 'Limitations', value: digest.limitations },
  ].filter((item) => item.value.trim());
}

function statusClass(status: ProcessingStatus): string {
  return `${styles.status} ${styles[status]}`;
}

function accessionClass(status: AccessionStatus): string {
  return `${styles.status} ${styles[status]}`;
}

function kindLabel(item: LibraryItem): string {
  if (item.kind === 'paper') {
    return 'Paper';
  }
  if (item.kind === 'document') {
    return 'Document';
  }
  return item.note?.source_type === 'handwritten' ? 'Handwritten' : 'Voice';
}

function kindFromContent(
  contentType: LibraryRecord['content_type']
): LibraryKind {
  if (contentType === 'scholarly_article') {
    return 'paper';
  }
  if (contentType === 'research_note') {
    return 'note';
  }
  return 'document';
}

function contentFromFilter(filter: LibraryFilter): string | undefined {
  if (filter === 'papers') {
    return 'scholarly_article';
  }
  if (filter === 'notes') {
    return 'research_note';
  }
  if (filter === 'documents') {
    return 'document';
  }
  return undefined;
}

const ACTIVE_SPACE_KEY = 'kh-active-space';

function kindClass(item: LibraryItem): string {
  if (item.kind === 'paper') {
    return `${styles.source} ${styles.paper}`;
  }
  if (item.kind === 'document') {
    return `${styles.source} ${styles.document}`;
  }
  return `${styles.source} ${styles[item.note?.source_type ?? 'handwritten']}`;
}

function linkedPaperLabel(note: LibraryNote, papers: Paper[]): string {
  if (!note.paper_ids.length) {
    return '';
  }
  const titles = new Map(papers.map((paper) => [paper.id, paper.title]));
  return note.paper_ids
    .map((id) => titles.get(id) ?? 'Unknown paper')
    .join(', ');
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

export function Library({
  spaceId,
  recordIds,
  onSpaceChange,
  onRecordIdsChange,
}: {
  spaceId: string | null;
  recordIds: string[];
  onSpaceChange?: (id: string, name: string) => void;
  onRecordIdsChange?: (ids: string[]) => void;
}) {
  const { uploads_enabled: uploadsEnabled } = useAppStatus();
  const [papers, setPapers] = useState<Paper[]>([]);
  const [notes, setNotes] = useState<LibraryNote[]>([]);
  const [documents, setDocuments] = useState<LibraryDocument[]>([]);
  const [spaces, setSpaces] = useState<SpaceInfo[]>([]);
  const [records, setRecords] = useState<LibraryRecord[]>([]);
  const [query, setQuery] = useState('');
  const [uploadKind, setUploadKind] = useState<
    'auto' | 'paper' | 'note' | 'document'
  >('auto');
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const queryRef = useRef(query);
  queryRef.current = query;
  const [selected, setSelected] = useState<SelectedRef | null>(null);
  const [filter, setFilter] = useState<LibraryFilter>('all');
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
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

  const items = useMemo(() => {
    const byId = {
      paper: new Map(papers.map((item) => [item.id, item])),
      note: new Map(notes.map((item) => [item.id, item])),
      document: new Map(documents.map((item) => [item.id, item])),
    };
    return records.map((record) => {
      const kind = kindFromContent(record.content_type);
      const paper = kind === 'paper' ? byId.paper.get(record.id) : undefined;
      const note = kind === 'note' ? byId.note.get(record.id) : undefined;
      const document =
        kind === 'document' ? byId.document.get(record.id) : undefined;
      return {
        kind,
        id: record.id,
        title: record.title,
        updatedAt: record.updated_at,
        status:
          paper?.processing_status ??
          note?.processing_status ??
          document?.processing_status ??
          record.status,
        accessionStatus:
          paper?.accession_status ??
          note?.accession_status ??
          document?.accession_status ??
          record.accession_status ??
          'accessioned',
        revision:
          paper?.revision ??
          note?.revision ??
          document?.revision ??
          record.revision,
        checksum:
          paper?.sha256 ?? note?.sha256 ?? document?.sha256 ?? record.checksum,
        chunkCount:
          paper?.chunk_count ?? note?.chunk_count ?? document?.chunk_count ?? 0,
        paper,
        note,
        document,
        highlight: record.highlight,
      };
    });
  }, [records, papers, notes, documents]);
  const visibleItems = items;

  const selectedItem =
    items.find(
      (item) =>
        selected != null &&
        item.kind === selected.kind &&
        item.id === selected.id
    ) ?? null;
  const selectedPaper = selectedItem?.paper ?? null;
  const selectedNote = selectedItem?.note ?? null;
  const selectedDocument = selectedItem?.document ?? null;

  const hasPending = items.some(
    (item) =>
      item.status === 'pending' ||
      item.status === 'processing' ||
      item.paper?.digest_status === 'pending'
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

  const loadDocuments = useCallback(async () => {
    try {
      const response = await fetch('/api/documents');
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      setDocuments((await response.json()) as LibraryDocument[]);
    } catch (err) {
      console.error('Failed to load documents:', err);
      setError('Could not load documents: ' + networkErrorMessage(err));
    }
  }, []);

  const loadRecords = useCallback(async () => {
    try {
      const type = contentFromFilter(filter);
      const trimmed = queryRef.current.trim();
      if (trimmed) {
        const response = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: trimmed,
            space_id: spaceId,
            content_type: type,
            limit: 50,
          }),
        });
        if (!response.ok) {
          throw new Error(await readErrorDetail(response));
        }
        const page = (await response.json()) as RecordPage;
        setRecords(page.items);
        return;
      }
      const params = new URLSearchParams();
      if (spaceId) {
        params.set('space_id', spaceId);
      }
      if (type) {
        params.set('content_type', type);
      }
      params.set('limit', '50');
      const response = await fetch(`/api/records?${params.toString()}`);
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      const page = (await response.json()) as RecordPage;
      setRecords(page.items);
    } catch (err) {
      console.error('Failed to load records:', err);
      setError('Could not load records: ' + networkErrorMessage(err));
    }
  }, [filter, spaceId]);

  const loadSpaces = useCallback(async () => {
    try {
      const response = await fetch('/api/spaces');
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      const next = (await response.json()) as SpaceInfo[];
      setSpaces(next);
      if (next.length === 0) {
        return;
      }
      const stored = window.localStorage.getItem(ACTIVE_SPACE_KEY);
      const match =
        next.find((item) => item.id === spaceId) ??
        next.find((item) => item.id === stored) ??
        next.find((item) => item.slug === 'default') ??
        next[0];
      if (match && match.id !== spaceId) {
        window.localStorage.setItem(ACTIVE_SPACE_KEY, match.id);
        onSpaceChange?.(match.id, match.name);
      }
    } catch (err) {
      console.error('Failed to load spaces:', err);
      setError('Could not load spaces: ' + networkErrorMessage(err));
    }
  }, [onSpaceChange, spaceId]);

  const loadLibrary = useCallback(async () => {
    await Promise.all([
      loadPapers(),
      loadNotes(),
      loadDocuments(),
      loadRecords(),
    ]);
  }, [loadPapers, loadNotes, loadDocuments, loadRecords]);

  useEffect(() => {
    void loadSpaces();
  }, [loadSpaces]);

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
      setDraftYear(
        selectedPaper.year != null ? String(selectedPaper.year) : ''
      );
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
    if (selectedDocument) {
      setDraftTitle(selectedDocument.title);
      return;
    }
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
  }, [selectedPaper, selectedNote, selectedDocument]);

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

  const saveDocument = useCallback(async () => {
    if (!selectedDocument) {
      return;
    }
    setIsSaving(true);
    setError(null);
    try {
      const response = await fetch(`/api/documents/${selectedDocument.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: draftTitle.trim() || selectedDocument.title,
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      await loadDocuments();
    } catch (err) {
      setError('Save failed: ' + networkErrorMessage(err));
    } finally {
      setIsSaving(false);
    }
  }, [selectedDocument, draftTitle, loadDocuments]);

  const deleteItem = useCallback(
    async (item: LibraryItem) => {
      try {
        const path =
          item.kind === 'paper'
            ? `/api/papers/${item.id}`
            : item.kind === 'document'
              ? `/api/documents/${item.id}`
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

  const toggleScope = (id: string) => {
    const next = recordIds.includes(id)
      ? recordIds.filter((item) => item !== id)
      : [...recordIds, id];
    onRecordIdsChange?.(next);
  };

  const uploadFiles = async (incoming: FileList | File[]) => {
    const files = Array.from(incoming);
    if (files.length === 0) {
      return;
    }
    setIsUploading(true);
    setError(null);
    try {
      for (const file of files) {
        const form = new FormData();
        form.append('file', file, file.name);
        if (uploadKind !== 'auto') {
          form.append('kind', uploadKind);
        }
        const response = await fetch('/api/library/upload', {
          method: 'POST',
          body: form,
        });
        if (!response.ok) {
          throw new Error(await readErrorDetail(response));
        }
      }
      await loadLibrary();
    } catch (err) {
      setError('Upload failed: ' + networkErrorMessage(err));
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className={`${styles.library} ${styles.compact}`}>
      {error && (
        <div className={styles.error} role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      <Box header="Library" icon={BookOpen} compact>
        {spaces.length > 0 ? (
          <div className={styles.filters} role="tablist" aria-label="Spaces">
            {spaces.map((space) => (
              <button
                key={space.id}
                type="button"
                role="tab"
                aria-selected={space.id === spaceId}
                className={`${styles.filterChip} ${
                  space.id === spaceId ? styles.active : ''
                }`}
                onClick={() => {
                  window.localStorage.setItem(ACTIVE_SPACE_KEY, space.id);
                  onSpaceChange?.(space.id, space.name);
                }}
              >
                {space.name}
              </button>
            ))}
          </div>
        ) : null}
        <form
          className={styles.searchRow}
          onSubmit={(event) => {
            event.preventDefault();
            void loadRecords();
          }}
        >
          <input
            className={styles.searchInput}
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search a document number or title…"
            aria-label="Search this space"
          />
          <button type="submit" className={styles.secondaryButton}>
            Search
          </button>
        </form>
        <div className={styles.uploadRow}>
          {uploadsEnabled ? (
            <>
              <label className={styles.kindLabel} htmlFor="library-upload-kind">
                File as
              </label>
              <select
                id="library-upload-kind"
                className={styles.kindSelect}
                value={uploadKind}
                onChange={(event) =>
                  setUploadKind(
                    event.target.value as 'auto' | 'paper' | 'note' | 'document'
                  )
                }
              >
                <option value="auto">Auto (AI suggests)</option>
                <option value="paper">Paper</option>
                <option value="note">Note</option>
                <option value="document">Document</option>
              </select>
              <button
                type="button"
                className={styles.secondaryButton}
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
              >
                <Upload className={styles.icon} />
                {isUploading ? 'Uploading…' : 'Upload'}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                className={styles.hiddenInput}
                accept=".pdf,.md,.txt,.csv,.docx,application/pdf"
                onChange={(event) => {
                  if (event.target.files?.length) {
                    void uploadFiles(event.target.files);
                  }
                  event.target.value = '';
                }}
              />
            </>
          ) : (
            <span className={styles.hint}>
              Uploads are disabled on the public demo.
            </span>
          )}
          {recordIds.length > 0 ? (
            <>
              <span className={styles.hint}>
                Chat is scoped to {recordIds.length} file
                {recordIds.length === 1 ? '' : 's'}
              </span>
              <button
                type="button"
                className={styles.secondaryButton}
                onClick={() => onRecordIdsChange?.([])}
              >
                Clear
              </button>
            </>
          ) : (
            <span className={styles.hint}>Tick files to ask this set</span>
          )}
        </div>
        <div
          className={styles.filters}
          role="tablist"
          aria-label="Library filter"
        >
          {(
            [
              ['all', 'All'],
              ['papers', 'Papers'],
              ['notes', 'Notes'],
              ['documents', 'Documents'],
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
            Nothing in this space yet. Upload a file or switch spaces.
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
                  <label className={styles.scopeCheck}>
                    <input
                      type="checkbox"
                      checked={recordIds.includes(item.id)}
                      onChange={() => toggleScope(item.id)}
                      aria-label={`Ask about ${item.title}`}
                    />
                  </label>
                  <button
                    type="button"
                    className={styles.selectButton}
                    onClick={() =>
                      setSelected({ kind: item.kind, id: item.id })
                    }
                  >
                    <span className={styles.itemTitle}>{item.title}</span>
                    {item.highlight ? (
                      <span className={styles.highlight}>{item.highlight}</span>
                    ) : null}
                    <span className={styles.itemMeta}>
                      <span className={kindClass(item)}>{kindLabel(item)}</span>
                      <span className={statusClass(item.status)}>
                        {item.status}
                      </span>
                      <span className={accessionClass(item.accessionStatus)}>
                        {item.accessionStatus}
                      </span>
                      <span>r{item.revision}</span>
                      <span>{item.chunkCount} chunks</span>
                      {item.note && linkedPaperLabel(item.note, papers) ? (
                        <span>
                          Linked: {linkedPaperLabel(item.note, papers)}
                        </span>
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

      {selectedPaper && selectedItem && (
        <Box header="Paper Details" icon={FileText} compact>
          <div className={styles.metaRow}>
            <span className={`${styles.source} ${styles.paper}`}>Paper</span>
            <span className={statusClass(selectedPaper.processing_status)}>
              {selectedPaper.processing_status}
            </span>
            <span className={accessionClass(selectedItem.accessionStatus)}>
              {selectedItem.accessionStatus}
            </span>
            <span className={styles.metaText}>
              {selectedPaper.page_count != null
                ? `${selectedPaper.page_count} pages`
                : 'Pages unknown'}
              {' · '}
              {selectedPaper.chunk_count} chunks
              {' · '}r{selectedItem.revision}
              {selectedItem.checksum
                ? ` · sha256 ${selectedItem.checksum.slice(0, 12)}…`
                : ''}
              {' · '}
              {selectedPaper.original_filename}
            </span>
          </div>

          {selectedPaper.processing_error && (
            <p className={styles.processingError}>
              {selectedPaper.processing_error}
            </p>
          )}

          <p className={styles.fieldLabel}>Understanding</p>
          {selectedPaper.digest_status === 'pending' ? (
            <p className={styles.empty}>Summarizing this paper…</p>
          ) : selectedPaper.digest_status === 'failed' &&
            !selectedPaper.summary ? (
            <p className={styles.empty}>Summary unavailable.</p>
          ) : selectedPaper.summary || digestLines(selectedPaper).length > 0 ? (
            <div className={styles.digestBlock}>
              {selectedPaper.summary ? (
                <p className={styles.digestSummary}>{selectedPaper.summary}</p>
              ) : null}
              <ul className={styles.digestList}>
                {digestLines(selectedPaper).map((item) => (
                  <li key={item.label}>
                    <span className={styles.digestKey}>{item.label}</span>
                    {item.value}
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className={styles.empty}>No stored summary yet.</p>
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

      {selectedDocument && selectedItem && (
        <Box header="Document Details" icon={FileText} compact>
          <div className={styles.metaRow}>
            <span className={`${styles.source} ${styles.document}`}>
              Document
            </span>
            <span className={statusClass(selectedDocument.processing_status)}>
              {selectedDocument.processing_status}
            </span>
            <span className={accessionClass(selectedItem.accessionStatus)}>
              {selectedItem.accessionStatus}
            </span>
            <span className={styles.metaText}>
              {selectedDocument.chunk_count} chunks
              {' · '}r{selectedItem.revision}
              {selectedItem.checksum
                ? ` · sha256 ${selectedItem.checksum.slice(0, 12)}…`
                : ''}
              {' · '}
              {selectedDocument.original_filename}
            </span>
          </div>
          {selectedDocument.processing_error && (
            <p className={styles.processingError}>
              {selectedDocument.processing_error}
            </p>
          )}
          <label className={styles.fieldLabel} htmlFor="library-document-title">
            Title
          </label>
          <TextBox
            id="library-document-title"
            mode="input"
            value={draftTitle}
            onChange={setDraftTitle}
            rows={2}
            ariaLabel="Document title"
          />
          <div className={styles.actions}>
            <button
              type="button"
              className={styles.primaryButton}
              onClick={() => void saveDocument()}
              disabled={isSaving || !draftTitle.trim()}
            >
              {isSaving ? 'Saving…' : 'Save'}
            </button>
            <a
              className={styles.secondaryButton}
              href={`/api/documents/${selectedDocument.id}/file`}
              target="_blank"
              rel="noreferrer"
            >
              Open file
            </a>
          </div>
        </Box>
      )}

      {selectedNote && selectedItem && (
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
            <span className={accessionClass(selectedItem.accessionStatus)}>
              {selectedItem.accessionStatus}
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
              <label
                className={styles.fieldLabel}
                htmlFor="library-note-summary"
              >
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
            <span className={styles.hint}>
              used when comparing those papers
            </span>
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
