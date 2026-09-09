import { NotebookPen, Copy, Check } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import styles from './VoiceNoteCard.module.css';
import { RelatedPapersPanel } from './RelatedPapers';
import type { Paper, ReviewStatus, VoiceNoteCardProps } from '../types';
import { Box } from './Box';
import { TextBox } from './TextBox';
import { Spinner } from './Spinner';

const LIST_FIELDS = [
  { key: 'observations', label: 'Observations' },
  { key: 'hypotheses', label: 'Hypotheses' },
  { key: 'questions', label: 'Questions' },
  { key: 'next_steps', label: 'Next steps' },
  { key: 'tags', label: 'Tags' },
] as const;

function listToLines(items: string[]): string {
  return items.join('\n');
}

function linesToList(value: string): string[] {
  return value
    .split('\n')
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatNoteForCopy(note: VoiceNoteCardProps['note']): string {
  const lists = LIST_FIELDS.map(({ key, label }) => {
    const items = note[key];
    if (!items.length) {
      return `${label}:`;
    }
    return `${label}:\n${items.map((item) => `- ${item}`).join('\n')}`;
  });
  return [`# ${note.title}`, note.summary, '', ...lists].join('\n');
}

export function VoiceNoteCard({
  note,
  isSaving,
  isExtracting,
  isCopied,
  onChange,
  onSave,
  onCopy,
  onSetStatus,
}: VoiceNoteCardProps) {
  const [papers, setPapers] = useState<Paper[]>([]);
  const paperIds = note.paper_ids ?? [];
  const linkByPaper = new Map(
    (note.paper_links ?? []).map((link) => [link.paper_id, link])
  );
  const noteRef = useRef(note);
  const onChangeRef = useRef(onChange);
  useEffect(() => {
    noteRef.current = note;
    onChangeRef.current = onChange;
  });

  useEffect(() => {
    const load = async () => {
      try {
        const response = await fetch('/api/papers');
        if (!response.ok) {
          return;
        }
        setPapers((await response.json()) as Paper[]);
      } catch {
        // Linking is optional; the note still saves without the paper list.
      }
    };
    void load();
  }, []);

  useEffect(() => {
    if (!note.id) {
      return;
    }
    const status = note.processing_status;
    const waitingProcess = status === 'pending' || status === 'processing';
    const updatedAt = note.updated_at
      ? new Date(note.updated_at).getTime()
      : Date.now();
    const waitingRelated =
      status === 'ready' &&
      !note.related_generated_at &&
      Date.now() - updatedAt < 90_000;
    if (!waitingProcess && !waitingRelated) {
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/notes/${note.id}`);
        if (!response.ok) {
          return;
        }
        const body = (await response.json()) as VoiceNoteCardProps['note'];
        const current = noteRef.current;
        onChangeRef.current({
          ...current,
          paper_ids: body.paper_ids,
          paper_links: body.paper_links,
          processing_status: body.processing_status,
          related_generated_at: body.related_generated_at,
        });
      } catch {
        // Keep the editor usable if polling fails.
      }
    }, 2500);
    return () => window.clearInterval(timer);
  }, [
    note.id,
    note.processing_status,
    note.related_generated_at,
    note.updated_at,
  ]);

  const updateField = <K extends keyof VoiceNoteCardProps['note']>(
    key: K,
    value: VoiceNoteCardProps['note'][K]
  ) => {
    const nextStatus: ReviewStatus =
      note.review_status === 'generated' ? 'draft' : note.review_status;
    onChange({ ...note, [key]: value, review_status: nextStatus });
  };

  return (
    <Box header="Research Note" icon={NotebookPen}>
      {isExtracting ? (
        <div className={styles.loading} aria-live="polite">
          <Spinner />
          <p>Extracting structured note…</p>
        </div>
      ) : (
        <>
          <div className={styles.meta}>
            <span className={`${styles.status} ${styles[note.review_status]}`}>
              {note.review_status}
            </span>
            {note.id ? (
              <span className={styles.savedLabel}>Saved</span>
            ) : (
              <span className={styles.savedLabel}>Unsaved preview</span>
            )}
          </div>

          <label className={styles.fieldLabel} htmlFor="note-title">
            Title
          </label>
          <TextBox
            id="note-title"
            mode="input"
            value={note.title}
            onChange={(value) => updateField('title', value)}
            rows={2}
            ariaLabel="Note title"
          />

          <label className={styles.fieldLabel} htmlFor="note-summary">
            Summary
          </label>
          <TextBox
            id="note-summary"
            mode="input"
            value={note.summary}
            onChange={(value) => updateField('summary', value)}
            rows={4}
            ariaLabel="Note summary"
          />

          {LIST_FIELDS.map(({ key, label }) => (
            <div key={key}>
              <label className={styles.fieldLabel} htmlFor={`note-${key}`}>
                {label}
                <span className={styles.hint}>one item per line</span>
              </label>
              <TextBox
                id={`note-${key}`}
                mode="input"
                value={listToLines(note[key])}
                onChange={(value) => updateField(key, linesToList(value))}
                rows={key === 'tags' ? 2 : 3}
                ariaLabel={label}
              />
            </div>
          ))}

          <label className={styles.fieldLabel} id="voice-note-papers-label">
            Linked papers
            <span className={styles.hint}>optional</span>
          </label>
          {papers.length === 0 ? (
            <p className={styles.hint}>No papers in the library yet.</p>
          ) : (
            <ul
              className={styles.checkList}
              aria-labelledby="voice-note-papers-label"
            >
              {papers.map((paper) => (
                <li key={paper.id}>
                  <label className={styles.checkItem}>
                    <input
                      type="checkbox"
                      checked={paperIds.includes(paper.id)}
                      onChange={() => {
                        const next = paperIds.includes(paper.id)
                          ? paperIds.filter((id) => id !== paper.id)
                          : [...paperIds, paper.id];
                        updateField('paper_ids', next);
                      }}
                    />
                    <span>
                      {paper.title}
                      {linkByPaper.get(paper.id)?.source === 'ai' && (
                        <span className={styles.aiBadge}> AI</span>
                      )}
                    </span>
                  </label>
                </li>
              ))}
            </ul>
          )}

          <RelatedPapersPanel
            noteId={note.id}
            isReady={note.processing_status === 'ready'}
            relatedGeneratedAt={note.related_generated_at}
            updatedAt={note.updated_at}
            paperLinks={note.paper_links ?? []}
            papers={papers}
            onNoteUpdate={(updated) => {
              onChange({
                ...note,
                paper_ids: updated.paper_ids,
                paper_links: updated.paper_links,
                processing_status: updated.processing_status,
                related_generated_at: updated.related_generated_at,
                updated_at: updated.updated_at,
              });
            }}
          />

          <div className={styles.actions}>
            <button
              type="button"
              className={styles.primaryButton}
              onClick={onSave}
              disabled={isSaving || !note.title.trim()}
            >
              {isSaving ? 'Saving…' : note.id ? 'Update note' : 'Save note'}
            </button>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={() => onCopy(formatNoteForCopy(note))}
            >
              {isCopied ? (
                <>
                  <Check className={styles.icon} />
                  Copied
                </>
              ) : (
                <>
                  <Copy className={styles.icon} />
                  Copy
                </>
              )}
            </button>
          </div>

          <div className={styles.statusActions}>
            <button
              type="button"
              className={styles.statusButton}
              onClick={() => onSetStatus('reviewed')}
              disabled={note.review_status === 'reviewed'}
            >
              Mark reviewed
            </button>
            <button
              type="button"
              className={styles.statusButton}
              onClick={() => onSetStatus('accepted')}
              disabled={note.review_status === 'accepted'}
            >
              Mark accepted
            </button>
          </div>
        </>
      )}
    </Box>
  );
}
