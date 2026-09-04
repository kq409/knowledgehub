import { Library, Trash2 } from 'lucide-react';
import styles from './VoiceNoteList.module.css';
import type { VoiceNoteListProps } from '../types';
import { Box } from './Box';

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return '';
  }
  return date.toLocaleString();
}

export function VoiceNoteList({
  notes,
  selectedId,
  onSelect,
  onDelete,
}: VoiceNoteListProps) {
  return (
    <Box header="Saved Voice Notes" icon={Library}>
      {notes.length === 0 ? (
        <p className={styles.empty}>No saved notes yet.</p>
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
                  onClick={() => onSelect(note)}
                >
                  <span className={styles.itemTitle}>{note.title}</span>
                  <span className={styles.itemMeta}>
                    <span className={styles.status}>{note.review_status}</span>
                    <span>{formatDate(note.updated_at)}</span>
                  </span>
                </button>
                <button
                  type="button"
                  className={styles.deleteButton}
                  aria-label={`Delete ${note.title}`}
                  onClick={() => onDelete(note.id)}
                >
                  <Trash2 className={styles.icon} />
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Box>
  );
}
