import { MessageSquarePlus, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import styles from './ChatHistory.module.css';
import type { ConversationSummary } from '../types';

function formatWhen(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return '';
  }
  return date.toLocaleString();
}

export function ChatHistory({
  activeId,
  refreshToken = 0,
  onSelect,
  onNew,
}: {
  activeId: string | null;
  refreshToken?: number;
  onSelect: (id: string) => void;
  onNew: () => void;
}) {
  const [items, setItems] = useState<ConversationSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const response = await fetch('/api/conversations');
      if (!response.ok) {
        throw new Error(`Request failed (${response.status})`);
      }
      setItems((await response.json()) as ConversationSummary[]);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load history');
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, refreshToken]);

  const remove = async (id: string) => {
    const response = await fetch(`/api/conversations/${id}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      setError('Could not delete that conversation');
      return;
    }
    setItems((current) => current.filter((item) => item.id !== id));
    if (activeId === id) {
      onNew();
    }
  };

  return (
    <div className={styles.history}>
      <button type="button" className={styles.newButton} onClick={onNew}>
        <MessageSquarePlus className={styles.newIcon} />
        New chat
      </button>
      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}
      {items.length === 0 ? (
        <p className={styles.empty}>No conversations yet.</p>
      ) : (
        <ul className={styles.list}>
          {items.map((item) => (
            <li key={item.id}>
              <div
                className={`${styles.item} ${
                  activeId === item.id ? styles.active : ''
                }`}
              >
                <button
                  type="button"
                  className={styles.select}
                  onClick={() => onSelect(item.id)}
                >
                  <span className={styles.title}>{item.title}</span>
                  <span className={styles.meta}>
                    {item.turn_count} turn{item.turn_count === 1 ? '' : 's'}
                    {item.updated_at ? ` · ${formatWhen(item.updated_at)}` : ''}
                  </span>
                </button>
                <button
                  type="button"
                  className={styles.delete}
                  aria-label={`Delete ${item.title}`}
                  onClick={() => void remove(item.id)}
                >
                  <Trash2 className={styles.deleteIcon} />
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
