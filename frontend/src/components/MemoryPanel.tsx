import { Trash2 } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import styles from './MemoryPanel.module.css';
import type { AgentMemoryItem } from '../types';

export function MemoryPanel() {
  const [memories, setMemories] = useState<AgentMemoryItem[]>([]);
  const [memoriesError, setMemoriesError] = useState<string | null>(null);
  const [memoriesLoading, setMemoriesLoading] = useState(false);

  const loadMemories = useCallback(async () => {
    setMemoriesLoading(true);
    setMemoriesError(null);
    try {
      const response = await fetch('/api/memories');
      if (!response.ok) {
        throw new Error(`Failed to load memories (${response.status})`);
      }
      const data = (await response.json()) as AgentMemoryItem[];
      setMemories(data);
    } catch (err) {
      setMemoriesError(
        err instanceof Error ? err.message : 'Failed to load memories'
      );
    } finally {
      setMemoriesLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadMemories();
  }, [loadMemories]);

  const deleteMemory = useCallback(async (key: string) => {
    setMemoriesError(null);
    try {
      const response = await fetch(`/api/memories/${encodeURIComponent(key)}`, {
        method: 'DELETE',
      });
      if (!response.ok && response.status !== 204) {
        throw new Error(`Failed to delete (${response.status})`);
      }
      setMemories((current) => current.filter((item) => item.key !== key));
    } catch (err) {
      setMemoriesError(
        err instanceof Error ? err.message : 'Failed to delete memory'
      );
    }
  }, []);

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.title}>Agent memory</span>
        <button
          type="button"
          className={styles.refreshButton}
          onClick={() => void loadMemories()}
          disabled={memoriesLoading}
        >
          {memoriesLoading ? 'Loading…' : 'Refresh'}
        </button>
      </div>
      <p className={styles.hint}>
        Chat auto-recalls relevant preferences across conversations. Delete any
        that are wrong; the agent also writes with memory_write.
      </p>
      {memoriesError && (
        <p className={styles.error} role="alert">
          {memoriesError}
        </p>
      )}
      {memories.length === 0 && !memoriesLoading ? (
        <p className={styles.empty}>No memories yet.</p>
      ) : (
        <ul className={styles.list}>
          {memories.map((item) => (
            <li key={item.id} className={styles.item}>
              <div className={styles.meta}>
                <span className={styles.key}>{item.key}</span>
                <span className={styles.category}>{item.category}</span>
              </div>
              <p className={styles.content}>{item.content}</p>
              <button
                type="button"
                className={styles.deleteButton}
                aria-label={`Delete memory ${item.key}`}
                onClick={() => void deleteMemory(item.key)}
              >
                <Trash2 className={styles.deleteIcon} />
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
