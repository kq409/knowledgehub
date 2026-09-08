import { ClipboardList, Settings, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import styles from './Header.module.css';
import { MemoryPanel } from './MemoryPanel';

export function Header({ onOpenEval }: { onOpenEval?: () => void }) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!settingsOpen) {
      return;
    }
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (target && panelRef.current && !panelRef.current.contains(target)) {
        setSettingsOpen(false);
      }
    };
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, [settingsOpen]);

  return (
    <header className={styles.header}>
      <div className={styles.brand}>
        <h1 className={styles.title}>KnowledgeHub</h1>
        <p className={styles.subtitle}>
          Your notes and papers, searchable and citable
        </p>
      </div>
      <div className={styles.settingsWrap} ref={panelRef}>
        {onOpenEval ? (
          <button
            type="button"
            className={styles.settingsButton}
            aria-label="Open evaluation runs"
            onClick={onOpenEval}
          >
            <ClipboardList className={styles.settingsIcon} />
          </button>
        ) : null}
        <button
          type="button"
          className={`${styles.settingsButton} ${
            settingsOpen ? styles.settingsOpen : ''
          }`}
          aria-label={settingsOpen ? 'Close settings' : 'Open settings'}
          aria-expanded={settingsOpen}
          onClick={() => setSettingsOpen((current) => !current)}
        >
          {settingsOpen ? (
            <X className={styles.settingsIcon} />
          ) : (
            <Settings className={styles.settingsIcon} />
          )}
        </button>
        {settingsOpen && (
          <div className={styles.settingsPanel} role="dialog" aria-label="Settings">
            <MemoryPanel />
          </div>
        )}
      </div>
    </header>
  );
}
