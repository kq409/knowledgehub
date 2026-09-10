import { ClipboardList, Settings, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import styles from './Header.module.css';
import { MemoryPanel } from './MemoryPanel';

const DEMO_USER_KEY = 'kh-demo-user';
const DEMO_USERS = ['alice', 'bob'] as const;

function readDemoUser(): string {
  const stored = window.localStorage.getItem(DEMO_USER_KEY);
  return stored === 'bob' ? 'bob' : 'alice';
}

export function Header({ onOpenEval }: { onOpenEval?: () => void }) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [demoUser, setDemoUser] = useState(readDemoUser);
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

  const switchUser = (user: string) => {
    window.localStorage.setItem(DEMO_USER_KEY, user);
    setDemoUser(user);
    window.location.reload();
  };

  return (
    <header className={styles.header}>
      <div className={styles.brand}>
        <h1 className={styles.title}>KnowledgeHub</h1>
        <p className={styles.subtitle}>
          R&D library first; Chat is an action on this space
        </p>
        <p className={styles.aclBanner}>
          Demo ACL as <strong>{demoUser}</strong> via X-User-Id — not SSO.
        </p>
      </div>
      <div className={styles.settingsWrap} ref={panelRef}>
        <label className={styles.userSwitch}>
          <span className={styles.userLabel}>User</span>
          <select
            className={styles.userSelect}
            value={demoUser}
            aria-label="Demo user"
            onChange={(event) => switchUser(event.target.value)}
          >
            {DEMO_USERS.map((user) => (
              <option key={user} value={user}>
                {user}
              </option>
            ))}
          </select>
        </label>
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
          <div
            className={styles.settingsPanel}
            role="dialog"
            aria-label="Settings"
          >
            <MemoryPanel />
          </div>
        )}
      </div>
    </header>
  );
}
