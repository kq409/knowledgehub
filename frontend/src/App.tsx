import { useState } from 'react';
import styles from './App.module.css';
import { Header } from './components/Header';
import { Library } from './components/Library';
import { ChatResearch } from './components/ChatResearch';
import { WorkspaceHost } from './components/WorkspaceHost';
import type { ChatTurn, WorkspaceState } from './types';

function App() {
  const [workspace, setWorkspace] = useState<WorkspaceState | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [composerSlot, setComposerSlot] = useState<HTMLDivElement | null>(null);
  const hasConversation = turns.length > 0;

  return (
    <div className={styles.app}>
      <Header />
      <div className={styles.shell}>
        <aside className={styles.sidebar}>
          <div className={styles.libraryPane}>
            <Library />
          </div>
          <div className={styles.chatPane}>
            <ChatResearch
              onWorkspace={setWorkspace}
              onTurnsChange={setTurns}
              composerSlot={hasConversation ? composerSlot : null}
            />
          </div>
        </aside>
        <main className={styles.workspace}>
          <WorkspaceHost
            workspace={workspace}
            turns={turns}
            onClose={() => setWorkspace(null)}
            showComposer={hasConversation}
            composerSlotRef={setComposerSlot}
          />
        </main>
      </div>
    </div>
  );
}

export default App;
