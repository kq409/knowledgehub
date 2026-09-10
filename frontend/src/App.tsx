import { useCallback, useState } from 'react';
import styles from './App.module.css';
import { Header } from './components/Header';
import { ChatHistory } from './components/ChatHistory';
import { ChatResearch } from './components/ChatResearch';
import { WorkspaceHost } from './components/WorkspaceHost';
import { AppStatusProvider } from './hooks/AppStatusProvider';
import type { ChatTurn, WorkspaceState } from './types';

function App() {
  const [workspace, setWorkspace] = useState<WorkspaceState | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [composerSlot, setComposerSlot] = useState<HTMLDivElement | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [historyTick, setHistoryTick] = useState(0);
  const [spaceId, setSpaceId] = useState<string | null>(null);
  const [spaceName, setSpaceName] = useState<string | null>(null);
  const [recordIds, setRecordIds] = useState<string[]>([]);

  const handleSpaceChange = useCallback((id: string, name: string) => {
    setSpaceId(id);
    setSpaceName(name);
    setRecordIds([]);
  }, []);

  return (
    <AppStatusProvider>
      <div className={styles.app}>
        <Header onOpenEval={() => setWorkspace({ module: 'eval' })} />
        <div className={styles.shell}>
          <main className={styles.workspace}>
            <WorkspaceHost
              workspace={workspace}
              turns={turns}
              onClose={() => setWorkspace(null)}
              composerSlotRef={setComposerSlot}
              spaceId={spaceId}
              recordIds={recordIds}
              onSpaceChange={handleSpaceChange}
              onRecordIdsChange={setRecordIds}
            />
          </main>
          <aside className={styles.sidebar}>
            <ChatHistory
              activeId={conversationId}
              refreshToken={historyTick}
              onSelect={setConversationId}
              onNew={() => setConversationId(null)}
            />
          </aside>
        </div>
        <ChatResearch
          onWorkspace={setWorkspace}
          onTurnsChange={setTurns}
          composerSlot={composerSlot}
          conversationId={conversationId}
          spaceId={spaceId}
          spaceName={spaceName}
          recordIds={recordIds}
          onConversation={(meta) => {
            setConversationId(meta.id);
            setHistoryTick((current) => current + 1);
          }}
        />
      </div>
    </AppStatusProvider>
  );
}

export default App;
