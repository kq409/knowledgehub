import { ClipboardList, Columns2, MessageSquareText, Mic, X } from 'lucide-react';
import { useState } from 'react';
import styles from './WorkspaceHost.module.css';
import type { ChatTurn, CompareResponse, WorkspaceState } from '../types';
import {
  CompareCitationList,
  CompareSynthesisView,
  CompareTable,
} from './CompareResult';
import { ChatThread } from './ChatThread';
import { EvalPanel } from './EvalPanel';
import { VoiceNotes } from './VoiceNotes';

type WorkspaceTab = 'conversation' | 'panel';

function workspaceTitle(workspace: WorkspaceState): string {
  if (workspace.module === 'compare') {
    return 'Compare';
  }
  if (workspace.module === 'eval') {
    return 'Eval';
  }
  return 'Voice notes';
}

function CompareWorkspace({ result }: { result: CompareResponse }) {
  const [showEvidence, setShowEvidence] = useState(true);

  return (
    <div className={styles.compare}>
      <h3 className={styles.compareTitle}>
        Comparison of {result.papers.length} paper
        {result.papers.length === 1 ? '' : 's'}
      </h3>
      <CompareTable result={result} />
      <CompareSynthesisView synthesis={result.synthesis} />
      {result.citations.length > 0 && (
        <>
          <button
            type="button"
            className={styles.evidenceToggle}
            aria-expanded={showEvidence}
            onClick={() => setShowEvidence((current) => !current)}
          >
            {result.citations.length} evidence snippet
            {result.citations.length === 1 ? '' : 's'}
          </button>
          {showEvidence && <CompareCitationList result={result} />}
        </>
      )}
    </div>
  );
}

export function WorkspaceHost({
  workspace,
  turns,
  onClose,
  showComposer = false,
  composerSlotRef,
}: {
  workspace: WorkspaceState | null;
  turns: ChatTurn[];
  onClose: () => void;
  showComposer?: boolean;
  composerSlotRef?: (node: HTMLDivElement | null) => void;
}) {
  const [tab, setTab] = useState<WorkspaceTab>('conversation');
  const [workspaceSnapshot, setWorkspaceSnapshot] = useState(workspace);
  const hasConversation = turns.length > 0;

  if (workspace !== workspaceSnapshot) {
    setWorkspaceSnapshot(workspace);
    setTab(workspace ? 'panel' : 'conversation');
  }

  if (!workspace && !hasConversation) {
    return (
      <div className={styles.empty}>
        <p className={styles.emptyText}>
          Send a message in Chat. The conversation appears here. Compare tables
          and voice notes still open in this panel when a task needs them.
        </p>
      </div>
    );
  }

  const showingPanel = tab === 'panel' && workspace != null;
  const PanelIcon =
    workspace?.module === 'compare'
      ? Columns2
      : workspace?.module === 'eval'
        ? ClipboardList
        : Mic;

  return (
    <section
      className={styles.host}
      aria-label={showingPanel && workspace ? workspaceTitle(workspace) : 'Conversation'}
    >
      <div className={styles.bar}>
        {hasConversation && workspace ? (
          <div className={styles.tabs} role="tablist" aria-label="Workspace views">
            <button
              type="button"
              role="tab"
              aria-selected={!showingPanel}
              className={`${styles.tab} ${!showingPanel ? styles.tabActive : ''}`}
              onClick={() => setTab('conversation')}
            >
              <MessageSquareText className={styles.barIcon} />
              Conversation
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={showingPanel}
              className={`${styles.tab} ${showingPanel ? styles.tabActive : ''}`}
              onClick={() => setTab('panel')}
            >
              <PanelIcon className={styles.barIcon} />
              {workspaceTitle(workspace)}
            </button>
          </div>
        ) : (
          <div className={styles.barTitle}>
            {showingPanel && workspace ? (
              <>
                <PanelIcon className={styles.barIcon} />
                <h2>{workspaceTitle(workspace)}</h2>
              </>
            ) : (
              <>
                <MessageSquareText className={styles.barIcon} />
                <h2>Conversation</h2>
              </>
            )}
          </div>
        )}
        {workspace && (
          <button
            type="button"
            className={styles.closeButton}
            aria-label="Close workspace"
            onClick={onClose}
          >
            <X className={styles.closeIcon} />
          </button>
        )}
      </div>
      {showingPanel && workspace ? (
        <div className={styles.body}>
          {workspace.module === 'compare' ? (
            <CompareWorkspace result={workspace.result} />
          ) : workspace.module === 'eval' ? (
            <EvalPanel />
          ) : (
            <VoiceNotes />
          )}
        </div>
      ) : (
        <ChatThread turns={turns} />
      )}
      {showComposer ? (
        <div className={styles.composerDock} ref={composerSlotRef} />
      ) : null}
    </section>
  );
}
