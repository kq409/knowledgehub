import {
  BookOpen,
  ClipboardList,
  Columns2,
  MessageSquareText,
  Mic,
  X,
} from 'lucide-react';
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
import { Library } from './Library';
import { VoiceNotes } from './VoiceNotes';

type WorkspaceTab = 'conversation' | 'library' | 'panel';

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
  composerSlotRef,
}: {
  workspace: WorkspaceState | null;
  turns: ChatTurn[];
  onClose: () => void;
  composerSlotRef?: (node: HTMLDivElement | null) => void;
}) {
  const [tab, setTab] = useState<WorkspaceTab>('conversation');
  const [workspaceSnapshot, setWorkspaceSnapshot] = useState(workspace);

  if (workspace !== workspaceSnapshot) {
    setWorkspaceSnapshot(workspace);
    setTab(workspace ? 'panel' : 'conversation');
  }

  const showingPanel = tab === 'panel' && workspace != null;
  const showingLibrary = tab === 'library';
  const PanelIcon =
    workspace?.module === 'compare'
      ? Columns2
      : workspace?.module === 'eval'
        ? ClipboardList
        : Mic;

  return (
    <section
      className={styles.host}
      aria-label={
        showingPanel && workspace
          ? workspaceTitle(workspace)
          : showingLibrary
            ? 'Library'
            : 'Conversation'
      }
    >
      <div className={styles.bar}>
        <div
          className={styles.tabs}
          role="tablist"
          aria-label="Workspace views"
        >
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'conversation'}
            className={`${styles.tab} ${tab === 'conversation' ? styles.tabActive : ''}`}
            onClick={() => setTab('conversation')}
          >
            <MessageSquareText className={styles.barIcon} />
            Conversation
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={showingLibrary}
            className={`${styles.tab} ${showingLibrary ? styles.tabActive : ''}`}
            onClick={() => setTab('library')}
          >
            <BookOpen className={styles.barIcon} />
            Library
          </button>
          {workspace ? (
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
          ) : null}
        </div>
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
      ) : showingLibrary ? (
        <div className={styles.body}>
          <Library />
        </div>
      ) : (
        <ChatThread turns={turns} />
      )}
      {tab === 'conversation' ? (
        <div className={styles.composerDock} ref={composerSlotRef} />
      ) : null}
    </section>
  );
}
