import { ChevronDown, ChevronRight } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import styles from './ChatThread.module.css';
import traceStyles from './ChatResearch.module.css';
import type {
  ChatArtifact,
  ChatAttachment,
  ChatCitation,
  ChatThreadProps,
  ChatTodoItem,
  ChatTurn,
  ChatVerdict,
  CitationSourceType,
  MemoryArtifact,
  SkillArtifact,
  TodoStatus,
} from '../types';
import { MarkdownAnswer } from './MarkdownAnswer';
import { compactAnswerCitations } from '../citations';
import { Spinner } from './Spinner';
import {
  CompactionNotices,
  HarnessNotices,
  MAIN_AGENT,
  NestedSubagents,
  ProgressLine,
  StepList,
  ApprovalCards,
} from './ChatResearch';

function sourceLabel(sourceType: CitationSourceType): string {
  if (sourceType === 'paper') {
    return 'Paper';
  }
  if (sourceType === 'voice') {
    return 'Voice';
  }
  if (sourceType === 'web') {
    return 'Web';
  }
  if (sourceType === 'document') {
    return 'Document';
  }
  return 'Handwritten';
}

function badgeClass(sourceType: CitationSourceType): string {
  if (sourceType === 'paper') {
    return styles.paper ?? '';
  }
  if (sourceType === 'voice') {
    return styles.voice ?? '';
  }
  if (sourceType === 'web') {
    return styles.web ?? '';
  }
  if (sourceType === 'document') {
    return styles.document ?? '';
  }
  return styles.handwritten ?? '';
}

function attachmentLabel(item: ChatAttachment): string {
  const kind =
    item.kind === 'paper'
      ? 'Paper'
      : item.kind === 'note'
        ? 'Note'
        : 'Document';
  return `${kind}: ${item.title || item.filename}`;
}

function citationMeta(citation: ChatCitation): string {
  if (citation.source_type === 'web') {
    return citation.url ?? '';
  }
  const parts: string[] = [];
  if (citation.year != null) {
    parts.push(String(citation.year));
  }
  if (citation.page != null) {
    parts.push(`p. ${citation.page}`);
  }
  if (citation.section) {
    parts.push(citation.section);
  }
  parts.push(
    citation.similarity == null
      ? 'read in source order'
      : `similarity ${citation.similarity.toFixed(2)}`
  );
  return parts.join(' · ');
}

function verdictNotice(verdict: ChatVerdict | null): string | null {
  if (!verdict) {
    return null;
  }
  if (verdict.status === 'unsupported') {
    return (
      'The evidence check did not clear this answer. ' +
      (verdict.reason || 'Read the cited snippets before relying on it.')
    );
  }
  if (verdict.status === 'incomplete') {
    return (
      'This answer did not cover every part of the question. ' +
      (verdict.reason || 'Ask again if you still need the rest.')
    );
  }
  if (verdict.status === 'impossible') {
    return (
      'This request cannot be completed from the library. ' +
      (verdict.reason || 'Try a different question or add sources.')
    );
  }
  if (verdict.status === 'unchecked') {
    return 'The reviewer could not be reached, so this answer was only checked for fabricated citation numbers.';
  }
  return null;
}

function todoDotClass(status: TodoStatus): string {
  if (status === 'in_progress') {
    return styles.todoInProgress ?? '';
  }
  if (status === 'completed') {
    return styles.todoCompleted ?? '';
  }
  if (status === 'cancelled') {
    return styles.todoCancelled ?? '';
  }
  return styles.todoPending ?? '';
}

function SkillChip({ artifact }: { artifact: SkillArtifact }) {
  return (
    <p className={styles.notice} role="status">
      Loaded skill: {artifact.data.name}
      {artifact.data.description ? ` — ${artifact.data.description}` : ''}
    </p>
  );
}

function MemoryChip({ artifact }: { artifact: MemoryArtifact }) {
  const label =
    artifact.data.action === 'forgot'
      ? `Forgot: ${artifact.data.key}`
      : `Remembered: ${artifact.data.key}`;
  return (
    <p className={styles.notice} role="status">
      {label}
    </p>
  );
}

function TurnArtifact({ artifact }: { artifact: ChatArtifact }) {
  if (artifact.kind === 'comparison') {
    return (
      <p className={styles.notice} role="status">
        Opened comparison in the workspace
      </p>
    );
  }
  if (artifact.kind === 'workspace') {
    return (
      <p className={styles.notice} role="status">
        Opened voice notes in the workspace
      </p>
    );
  }
  if (artifact.kind === 'skill') {
    return <SkillChip artifact={artifact} />;
  }
  return <MemoryChip artifact={artifact} />;
}

function TodoPanel({ items }: { items: ChatTodoItem[] }) {
  if (items.length === 0) {
    return null;
  }
  return (
    <div className={styles.todoPanel}>
      <h4 className={styles.todoTitle}>Plan</h4>
      <ol className={styles.todoList}>
        {items.map((item) => (
          <li key={item.id} className={styles.todoItem}>
            <span
              className={`${styles.todoDot} ${todoDotClass(item.status)}`}
              aria-hidden
            />
            <span
              className={
                item.status === 'cancelled'
                  ? styles.todoCancelledText
                  : undefined
              }
            >
              {item.content}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function Citations({ citations }: { citations: ChatCitation[] }) {
  const [open, setOpen] = useState(false);

  if (citations.length === 0) {
    return null;
  }

  return (
    <div className={styles.citeBlock}>
      <button
        type="button"
        className={styles.citeToggle}
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        {open ? (
          <ChevronDown className={styles.citeChevron} />
        ) : (
          <ChevronRight className={styles.citeChevron} />
        )}
        {citations.length} citation{citations.length === 1 ? '' : 's'}
      </button>
      {open && (
        <ol className={styles.citations}>
          {citations.map((citation) => (
            <li
              key={citation.chunk_id ?? citation.url ?? String(citation.index)}
              className={styles.citation}
            >
              <div className={styles.citationHeader}>
                <span className={styles.index}>[{citation.index}]</span>
                <span
                  className={`${styles.badge} ${badgeClass(citation.source_type)}`}
                >
                  {sourceLabel(citation.source_type)}
                </span>
                {citation.url ? (
                  <a
                    className={styles.citationTitle}
                    href={citation.url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {citation.title}
                  </a>
                ) : (
                  <span className={styles.citationTitle}>{citation.title}</span>
                )}
              </div>
              <p className={styles.citationMeta}>{citationMeta(citation)}</p>
              {citation.snippet ? (
                <p className={styles.snippet}>{citation.snippet}</p>
              ) : null}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function AssistantBody({ turn }: { turn: ChatTurn }) {
  const isRetrying = turn.isRunning && turn.verdict?.status === 'retrying';
  const notice = turn.isRunning ? null : verdictNotice(turn.verdict);
  const visible = compactAnswerCitations(turn.answer, turn.citations);

  return (
    <div className={styles.assistant}>
      <TodoPanel items={turn.todos} />

      <HarnessNotices items={turn.notices ?? []} />
      <ApprovalCards items={turn.approvals ?? []} />
      <ProgressLine progress={turn.progress ?? null} />

      {isRetrying && (
        <p className={styles.retrying} role="status">
          Rechecking the draft against the library…
        </p>
      )}

      {turn.isRunning && !turn.answer && (
        <div className={styles.thinking} role="status">
          <Spinner />
          <p>Thinking…</p>
        </div>
      )}

      {notice && (
        <p
          className={
            turn.verdict?.status === 'unsupported'
              ? styles.caveat
              : styles.softNotice
          }
          role="status"
        >
          {notice}
        </p>
      )}

      {turn.artifacts.map((artifact, artifactIndex) => (
        <TurnArtifact key={artifactIndex} artifact={artifact} />
      ))}

      {visible.answer && <MarkdownAnswer text={visible.answer} />}

      {turn.error && (
        <p className={styles.caveat} role="status">
          {turn.error}
        </p>
      )}

      {turn.model && <p className={styles.modelMeta}>{turn.model}</p>}

      <Citations citations={visible.citations} />
    </div>
  );
}

export function ChatThread({ turns }: ChatThreadProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const [openTraces, setOpenTraces] = useState<Record<number, boolean>>({});
  const [openSubagents, setOpenSubagents] = useState<Record<string, boolean>>(
    {}
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' });
  }, [turns]);

  return (
    <div className={styles.thread}>
      <div className={styles.column}>
        {turns.length === 0 && (
          <p className={styles.empty}>
            Ask in your own words, or attach a file with the plus button. The
            file is filed as a paper, note, or document from its contents and
            your message.
          </p>
        )}
        {turns.map((turn, index) => {
          const isTraceOpen = openTraces[index] ?? turn.isRunning;
          const mainSteps = turn.steps.filter(
            (step) => step.agentId === MAIN_AGENT || !step.agentId
          );
          const hasTrace = mainSteps.length > 0 || turn.subagents.length > 0;
          return (
            <article
              key={index}
              className={styles.exchange}
              aria-label="Chat turn"
            >
              <div className={styles.userRow}>
                <div className={styles.userCluster}>
                  {(turn.attachments ?? []).length > 0 && (
                    <ul className={styles.attachmentList}>
                      {(turn.attachments ?? []).map((item) => (
                        <li
                          key={`${item.kind}-${item.id}`}
                          className={styles.attachment}
                        >
                          {attachmentLabel(item)}
                          {item.status && item.status !== 'ready'
                            ? ` · ${item.status}`
                            : ''}
                        </li>
                      ))}
                    </ul>
                  )}
                  <p className={styles.userBubble}>{turn.question}</p>
                </div>
              </div>
              {hasTrace ? (
                <div className={traceStyles.trace}>
                  <button
                    type="button"
                    className={traceStyles.traceToggle}
                    aria-expanded={isTraceOpen}
                    onClick={() =>
                      setOpenTraces((current) => ({
                        ...current,
                        [index]: !isTraceOpen,
                      }))
                    }
                  >
                    {isTraceOpen ? (
                      <ChevronDown className={traceStyles.chevron} />
                    ) : (
                      <ChevronRight className={traceStyles.chevron} />
                    )}
                    {mainSteps.length} tool step
                    {mainSteps.length === 1 ? '' : 's'}
                    {turn.subagents.length > 0
                      ? ` · ${turn.subagents.length} nested`
                      : ''}
                  </button>
                  {isTraceOpen && (
                    <>
                      <CompactionNotices items={turn.compactions} />
                      {mainSteps.length > 0 && <StepList steps={mainSteps} />}
                      <NestedSubagents
                        subagents={turn.subagents}
                        openIds={openSubagents}
                        onToggle={(id) =>
                          setOpenSubagents((current) => ({
                            ...current,
                            [id]: !(current[id] ?? true),
                          }))
                        }
                      />
                    </>
                  )}
                </div>
              ) : turn.isRunning ? (
                <p className={traceStyles.working} role="status">
                  Waiting for tools…
                </p>
              ) : turn.steps.length === 0 && turn.answer ? (
                <p className={traceStyles.working}>No tools used</p>
              ) : null}
              <AssistantBody turn={turn} />
            </article>
          );
        })}
        <div ref={endRef} />
      </div>
    </div>
  );
}
