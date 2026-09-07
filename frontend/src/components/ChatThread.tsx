import { useEffect, useRef } from 'react';
import styles from './ChatThread.module.css';
import type {
  ChatArtifact,
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
import { Spinner } from './Spinner';

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
  return styles.handwritten ?? '';
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
                item.status === 'cancelled' ? styles.todoCancelledText : undefined
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
  if (citations.length === 0) {
    return null;
  }
  return (
    <ol className={styles.citations}>
      {citations.map((citation) => (
        <li
          key={citation.chunk_id ?? citation.url ?? String(citation.index)}
          className={styles.citation}
        >
          <div className={styles.citationHeader}>
            <span className={styles.index}>[{citation.index}]</span>
            <span className={`${styles.badge} ${badgeClass(citation.source_type)}`}>
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
  );
}

function AssistantBody({ turn }: { turn: ChatTurn }) {
  const isRetrying = turn.isRunning && turn.verdict?.status === 'retrying';
  const notice = turn.isRunning ? null : verdictNotice(turn.verdict);

  return (
    <div className={styles.assistant}>
      <TodoPanel items={turn.todos} />

      {isRetrying && (
        <p className={styles.retrying} role="status">
          The answer was not backed by the evidence gathered. Looking again…
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

      {turn.answer && <p className={styles.answer}>{turn.answer}</p>}

      {turn.error && (
        <p className={styles.caveat} role="status">
          {turn.error}
        </p>
      )}

      {turn.model && (
        <p className={styles.modelMeta}>
          {turn.model}
          {` · ${turn.citations.length} cited chunk(s)`}
        </p>
      )}

      <Citations citations={turn.citations} />
    </div>
  );
}

export function ChatThread({ turns }: ChatThreadProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' });
  }, [turns]);

  return (
    <div className={styles.thread}>
      <div className={styles.column}>
        {turns.map((turn, index) => (
          <article key={index} className={styles.exchange} aria-label="Chat turn">
            <div className={styles.userRow}>
              <p className={styles.userBubble}>{turn.question}</p>
            </div>
            <AssistantBody turn={turn} />
          </article>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}
