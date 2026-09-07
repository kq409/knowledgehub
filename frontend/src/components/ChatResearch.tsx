import { ChevronDown, ChevronRight, Mic, Square } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import styles from './ChatResearch.module.css';
import type {
  ChatArtifact,
  ChatEvent,
  ChatMessage,
  ChatResearchProps,
  ChatSubagent,
  ChatToolStep,
  ChatTurn,
  WorkspaceState,
} from '../types';
import { Box } from './Box';
import { TextBox } from './TextBox';
import { useSpeechToComposer } from '../hooks/useSpeechToComposer';

const TOOL_LABELS: Record<string, string> = {
  search_library: 'Searching the library',
  list_papers: 'Listing papers',
  list_notes: 'Listing notes',
  read_paper: 'Reading a paper',
  read_note: 'Reading a note',
  compare_papers: 'Comparing papers',
  present_workspace: 'Opening a workspace',
  preview_note_extraction: 'Structuring a note',
  todo_write: 'Updating the plan',
  spawn_subagent: 'Starting nested research',
  list_skills: 'Listing skills',
  load_skill: 'Loading a skill',
  memory_search: 'Searching memories',
  memory_write: 'Remembering',
  memory_delete: 'Forgetting a memory',
};

const MAIN_AGENT = 'main';

function networkErrorMessage(err: unknown): string {
  if (err instanceof TypeError && err.message === 'Failed to fetch') {
    return 'Cannot reach the backend. Make sure the API server is running on port 8000.';
  }
  if (err instanceof Error && err.message) {
    return err.message;
  }
  return 'Unknown error';
}

async function readErrorDetail(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const data = JSON.parse(text) as { detail?: unknown };
    if (typeof data.detail === 'string' && data.detail.trim()) {
      return data.detail;
    }
  } catch {
    // Response was not JSON.
  }
  return text || `Request failed (${response.status})`;
}

function toolStepLabel(step: ChatToolStep): string {
  if (step.summary) {
    return step.summary;
  }
  const label = TOOL_LABELS[step.name] ?? step.name;
  const query = step.arguments.query;
  return typeof query === 'string' ? `${label}: "${query}"` : `${label}…`;
}

function wasSkipped(step: ChatToolStep): boolean {
  if (step.permission?.decision === 'skip') {
    return true;
  }
  return (step.permission?.reason ?? '').includes('tool budget');
}

function wasRefused(step: ChatToolStep): boolean {
  return (
    step.permission != null &&
    step.permission.decision !== 'allow' &&
    !wasSkipped(step)
  );
}

function stepDetail(step: ChatToolStep): string {
  if (wasSkipped(step)) {
    return `Skipped — ${step.permission?.reason ?? 'this turn’s tool budget is spent'}`;
  }
  if (wasRefused(step)) {
    return `Not allowed — it ${step.permission?.reason}`;
  }
  return toolStepLabel(step);
}

function emptyTurn(question: string): ChatTurn {
  return {
    question,
    steps: [],
    answer: '',
    citations: [],
    artifacts: [],
    todos: [],
    subagents: [],
    compactions: [],
    model: null,
    error: null,
    verdict: null,
    isRunning: true,
  };
}

function patchOpenStep(
  steps: ChatToolStep[],
  name: string,
  patch: Partial<ChatToolStep>
): ChatToolStep[] {
  const next = [...steps];
  for (let i = next.length - 1; i >= 0; i -= 1) {
    const step = next[i];
    if (step && step.summary === null && step.name === name) {
      next[i] = { ...step, ...patch };
      break;
    }
  }
  return next;
}

function workspaceFromArtifact(artifact: ChatArtifact): WorkspaceState | null {
  if (artifact.kind === 'comparison') {
    return { module: 'compare', result: artifact.data };
  }
  if (artifact.kind === 'workspace' && artifact.data.module === 'voice_notes') {
    return { module: 'voice_notes' };
  }
  return null;
}

function CompactionNotices({ items }: { items: ChatTurn['compactions'] }) {
  if (items.length === 0) {
    return null;
  }
  return (
    <>
      {items.map((item, index) => (
        <p key={index} className={styles.compactNotice} role="status">
          Context compacted
          {item.agentId !== MAIN_AGENT ? ` (${item.agentId})` : ''}:{' '}
          {Math.round(item.beforeChars / 1000)}k →{' '}
          {Math.round(item.afterChars / 1000)}k chars ({item.mode})
        </p>
      ))}
    </>
  );
}

function StepList({ steps }: { steps: ChatToolStep[] }) {
  return (
    <ol className={styles.steps}>
      {steps.map((step, stepIndex) => (
        <li
          key={stepIndex}
          className={`${styles.step} ${
            wasSkipped(step)
              ? (styles.skippedStep ?? '')
              : wasRefused(step)
                ? (styles.refusedStep ?? '')
                : ''
          }`}
        >
          <span className={styles.stepName}>{step.name}</span>
          <span className={styles.stepDetail}>{stepDetail(step)}</span>
        </li>
      ))}
    </ol>
  );
}

function NestedSubagents({
  subagents,
  openIds,
  onToggle,
}: {
  subagents: ChatSubagent[];
  openIds: Record<string, boolean>;
  onToggle: (id: string) => void;
}) {
  if (subagents.length === 0) {
    return null;
  }
  return (
    <>
      {subagents.map((sub) => {
        const isOpen = openIds[sub.id] ?? true;
        return (
          <div key={sub.id} className={styles.nestedTrace}>
            <button
              type="button"
              className={styles.traceToggle}
              aria-expanded={isOpen}
              onClick={() => onToggle(sub.id)}
            >
              {isOpen ? (
                <ChevronDown className={styles.chevron} />
              ) : (
                <ChevronRight className={styles.chevron} />
              )}
              Nested: {sub.goal}
            </button>
            <p className={styles.subagentMeta}>
              {sub.status}
              {sub.summary ? ` — ${sub.summary}` : ''}
            </p>
            {isOpen && sub.steps.length > 0 && <StepList steps={sub.steps} />}
          </div>
        );
      })}
    </>
  );
}

/** Split an SSE byte stream into decoded `data:` payloads. */
async function* readEvents(
  body: ReadableStream<Uint8Array>
): AsyncGenerator<ChatEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf('\n\n');
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf('\n\n');

      const line = frame
        .split('\n')
        .find((candidate) => candidate.startsWith('data: '));
      if (!line) {
        continue;
      }
      try {
        yield JSON.parse(line.slice(6)) as ChatEvent;
      } catch {
        // Ignore a frame we cannot parse; the stream keeps going.
      }
    }
  }
}

export function ChatResearch({
  onWorkspace,
  onTurnsChange,
  composerSlot,
}: ChatResearchProps) {
  const [question, setQuestion] = useState('');
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [includePapers, setIncludePapers] = useState(true);
  const [includeVoiceNotes, setIncludeVoiceNotes] = useState(true);
  const [includeHandwrittenNotes, setIncludeHandwrittenNotes] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openTraces, setOpenTraces] = useState<Record<number, boolean>>({});
  const [openSubagents, setOpenSubagents] = useState<Record<string, boolean>>({});
  const onWorkspaceRef = useRef(onWorkspace);
  onWorkspaceRef.current = onWorkspace;
  const onTurnsChangeRef = useRef(onTurnsChange);
  onTurnsChangeRef.current = onTurnsChange;
  const speech = useSpeechToComposer(question, setQuestion);
  const speechRef = useRef(speech);
  speechRef.current = speech;

  useEffect(() => {
    onTurnsChangeRef.current?.(turns);
  }, [turns]);

  const hasSource =
    includePapers || includeVoiceNotes || includeHandwrittenNotes;
  const canSend =
    question.trim().length > 0 &&
    hasSource &&
    !isRunning &&
    !speech.isTranscribing &&
    !(speech.isListening && !speech.supportsLive);

  const updateLastTurn = useCallback(
    (update: (turn: ChatTurn) => ChatTurn) => {
      setTurns((current) => {
        if (current.length === 0) {
          return current;
        }
        const next = [...current];
        const last = next[next.length - 1];
        if (!last) {
          return current;
        }
        next[next.length - 1] = update(last);
        return next;
      });
    },
    []
  );

  const send = useCallback(async () => {
    const trimmed = question.trim();
    if (!trimmed) {
      setError('Enter a question first.');
      return;
    }
    if (!hasSource) {
      setError('Select at least one source.');
      return;
    }
    if (speechRef.current.isListening) {
      speechRef.current.stop();
    }

    const history: ChatMessage[] = turns.flatMap((turn) =>
      turn.answer
        ? [
            { role: 'user' as const, content: turn.question },
            { role: 'assistant' as const, content: turn.answer },
          ]
        : []
    );

    setQuestion('');
    setError(null);
    setIsRunning(true);
    setTurns((current) => [...current, emptyTurn(trimmed)]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...history, { role: 'user', content: trimmed }],
          include_papers: includePapers,
          include_voice_notes: includeVoiceNotes,
          include_handwritten_notes: includeHandwrittenNotes,
        }),
      });
      if (!response.ok || !response.body) {
        throw new Error(await readErrorDetail(response));
      }

      for await (const event of readEvents(response.body)) {
        if (event.type === 'tool_call') {
          const agentId = event.agent_id ?? MAIN_AGENT;
          const step: ChatToolStep = {
            name: event.name,
            arguments: event.arguments,
            summary: null,
            permission: null,
            agentId,
          };
          updateLastTurn((turn) => {
            if (agentId === MAIN_AGENT) {
              return { ...turn, steps: [...turn.steps, step] };
            }
            return {
              ...turn,
              subagents: turn.subagents.map((sub) =>
                sub.id === agentId
                  ? { ...sub, steps: [...sub.steps, step] }
                  : sub
              ),
            };
          });
        } else if (event.type === 'tool_result') {
          const agentId = event.agent_id ?? MAIN_AGENT;
          updateLastTurn((turn) => {
            if (agentId === MAIN_AGENT) {
              return {
                ...turn,
                steps: patchOpenStep(turn.steps, event.name, {
                  summary: event.summary,
                  permission: event.permission ?? null,
                }),
              };
            }
            return {
              ...turn,
              subagents: turn.subagents.map((sub) =>
                sub.id === agentId
                  ? {
                      ...sub,
                      steps: patchOpenStep(sub.steps, event.name, {
                        summary: event.summary,
                        permission: event.permission ?? null,
                      }),
                    }
                  : sub
              ),
            };
          });
        } else if (event.type === 'todo') {
          updateLastTurn((turn) => ({ ...turn, todos: event.items }));
        } else if (event.type === 'subagent') {
          updateLastTurn((turn) => {
            const existing = turn.subagents.find((sub) => sub.id === event.id);
            if (!existing) {
              return {
                ...turn,
                subagents: [
                  ...turn.subagents,
                  {
                    id: event.id,
                    status: event.status,
                    goal: event.goal,
                    summary: event.summary ?? null,
                    steps: [],
                  },
                ],
              };
            }
            return {
              ...turn,
              subagents: turn.subagents.map((sub) =>
                sub.id === event.id
                  ? {
                      ...sub,
                      status: event.status,
                      summary: event.summary ?? sub.summary,
                    }
                  : sub
              ),
            };
          });
        } else if (event.type === 'compact') {
          updateLastTurn((turn) => ({
            ...turn,
            compactions: [
              ...turn.compactions,
              {
                mode: event.mode,
                beforeChars: event.before_chars,
                afterChars: event.after_chars,
                agentId: event.agent_id ?? MAIN_AGENT,
              },
            ],
          }));
        } else if (event.type === 'artifact') {
          const artifact = {
            kind: event.kind,
            tool: event.tool,
            data: event.data,
          } as ChatArtifact;
          const workspace = workspaceFromArtifact(artifact);
          if (workspace) {
            onWorkspaceRef.current?.(workspace);
          }
          updateLastTurn((turn) => ({
            ...turn,
            artifacts: [...turn.artifacts, artifact],
          }));
        } else if (event.type === 'verdict') {
          updateLastTurn((turn) => ({
            ...turn,
            verdict: {
              status: event.status,
              reason: event.reason,
              checked_by: event.checked_by,
              problems: event.problems,
            },
          }));
        } else if (event.type === 'token') {
          updateLastTurn((turn) => ({
            ...turn,
            answer: turn.answer + event.text,
          }));
        } else if (event.type === 'citations') {
          updateLastTurn((turn) => ({ ...turn, citations: event.citations }));
        } else if (event.type === 'done') {
          updateLastTurn((turn) => ({ ...turn, model: event.model }));
        } else if (event.type === 'error') {
          updateLastTurn((turn) => ({ ...turn, error: event.message }));
        }
      }
    } catch (err) {
      const message = networkErrorMessage(err);
      setError('Chat failed: ' + message);
      updateLastTurn((turn) => ({ ...turn, error: message }));
    } finally {
      updateLastTurn((turn) => ({ ...turn, isRunning: false }));
      setIsRunning(false);
    }
  }, [
    question,
    turns,
    hasSource,
    includePapers,
    includeVoiceNotes,
    includeHandwrittenNotes,
    updateLastTurn,
  ]);

  const docked = composerSlot != null;
  const hideInlineComposer = turns.length > 0;
  const composer = (
    <div
      className={`${styles.composer} ${docked ? styles.composerDocked : ''}`}
    >
      {(error || speech.error) && (
        <div className={styles.error} role="alert">
          <span>{error || speech.error}</span>
          <button
            type="button"
            onClick={() => {
              setError(null);
              speech.dismissError();
            }}
          >
            Dismiss
          </button>
        </div>
      )}

      <fieldset className={styles.sources} disabled={isRunning}>
        <legend className={styles.legend}>Sources</legend>
        <label className={styles.check}>
          <input
            type="checkbox"
            checked={includePapers}
            onChange={(e) => setIncludePapers(e.target.checked)}
          />
          Papers
        </label>
        <label className={styles.check}>
          <input
            type="checkbox"
            checked={includeVoiceNotes}
            onChange={(e) => setIncludeVoiceNotes(e.target.checked)}
          />
          Voice notes
        </label>
        <label className={styles.check}>
          <input
            type="checkbox"
            checked={includeHandwrittenNotes}
            onChange={(e) => setIncludeHandwrittenNotes(e.target.checked)}
          />
          Handwritten
        </label>
      </fieldset>

      <label className={styles.fieldLabel} htmlFor="chat-question">
        Message
      </label>
      <TextBox
        id="chat-question"
        mode="input"
        value={question}
        onChange={setQuestion}
        rows={3}
        isDisabled={isRunning || speech.isTranscribing}
        placeholder="Ask, compare papers, or capture a note…"
        ariaLabel="Research question"
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            if (canSend) {
              void send();
            }
          }
        }}
      />

      <div className={styles.actions}>
        <button
          type="button"
          className={`${styles.micButton} ${
            speech.isListening ? styles.micActive : ''
          }`}
          onClick={speech.toggle}
          disabled={isRunning || speech.isTranscribing}
          aria-label={
            speech.isListening
              ? 'Stop voice input'
              : speech.supportsLive
                ? 'Start voice input'
                : 'Record voice input'
          }
          title={
            speech.supportsLive
              ? 'Speak into the box'
              : 'Hold to record, then transcribe into the box'
          }
        >
          {speech.isListening ? (
            <Square className={styles.micIcon} />
          ) : (
            <Mic className={styles.micIcon} />
          )}
        </button>
        <button
          type="button"
          className={styles.primaryButton}
          onClick={() => void send()}
          disabled={!canSend}
        >
          {isRunning
            ? 'Working…'
            : speech.isTranscribing
              ? 'Transcribing…'
              : 'Send'}
        </button>
        {turns.length > 0 && !isRunning && (
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() => {
              setTurns([]);
              setOpenTraces({});
              setOpenSubagents({});
            }}
          >
            New conversation
          </button>
        )}
      </div>
    </div>
  );

  return (
    <div className={styles.chat}>
      <div className={styles.thread}>
        {turns.length === 0 && (
          <p className={styles.intro}>
            Ask in your own words. After you send, the conversation and this
            box move to the right; tool calls stay here.
          </p>
        )}

        {turns.map((turn, index) => {
          const isTraceOpen = openTraces[index] ?? turn.isRunning;
          const mainSteps = turn.steps.filter(
            (step) => step.agentId === MAIN_AGENT || !step.agentId
          );
          const hasTrace = mainSteps.length > 0 || turn.subagents.length > 0;
          return (
            <Box key={index} header={turn.question} compact>
              <CompactionNotices items={turn.compactions} />

              {hasTrace ? (
                <div className={styles.trace}>
                  <button
                    type="button"
                    className={styles.traceToggle}
                    aria-expanded={isTraceOpen}
                    onClick={() =>
                      setOpenTraces((current) => ({
                        ...current,
                        [index]: !isTraceOpen,
                      }))
                    }
                  >
                    {isTraceOpen ? (
                      <ChevronDown className={styles.chevron} />
                    ) : (
                      <ChevronRight className={styles.chevron} />
                    )}
                    {mainSteps.length} tool step
                    {mainSteps.length === 1 ? '' : 's'}
                    {turn.subagents.length > 0
                      ? ` · ${turn.subagents.length} nested`
                      : ''}
                  </button>
                  {isTraceOpen && (
                    <>
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
                <p className={styles.working} role="status">
                  Waiting for tools…
                </p>
              ) : (
                <p className={styles.working}>No tools used</p>
              )}

              {turn.error && (
                <p className={styles.caveat} role="status">
                  {turn.error}
                </p>
              )}
            </Box>
          );
        })}
      </div>

      {composerSlot
        ? createPortal(composer, composerSlot)
        : hideInlineComposer
          ? null
          : composer}
    </div>
  );
}
