import { ChevronDown, ChevronRight, Mic, Plus, Square, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import styles from './ChatResearch.module.css';
import type {
  ChatApproval,
  ChatArtifact,
  ChatEvent,
  ChatMessage,
  ChatProgress,
  ChatResearchProps,
  ChatSubagent,
  ChatToolStep,
  ChatTurn,
  WorkspaceState,
} from '../types';
import { TextBox } from './TextBox';
import { useSpeechToComposer } from '../hooks/useSpeechToComposer';

const TOOL_LABELS: Record<string, string> = {
  search_library: 'Searching the library',
  web_search: 'Searching the web',
  list_papers: 'Listing papers',
  list_notes: 'Listing notes',
  list_documents: 'Listing documents',
  read_paper: 'Reading a paper',
  read_note: 'Reading a note',
  read_document: 'Reading a document',
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
  link_note: 'Linking a note',
  unlink_note: 'Unlinking a note',
  connect_note: 'Connecting a note',
  fetch_tool_result: 'Reading a truncated result',
};

export const MAIN_AGENT = 'main';

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

function emptyTurn(
  question: string,
  attachments: ChatTurn['attachments'] = []
): ChatTurn {
  return {
    question,
    steps: [],
    answer: '',
    citations: [],
    artifacts: [],
    todos: [],
    subagents: [],
    compactions: [],
    notices: [],
    approvals: [],
    progress: null,
    attachments,
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

export function CompactionNotices({
  items,
}: {
  items: ChatTurn['compactions'];
}) {
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

export function HarnessNotices({ items }: { items: ChatTurn['notices'] }) {
  if (items.length === 0) {
    return null;
  }
  return (
    <>
      {items.map((item, index) => (
        <p key={index} className={styles.compactNotice} role="status">
          {item.message}
        </p>
      ))}
    </>
  );
}

export function ProgressLine({ progress }: { progress: ChatProgress | null }) {
  if (!progress) {
    return null;
  }
  const total = progress.total > 0 ? `/${progress.total}` : '';
  return (
    <p className={styles.compactNotice} role="status">
      {progress.tool} · {progress.phase}: {progress.label} ({progress.done}
      {total}
      {progress.cached ? ', reused' : ''})
    </p>
  );
}

function argumentPreview(value: Record<string, unknown>): string {
  const entries = Object.entries(value).slice(0, 4);
  if (entries.length === 0) {
    return '';
  }
  return entries
    .map(([key, item]) => `${key}: ${String(item).slice(0, 80)}`)
    .join(' · ');
}

async function sendApproval(requestId: string, approved: boolean): Promise<void> {
  const response = await fetch('/api/chat/approve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request_id: requestId, approved }),
  });
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }
}

export function ApprovalCards({ items }: { items: ChatApproval[] }) {
  if (items.length === 0) {
    return null;
  }
  return (
    <div className={styles.approvalList}>
      {items.map((item) => (
        <div key={item.request_id} className={styles.approvalCard} role="status">
          <p className={styles.approvalTitle}>
            {item.status === 'pending'
              ? `Allow ${item.tool}?`
              : item.status === 'approved'
                ? `Allowed ${item.tool}`
                : item.status === 'timeout'
                  ? `${item.tool} timed out`
                  : `Declined ${item.tool}`}
          </p>
          {item.reason ? (
            <p className={styles.approvalReason}>{item.reason}</p>
          ) : null}
          {argumentPreview(item.arguments) ? (
            <p className={styles.approvalReason}>
              {argumentPreview(item.arguments)}
            </p>
          ) : null}
          {item.status === 'pending' ? (
            <div className={styles.approvalActions}>
              <button
                type="button"
                className={styles.approvalAllow}
                onClick={() => {
                  void sendApproval(item.request_id, true);
                }}
              >
                Allow
              </button>
              <button
                type="button"
                className={styles.approvalDeny}
                onClick={() => {
                  void sendApproval(item.request_id, false);
                }}
              >
                Decline
              </button>
            </div>
          ) : null}
        </div>
      ))}
    </div>
  );
}

export function StepList({ steps }: { steps: ChatToolStep[] }) {
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

export function NestedSubagents({
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
  conversationId = null,
  onConversation,
}: ChatResearchProps) {
  const [question, setQuestion] = useState('');
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [includePapers, setIncludePapers] = useState(true);
  const [includeVoiceNotes, setIncludeVoiceNotes] = useState(true);
  const [includeHandwrittenNotes, setIncludeHandwrittenNotes] = useState(true);
  const [includeDocuments, setIncludeDocuments] = useState(true);
  const [attachments, setAttachments] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const skipNextLoad = useRef(false);
  const onWorkspaceRef = useRef(onWorkspace);
  onWorkspaceRef.current = onWorkspace;
  const onTurnsChangeRef = useRef(onTurnsChange);
  onTurnsChangeRef.current = onTurnsChange;
  const onConversationRef = useRef(onConversation);
  onConversationRef.current = onConversation;
  const speech = useSpeechToComposer(question, setQuestion);
  const speechRef = useRef(speech);
  speechRef.current = speech;

  useEffect(() => {
    onTurnsChangeRef.current?.(turns);
  }, [turns]);

  useEffect(() => {
    if (skipNextLoad.current) {
      skipNextLoad.current = false;
      return;
    }
    if (!conversationId) {
      setTurns([]);
      return;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch(`/api/conversations/${conversationId}`);
        if (!response.ok) {
          throw new Error(await readErrorDetail(response));
        }
        const data = (await response.json()) as {
          turns: { payload: ChatTurn }[];
        };
        if (cancelled) {
          return;
        }
        setTurns(
          data.turns.map((item) => ({
            ...emptyTurn(item.payload?.question ?? ''),
            ...item.payload,
            attachments: item.payload?.attachments ?? [],
            approvals: item.payload?.approvals ?? [],
            isRunning: false,
          }))
        );
      } catch (err) {
        if (!cancelled) {
          setError('Could not load conversation: ' + networkErrorMessage(err));
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [conversationId]);

  const hasSource =
    includePapers ||
    includeVoiceNotes ||
    includeHandwrittenNotes ||
    includeDocuments;
  const canSend =
    (question.trim().length > 0 || attachments.length > 0) &&
    hasSource &&
    !isRunning &&
    !speech.isTranscribing &&
    !(speech.isListening && !speech.supportsLive);

  const addFiles = useCallback((incoming: FileList | File[]) => {
    const next: File[] = [];
    const rejected: string[] = [];
    for (const file of Array.from(incoming)) {
      if (file.type.startsWith('audio/')) {
        rejected.push(`${file.name} is audio — use the Voice notes tab`);
        continue;
      }
      const name = file.name.toLowerCase();
      const ok =
        file.type.includes('pdf') ||
        name.endsWith('.pdf') ||
        name.endsWith('.md') ||
        name.endsWith('.txt') ||
        name.endsWith('.csv') ||
        name.endsWith('.docx') ||
        file.type.includes('csv') ||
        file.type.includes('markdown') ||
        file.type.includes('wordprocessingml');
      if (!ok) {
        rejected.push(
          `${file.name} is not a PDF, Markdown, text, CSV, or Word file`
        );
        continue;
      }
      next.push(file);
    }
    if (rejected.length) {
      setError(rejected.join('. '));
    }
    if (next.length) {
      setAttachments((current) => [...current, ...next]);
    }
  }, []);

  const updateLastTurn = useCallback((update: (turn: ChatTurn) => ChatTurn) => {
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
  }, []);

  const send = useCallback(async () => {
    const trimmed = question.trim();
    const pendingFiles = attachments;
    if (!trimmed && pendingFiles.length === 0) {
      setError('Enter a question or attach a file.');
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
    const firstFile = pendingFiles[0];
    const userText =
      trimmed ||
      (pendingFiles.length === 1 && firstFile
        ? `Please file and use the attached file (${firstFile.name}).`
        : `Please file and use the ${pendingFiles.length} attached files.`);

    setQuestion('');
    setAttachments([]);
    setError(null);
    setIsRunning(true);
    setTurns((current) => [
      ...current,
      emptyTurn(
        userText,
        pendingFiles.map((file) => ({
          kind: 'document',
          id: file.name,
          title: file.name,
          filename: file.name,
          status: 'pending',
        }))
      ),
    ]);

    try {
      const formData = new FormData();
      formData.append(
        'messages',
        JSON.stringify([...history, { role: 'user', content: userText }])
      );
      formData.append('include_papers', String(includePapers));
      formData.append('include_voice_notes', String(includeVoiceNotes));
      formData.append(
        'include_handwritten_notes',
        String(includeHandwrittenNotes)
      );
      formData.append('include_documents', String(includeDocuments));
      if (conversationId) {
        formData.append('conversation_id', conversationId);
      }
      for (const file of pendingFiles) {
        formData.append('files', file, file.name);
      }
      const response = await fetch('/api/chat', {
        method: 'POST',
        body: formData,
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
        } else if (event.type === 'notice') {
          updateLastTurn((turn) => ({
            ...turn,
            notices: [
              ...turn.notices,
              { kind: event.kind, message: event.message },
            ],
          }));
        } else if (event.type === 'progress') {
          // Only the latest matters: a comparison reports once per paper and
          // the researcher wants to know where it is, not where it has been.
          updateLastTurn((turn) => ({
            ...turn,
            progress: {
              tool: event.tool,
              phase: event.phase,
              label: event.label,
              done: event.done,
              total: event.total,
              cached: event.cached,
            },
          }));
        } else if (event.type === 'approval') {
          updateLastTurn((turn) => {
            const next = [...(turn.approvals ?? [])];
            const index = next.findIndex(
              (item) => item.request_id === event.request_id
            );
            const card: ChatApproval = {
              request_id: event.request_id,
              tool: event.tool,
              arguments: event.arguments,
              reason: event.reason,
              status: event.status,
            };
            if (index >= 0) {
              next[index] = card;
            } else {
              next.push(card);
            }
            return { ...turn, approvals: next };
          });
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
          updateLastTurn((turn) => ({
            ...turn,
            model: event.model,
            progress: null,
          }));
        } else if (event.type === 'attachment') {
          updateLastTurn((turn) => {
            const next = turn.attachments.filter(
              (item) => item.id !== event.filename && item.id !== event.id
            );
            const pendingIndex = next.findIndex(
              (item) =>
                item.filename === event.filename && item.status === 'pending'
            );
            const filed = {
              kind: event.kind,
              id: event.id,
              title: event.title,
              filename: event.filename,
              status: event.status,
            };
            if (pendingIndex >= 0) {
              next[pendingIndex] = filed;
              return { ...turn, attachments: next };
            }
            return { ...turn, attachments: [...next, filed] };
          });
        } else if (event.type === 'conversation') {
          skipNextLoad.current = true;
          onConversationRef.current?.({ id: event.id, title: event.title });
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
    attachments,
    turns,
    hasSource,
    includePapers,
    includeVoiceNotes,
    includeHandwrittenNotes,
    includeDocuments,
    conversationId,
    updateLastTurn,
  ]);

  const composer = (
    <div
      className={`${styles.composer} ${styles.composerDocked} ${
        isDragging ? styles.composerDragging : ''
      }`}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={(event) => {
        event.preventDefault();
        setIsDragging(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        setIsDragging(false);
        if (event.dataTransfer.files?.length) {
          addFiles(event.dataTransfer.files);
        }
      }}
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
        <label className={styles.check}>
          <input
            type="checkbox"
            checked={includeDocuments}
            onChange={(e) => setIncludeDocuments(e.target.checked)}
          />
          Documents
        </label>
      </fieldset>

      {attachments.length > 0 && (
        <ul className={styles.chips}>
          {attachments.map((file, index) => (
            <li key={`${file.name}-${index}`} className={styles.chip}>
              <span>{file.name}</span>
              <button
                type="button"
                className={styles.chipRemove}
                aria-label={`Remove ${file.name}`}
                onClick={() =>
                  setAttachments((current) =>
                    current.filter((_, itemIndex) => itemIndex !== index)
                  )
                }
              >
                <X className={styles.chipIcon} />
              </button>
            </li>
          ))}
        </ul>
      )}

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
        placeholder="Ask, attach a file, compare papers, or capture a note…"
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
          className={styles.plusButton}
          onClick={() => fileInputRef.current?.click()}
          disabled={isRunning}
          aria-label="Attach a file"
          title="Attach PDF, Markdown, text, CSV, or Word"
        >
          <Plus className={styles.plusIcon} />
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.md,.txt,.csv,.docx,application/pdf,text/markdown,text/plain,text/csv,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
          className={styles.hiddenInput}
          onChange={(event) => {
            if (event.target.files?.length) {
              addFiles(event.target.files);
            }
            event.target.value = '';
          }}
        />
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
      </div>
    </div>
  );

  if (!composerSlot) {
    return null;
  }
  return createPortal(composer, composerSlot);
}
