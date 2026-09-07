export type RecordingState = 'idle' | 'recording' | 'processing';

export type ReviewStatus = 'generated' | 'draft' | 'reviewed' | 'accepted';

export type ProcessingStatus = 'pending' | 'processing' | 'ready' | 'failed';

export type NoteLinkSource = 'researcher' | 'ai';

export type ConnectReasonMode = 'snippet' | 'llm';

export interface NotePaperLink {
  paper_id: string;
  source: NoteLinkSource;
  similarity: number | null;
  snippet: string | null;
  reason: string | null;
  reason_mode: ConnectReasonMode | null;
}

export interface StructuredNoteFields {
  title: string;
  summary: string;
  observations: string[];
  hypotheses: string[];
  questions: string[];
  next_steps: string[];
  tags: string[];
}

export interface VoiceNoteDraft extends StructuredNoteFields {
  id?: string;
  raw_transcript: string;
  cleaned_transcript: string;
  review_status: ReviewStatus;
  model?: string | null;
  prompt_version?: string | null;
  paper_ids?: string[];
  paper_links?: NotePaperLink[];
  processing_status?: ProcessingStatus;
  related_generated_at?: string | null;
  updated_at?: string;
}

export interface VoiceNote extends VoiceNoteDraft {
  id: string;
  created_at: string;
  updated_at: string;
}

export interface ExtractNoteResponse extends StructuredNoteFields {
  model: string;
  prompt_version: string;
}

export interface AppState {
  isRecording: boolean;
  isProcessing: boolean;
  rawText: string | null;
  cleanedText: string | null;
  isCleaningWithLLM: boolean;
  error: string | null;
  useLLM: boolean;
  isCopied: boolean;
  systemPrompt: string;
  isLoadingPrompt: boolean;
  isDragging: boolean;
}

export interface TranscriptionResponse {
  text: string;
}

export interface CleanTextResponse {
  cleaned_text: string;
}

export interface SystemPromptResponse {
  default_prompt: string;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface HeaderProps {}

export interface RecordButtonProps {
  isRecording: boolean;
  isProcessing: boolean;
  onStartRecording: () => Promise<void>;
  onStopRecording: () => void;
}

export interface UploadZoneProps {
  isProcessing: boolean;
  isDragging: boolean;
  onFileSelect: (file: File) => void;
  onDragEnter: () => void;
  onDragLeave: () => void;
  onDrop: (file: File) => void;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
}

export interface TextInputZoneProps {
  isProcessing: boolean;
  onTextSubmit: (text: string) => Promise<void>;
}

export interface SettingsPanelProps {
  useLLM: boolean;
  systemPrompt: string;
  isLoadingPrompt: boolean;
  onToggleLLM: (value: boolean) => void;
  onPromptChange: (value: string) => void;
}

export interface VoiceNoteCardProps {
  note: VoiceNoteDraft;
  isSaving: boolean;
  isExtracting: boolean;
  isCopied: boolean;
  onChange: (note: VoiceNoteDraft) => void;
  onSave: () => void;
  onCopy: (text: string) => void;
  onSetStatus: (status: ReviewStatus) => void;
}

export interface VoiceNoteListProps {
  notes: VoiceNote[];
  selectedId?: string;
  onSelect: (note: VoiceNote) => void;
  onDelete: (id: string) => void;
}

export type PaperStatus = ProcessingStatus;

export type NoteSourceType = 'voice' | 'handwritten';

export interface LibraryNote extends StructuredNoteFields {
  id: string;
  source_type: NoteSourceType;
  raw_transcript: string;
  cleaned_transcript: string;
  review_status: ReviewStatus;
  model?: string | null;
  prompt_version?: string | null;
  original_filename: string | null;
  page_count: number | null;
  extracted_text: string | null;
  paper_ids: string[];
  paper_links: NotePaperLink[];
  related_generated_at: string | null;
  processing_status: ProcessingStatus;
  processing_error: string | null;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export interface Paper {
  id: string;
  title: string;
  authors: string[];
  year: number | null;
  abstract: string | null;
  source: string;
  tags: string[];
  original_filename: string;
  page_count: number | null;
  processing_status: PaperStatus;
  processing_error: string | null;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export interface RelatedPaper {
  paper_id: string;
  title: string;
  year: number | null;
  similarity: number;
  snippet: string;
  reason: string;
  reason_mode: ConnectReasonMode;
  linked: boolean;
  source: NoteLinkSource | null;
  page: number | null;
  section: string | null;
}

export interface ConnectResponse {
  note_id: string;
  reason_mode: ConnectReasonMode;
  papers: RelatedPaper[];
  linked_paper_ids: string[];
  skipped_paper_ids: string[];
  model: string | null;
  prompt_version: string;
  generated_at: string;
}

export type CitationSourceType = 'paper' | 'voice' | 'handwritten' | 'web';

export type QueryKind = 'library' | 'field_wide';

export type ExternalSearchStatus = 'unavailable' | 'ready' | 'ran' | 'failed';

export interface AskCitation {
  index: number;
  source_type: CitationSourceType;
  source_id: string | null;
  chunk_id: string | null;
  title: string;
  page: number | null;
  section: string | null;
  year: number | null;
  snippet: string;
  similarity: number;
  url?: string | null;
}

export interface LibraryCoverage {
  paper_count: number;
  unique_retrieved_papers: number;
  years: number[];
}

export interface AskResponse {
  answer: string;
  insufficient_evidence: boolean;
  max_similarity: number | null;
  citations: AskCitation[];
  model: string;
  prompt_version: string;
  query_kind: QueryKind;
  library_coverage: LibraryCoverage;
  suggest_external_search: boolean;
  external_search_status: ExternalSearchStatus;
}

export type WorkspaceState =
  | { module: 'compare'; result: CompareResponse }
  | { module: 'voice_notes' };

/**
 * Evidence the agent read through a tool. `similarity` is null for chunks it
 * read in source order rather than retrieved by search.
 */
export interface ChatCitation extends Omit<AskCitation, 'similarity'> {
  similarity: number | null;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export type PermissionDecision = 'allow' | 'deny' | 'ask' | 'skip';

export interface ChatPermission {
  decision: PermissionDecision;
  reason: string;
}

export interface ChatToolStep {
  name: string;
  arguments: Record<string, unknown>;
  summary: string | null;
  permission: ChatPermission | null;
  agentId: string;
}

export type GateStatus =
  | 'supported'
  | 'unsupported'
  | 'unchecked'
  | 'retrying';

export interface ChatGateProblem {
  kind: string;
  detail: string;
}

export interface ChatVerdict {
  status: GateStatus;
  reason: string;
  checked_by: string;
  problems: ChatGateProblem[];
}

/** Structured tool output the chat renders itself instead of reading as prose. */
export interface ComparisonArtifact {
  kind: 'comparison';
  tool: string;
  data: CompareResponse;
}

export interface SkillArtifact {
  kind: 'skill';
  tool: string;
  data: { name: string; description: string };
}

export interface MemoryArtifact {
  kind: 'memory';
  tool: string;
  data: {
    action: 'remembered' | 'forgot';
    key: string;
    content?: string;
    category?: string;
  };
}

export type WorkspaceModuleId = 'voice_notes';

export interface WorkspaceArtifact {
  kind: 'workspace';
  tool: string;
  data: { module: WorkspaceModuleId };
}

export type ChatArtifact =
  | ComparisonArtifact
  | SkillArtifact
  | MemoryArtifact
  | WorkspaceArtifact;

export interface AgentMemoryItem {
  id: string;
  key: string;
  content: string;
  category: string;
  source_turn: string | null;
  created_at: string;
  updated_at: string;
}

export type TodoStatus =
  | 'pending'
  | 'in_progress'
  | 'completed'
  | 'cancelled';

export interface ChatTodoItem {
  id: string;
  content: string;
  status: TodoStatus;
}

export type SubagentStatus = 'started' | 'finished' | 'failed';

export interface ChatSubagent {
  id: string;
  status: SubagentStatus;
  goal: string;
  summary: string | null;
  steps: ChatToolStep[];
}

export interface ChatCompaction {
  mode: 'truncate' | 'summarize';
  beforeChars: number;
  afterChars: number;
  agentId: string;
}

export interface ChatTurn {
  question: string;
  steps: ChatToolStep[];
  answer: string;
  citations: ChatCitation[];
  artifacts: ChatArtifact[];
  todos: ChatTodoItem[];
  subagents: ChatSubagent[];
  compactions: ChatCompaction[];
  model: string | null;
  error: string | null;
  verdict: ChatVerdict | null;
  isRunning: boolean;
}

export interface ChatResearchProps {
  onWorkspace?: (workspace: WorkspaceState) => void;
  onTurnsChange?: (turns: ChatTurn[]) => void;
  composerSlot?: HTMLElement | null;
}

export interface ChatThreadProps {
  turns: ChatTurn[];
}

export type ChatEvent =
  | {
      type: 'tool_call';
      name: string;
      arguments: Record<string, unknown>;
      agent_id?: string;
    }
  | {
      type: 'tool_result';
      name: string;
      summary: string;
      permission?: ChatPermission;
      agent_id?: string;
    }
  | ({ type: 'artifact' } & ChatArtifact)
  | { type: 'token'; text: string }
  | { type: 'citations'; citations: ChatCitation[] }
  | ({ type: 'verdict' } & ChatVerdict)
  | { type: 'todo'; items: ChatTodoItem[] }
  | {
      type: 'subagent';
      id: string;
      status: SubagentStatus;
      goal: string;
      summary?: string;
    }
  | {
      type: 'compact';
      mode: 'truncate' | 'summarize';
      before_chars: number;
      after_chars: number;
      agent_id?: string;
    }
  | { type: 'done'; model: string; prompt_version: string }
  | { type: 'error'; message: string };

export const DEFAULT_COMPARE_DIMENSIONS = [
  'problem',
  'method',
  'dataset',
  'evaluation',
  'key_results',
  'strengths',
  'limitations',
] as const;

export type DefaultCompareDimension =
  (typeof DEFAULT_COMPARE_DIMENSIONS)[number];

export function dimensionLabel(value: string): string {
  return value.replace(/_/g, ' ');
}

export interface CompareLinkedNote {
  id: string;
  paper_id: string;
  title: string;
  source_type: CitationSourceType;
}

export interface ComparePaperResult {
  paper_id: string;
  title: string;
  year: number | null;
  values: Record<string, string>;
  linked_notes: CompareLinkedNote[];
}

export interface CompareSynthesis {
  agreements: string;
  disagreements: string;
  research_gap: string;
}

export interface CompareCitation {
  index: number;
  source_type: CitationSourceType;
  source_id: string;
  chunk_id: string;
  paper_id: string;
  title: string;
  page: number | null;
  section: string | null;
  year: number | null;
  snippet: string;
  similarity: number;
}

export interface CompareResponse {
  id: string;
  paper_ids: string[];
  dimensions: string[];
  papers: ComparePaperResult[];
  synthesis: CompareSynthesis;
  citations: CompareCitation[];
  model: string;
  prompt_version: string;
  created_at: string;
}

export interface CompareSummary {
  id: string;
  paper_ids: string[];
  paper_titles: string[];
  created_at: string;
}

export interface TranscriptionResultsProps {
  rawText: string | null;
  cleanedText: string | null;
  useLLM: boolean;
  isCopied: boolean;
  isCleaningWithLLM: boolean;
  isProcessing: boolean;
  isOriginalExpanded: boolean;
  onCopy: (text: string) => void;
  onToggleOriginalExpanded: () => void;
}

export interface ErrorMessageProps {
  message: string;
  onDismiss: () => void;
}

export interface TextBoxProps {
  value: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  mode: 'input' | 'display';
  variant?: 'default' | 'enhanced';
  isLoading?: boolean;
  isDisabled?: boolean;
  showCopyButton?: boolean;
  isCopied?: boolean;
  onCopy?: () => void;
  rows?: number;
  maxHeight?: string;
  ariaLabel?: string;
  id?: string;
  onKeyDown?: React.KeyboardEventHandler<HTMLTextAreaElement>;
}

export interface BoxProps {
  children: React.ReactNode;
  header?: string;
  icon?: React.ComponentType<{ className?: string }>;
  collapsible?: boolean;
  isExpanded?: boolean;
  onToggleExpanded?: () => void;
  className?: string;
  compact?: boolean;
}

export type AudioFileType =
  | 'audio/mpeg'
  | 'audio/wav'
  | 'audio/webm'
  | 'audio/ogg'
  | 'audio/x-m4a';

export const ACCEPTED_AUDIO_TYPES: AudioFileType[] = [
  'audio/mpeg',
  'audio/wav',
  'audio/webm',
  'audio/ogg',
  'audio/x-m4a',
];
