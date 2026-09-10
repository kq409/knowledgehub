export interface EvalGrade {
  name: string;
  passed: boolean;
  score: number;
  detail: string;
  required?: boolean;
}

export interface EvalTrial {
  task_id: string;
  trial_index?: number;
  kind?: string;
  query: string;
  answer?: string;
  passed: boolean;
  score: number;
  grades: EvalGrade[];
  retrieved_titles?: string[];
  tools_called?: string[];
  gate_status?: string | null;
  source?: string;
  events?: Record<string, unknown>[];
}

export interface EvalRunSummary {
  suite?: string;
  tasks?: number;
  trials?: number;
  pass_at_1?: number;
  pass_hat_k?: number;
  pass_rate?: number;
  precision_at_k?: number;
  recall_at_k?: number;
  mrr?: number;
  regression_tasks?: number;
  quality_tasks?: number;
  regression_pass_rate?: number;
  regression_failed?: boolean;
  quality_pass_at_1?: number | null;
  quality_pass_hat_k?: number | null;
  gate?: string;
}

export interface EvalRun {
  id: string;
  suite: string;
  use_llm?: boolean;
  model?: string;
  created_at?: string;
  summary: EvalRunSummary | EvalRunSummary[];
  suites?: EvalRunSummary[];
  trials?: EvalTrial[];
}
