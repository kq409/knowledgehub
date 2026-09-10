import { createContext, useContext } from 'react';

export interface AppStatus {
  status: string;
  demo_mode: boolean;
  whisper_enabled: boolean;
  grobid_enabled: boolean;
  uploads_enabled: boolean;
  eval_run_allowed: boolean;
  llm_model: string | null;
  embedding_model: string | null;
  git_sha: string;
  replicas: number;
  notice: string | null;
}

export const LOCAL_DEFAULTS: AppStatus = {
  status: 'ready',
  demo_mode: false,
  whisper_enabled: true,
  grobid_enabled: true,
  uploads_enabled: true,
  eval_run_allowed: true,
  llm_model: null,
  embedding_model: null,
  git_sha: 'unknown',
  replicas: 1,
  notice: null,
};

export const AppStatusContext = createContext<AppStatus>(LOCAL_DEFAULTS);

export function useAppStatus(): AppStatus {
  return useContext(AppStatusContext);
}
