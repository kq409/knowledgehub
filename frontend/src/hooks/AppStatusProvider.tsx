import { useEffect, useState, type ReactNode } from 'react';
import { AppStatusContext, LOCAL_DEFAULTS, type AppStatus } from './appStatus';

export function AppStatusProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AppStatus>(LOCAL_DEFAULTS);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch('/api/status');
        if (!response.ok) {
          return;
        }
        const body = (await response.json()) as Partial<AppStatus>;
        if (cancelled) {
          return;
        }
        setStatus({
          ...LOCAL_DEFAULTS,
          ...body,
          whisper_enabled: body.whisper_enabled !== false,
          grobid_enabled: body.grobid_enabled !== false,
          uploads_enabled: body.uploads_enabled !== false,
          eval_run_allowed: body.eval_run_allowed !== false,
          replicas: body.replicas ?? 1,
        });
      } catch {
        // Keep local defaults if the API is still starting.
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <AppStatusContext.Provider value={status}>
      {children}
    </AppStatusContext.Provider>
  );
}
