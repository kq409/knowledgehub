import { useCallback, useEffect, useState } from 'react';
import styles from './EvalPanel.module.css';
import type { EvalRun, EvalTrial } from '../types/eval';

function summaryLine(run: EvalRun): string {
  const raw = Array.isArray(run.summary) ? run.summary[0] : run.summary;
  if (!raw) {
    return run.suite;
  }
  const rate =
    raw.pass_rate != null ? `${Math.round(raw.pass_rate * 100)}%` : '—';
  return `${run.suite} · ${rate} pass · ${raw.tasks ?? '—'} tasks`;
}

export function EvalPanel() {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [selected, setSelected] = useState<EvalRun | null>(null);
  const [trial, setTrial] = useState<EvalTrial | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);

  const loadRuns = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/eval/runs');
      if (!response.ok) {
        throw new Error(`Failed to load eval runs (${response.status})`);
      }
      setRuns((await response.json()) as EvalRun[]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load eval runs');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadRuns();
  }, [loadRuns]);

  const openRun = useCallback(async (id: string) => {
    setError(null);
    setTrial(null);
    try {
      const response = await fetch(`/api/eval/runs/${id}`);
      if (!response.ok) {
        throw new Error(`Failed to load run ${id}`);
      }
      setSelected((await response.json()) as EvalRun);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load run');
    }
  }, []);

  const runAsk = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const response = await fetch('/api/eval/run?suite=ask', {
        method: 'POST',
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as {
          detail?: string;
        } | null;
        throw new Error(body?.detail || `Eval failed (${response.status})`);
      }
      const manifest = (await response.json()) as EvalRun;
      await loadRuns();
      if (manifest.id) {
        await openRun(manifest.id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Eval run failed');
    } finally {
      setRunning(false);
    }
  }, [loadRuns, openRun]);

  return (
    <div className={styles.panel}>
      <div className={styles.toolbar}>
        <p className={styles.lead}>
          Latest Ask/Chat eval runs. Open a trial to read the transcript.
        </p>
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.button}
            onClick={() => void loadRuns()}
          >
            Refresh
          </button>
          <button
            type="button"
            className={styles.button}
            disabled={running}
            onClick={() => void runAsk()}
          >
            {running ? 'Running…' : 'Run Ask suite'}
          </button>
        </div>
      </div>
      {error ? <p className={styles.error}>{error}</p> : null}
      {loading && runs.length === 0 ? (
        <p className={styles.muted}>Loading…</p>
      ) : null}
      <div className={styles.columns}>
        <ul className={styles.runList}>
          {runs.map((run) => (
            <li key={run.id}>
              <button
                type="button"
                className={`${styles.runButton} ${
                  selected?.id === run.id ? styles.runActive : ''
                }`}
                onClick={() => void openRun(run.id)}
              >
                <span className={styles.runId}>{run.id}</span>
                <span className={styles.runMeta}>{summaryLine(run)}</span>
              </button>
            </li>
          ))}
        </ul>
        <div className={styles.detail}>
          {selected ? (
            <>
              <ul className={styles.trialList}>
                {(selected.trials || []).map((item) => (
                  <li key={`${item.task_id}-${item.trial_index ?? 0}`}>
                    <button
                      type="button"
                      className={styles.trialButton}
                      onClick={() => setTrial(item)}
                    >
                      <span className={item.passed ? styles.pass : styles.fail}>
                        {item.passed ? 'PASS' : 'FAIL'}
                      </span>
                      {item.task_id}
                    </button>
                  </li>
                ))}
              </ul>
              {trial ? (
                <div className={styles.transcript}>
                  <h3>{trial.task_id}</h3>
                  <p>
                    <strong>Query:</strong> {trial.query}
                  </p>
                  {trial.answer ? (
                    <p>
                      <strong>Answer:</strong> {trial.answer}
                    </p>
                  ) : null}
                  <ul>
                    {trial.grades.map((grade) => (
                      <li key={grade.name}>
                        {grade.passed ? 'ok' : 'no'} {grade.name}:{' '}
                        {grade.detail}
                      </li>
                    ))}
                  </ul>
                  {trial.tools_called && trial.tools_called.length > 0 ? (
                    <p>Tools: {trial.tools_called.join(', ')}</p>
                  ) : null}
                </div>
              ) : (
                <p className={styles.muted}>
                  Select a task to read its transcript.
                </p>
              )}
            </>
          ) : (
            <p className={styles.muted}>
              No run selected. Run the Ask suite or pick a previous run.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
