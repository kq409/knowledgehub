import { useCallback, useEffect, useMemo, useState } from 'react';
import styles from './EvalPanel.module.css';
import type {
  EvalGrade,
  EvalRun,
  EvalRunSummary,
  EvalTrial,
} from '../types/eval';
import { useAppStatus } from '../hooks/appStatus';

function asSummaries(run: EvalRun): EvalRunSummary[] {
  if (Array.isArray(run.suites) && run.suites.length > 0) {
    return run.suites;
  }
  if (Array.isArray(run.summary)) {
    return run.summary;
  }
  return run.summary ? [run.summary] : [];
}

function formatRate(value: number | undefined | null): string {
  if (value == null || Number.isNaN(value)) {
    return '—';
  }
  return `${Math.round(value * 100)}%`;
}

function summaryLine(run: EvalRun): string {
  const summaries = asSummaries(run);
  if (summaries.length === 0) {
    return run.suite;
  }
  const first = summaries[0];
  if (!first) {
    return run.suite;
  }
  const gate = first.gate ?? (first.regression_failed ? 'fail' : 'pass');
  return `${run.suite} · gate ${gate} · ${formatRate(first.pass_rate)} pass · ${
    first.tasks ?? '—'
  } tasks`;
}

function firstFailedTrial(run: EvalRun): EvalTrial | null {
  const trials = run.trials || [];
  return trials.find((item) => !item.passed) ?? trials[0] ?? null;
}

function gradeKey(grade: EvalGrade, index: number): string {
  return `${grade.name}-${index}`;
}

function eventLabel(event: Record<string, unknown>): string {
  const type = typeof event.type === 'string' ? event.type : 'event';
  if (type === 'tool_call' && typeof event.name === 'string') {
    return `tool_call ${event.name}`;
  }
  if (type === 'token' && typeof event.text === 'string') {
    const text = event.text.replace(/\s+/g, ' ').trim();
    return text ? `token ${text.slice(0, 80)}` : 'token';
  }
  if (type === 'verdict' && typeof event.status === 'string') {
    return `verdict ${event.status}`;
  }
  return type;
}

export function EvalPanel() {
  const { eval_run_allowed: evalRunAllowed } = useAppStatus();
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
    try {
      const response = await fetch(`/api/eval/runs/${id}`);
      if (!response.ok) {
        throw new Error(`Failed to load run ${id}`);
      }
      const run = (await response.json()) as EvalRun;
      setSelected(run);
      setTrial(firstFailedTrial(run));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load run');
    }
  }, []);

  const runAsk = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const response = await fetch('/api/eval/run?suite=ask-regression', {
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

  const summaries = useMemo(
    () => (selected ? asSummaries(selected) : []),
    [selected]
  );

  return (
    <div className={styles.panel}>
      <div className={styles.toolbar}>
        <p className={styles.lead}>
          Regression evals are the CD gate. Open a run to read the first failing
          transcript.
        </p>
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.button}
            onClick={() => void loadRuns()}
          >
            Refresh
          </button>
          {evalRunAllowed ? (
            <button
              type="button"
              className={styles.button}
              disabled={running}
              onClick={() => void runAsk()}
            >
              {running ? 'Running…' : 'Run Ask regression'}
            </button>
          ) : (
            <span className={styles.muted}>
              Eval runs are locked on this demo. They run in CI.
            </span>
          )}
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
              <ul className={styles.metrics}>
                {summaries.map((item) => (
                  <li key={item.suite ?? 'suite'}>
                    <strong>{item.suite ?? selected.suite}</strong>
                    {` gate ${item.gate ?? '—'} · pass@1 ${formatRate(
                      item.pass_at_1
                    )} · pass^k ${formatRate(item.pass_hat_k)} · regression ${formatRate(
                      item.regression_pass_rate
                    )}`}
                  </li>
                ))}
              </ul>
              <ul className={styles.trialList}>
                {(selected.trials || []).map((item) => (
                  <li key={`${item.task_id}-${item.trial_index ?? 0}`}>
                    <button
                      type="button"
                      className={`${styles.trialButton} ${
                        trial?.task_id === item.task_id &&
                        (trial.trial_index ?? 0) === (item.trial_index ?? 0)
                          ? styles.runActive
                          : ''
                      }`}
                      onClick={() => setTrial(item)}
                    >
                      <span className={item.passed ? styles.pass : styles.fail}>
                        {item.passed ? 'PASS' : 'FAIL'}
                      </span>
                      {item.task_id}
                      {item.kind ? (
                        <span className={styles.runMeta}>{item.kind}</span>
                      ) : null}
                    </button>
                  </li>
                ))}
              </ul>
              {trial ? (
                <div className={styles.transcript}>
                  <h3>{trial.task_id}</h3>
                  <p className={styles.metaLine}>
                    {trial.kind ?? 'task'}
                    {trial.source ? ` · ${trial.source}` : ''}
                    {trial.gate_status ? ` · gate ${trial.gate_status}` : ''}
                    {` · ${trial.tools_called?.length ?? 0} tools`}
                  </p>
                  <p>
                    <strong>Query:</strong> {trial.query}
                  </p>
                  {trial.answer ? (
                    <p>
                      <strong>Answer:</strong> {trial.answer}
                    </p>
                  ) : null}
                  <ul className={styles.gradeList}>
                    {trial.grades.map((grade, index) => (
                      <li key={gradeKey(grade, index)}>
                        <span
                          className={grade.passed ? styles.pass : styles.fail}
                        >
                          {grade.passed ? 'ok' : 'no'}
                        </span>{' '}
                        {grade.name}
                        {grade.required === false ? ' (diagnostic)' : ''}:{' '}
                        {grade.detail}
                      </li>
                    ))}
                  </ul>
                  {trial.tools_called && trial.tools_called.length > 0 ? (
                    <p>
                      <strong>Tools:</strong> {trial.tools_called.join(', ')}
                    </p>
                  ) : null}
                  {trial.retrieved_titles &&
                  trial.retrieved_titles.length > 0 ? (
                    <p>
                      <strong>Retrieved:</strong>{' '}
                      {trial.retrieved_titles.join(', ')}
                    </p>
                  ) : null}
                  {trial.events && trial.events.length > 0 ? (
                    <ol className={styles.eventList}>
                      {trial.events.map((event, index) => (
                        <li key={`${eventLabel(event)}-${index}`}>
                          {eventLabel(event)}
                        </li>
                      ))}
                    </ol>
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
              No run selected. Pick a previous run. Failures open first.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
