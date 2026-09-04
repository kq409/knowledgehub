import { MessageSquareText } from 'lucide-react';
import { useCallback, useState } from 'react';
import styles from './AskResearch.module.css';
import type {
  AskCitation,
  AskResponse,
  CitationSourceType,
  LibraryCoverage,
} from '../types';
import { Box } from './Box';
import { TextBox } from './TextBox';
import { Spinner } from './Spinner';

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

function sourceLabel(sourceType: CitationSourceType): string {
  if (sourceType === 'paper') {
    return 'Paper';
  }
  if (sourceType === 'voice') {
    return 'Voice';
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
  return styles.handwritten ?? '';
}

function citationMeta(citation: AskCitation): string {
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
  parts.push(`similarity ${citation.similarity.toFixed(2)}`);
  return parts.join(' · ');
}

function coverageLabel(coverage: LibraryCoverage): string {
  const paperLabel = coverage.paper_count === 1 ? 'paper' : 'papers';
  const yearPart =
    coverage.years.length > 0 ? ` · years ${coverage.years.join(', ')}` : '';
  return `${coverage.paper_count} ${paperLabel} in library${yearPart}`;
}

export function AskResearch() {
  const [question, setQuestion] = useState('');
  const [includePapers, setIncludePapers] = useState(true);
  const [includeVoiceNotes, setIncludeVoiceNotes] = useState(true);
  const [includeHandwrittenNotes, setIncludeHandwrittenNotes] = useState(true);
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);

  const canAsk =
    question.trim().length > 0 &&
    (includePapers || includeVoiceNotes || includeHandwrittenNotes) &&
    !isAsking;

  const ask = useCallback(async () => {
    const trimmed = question.trim();
    if (!trimmed) {
      setError('Enter a question first.');
      return;
    }
    if (!includePapers && !includeVoiceNotes && !includeHandwrittenNotes) {
      setError('Select at least one source.');
      return;
    }

    setIsAsking(true);
    setError(null);
    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: trimmed,
          include_papers: includePapers,
          include_voice_notes: includeVoiceNotes,
          include_handwritten_notes: includeHandwrittenNotes,
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      const data = (await response.json()) as AskResponse;
      setResult(data);
    } catch (err) {
      setError('Ask failed: ' + networkErrorMessage(err));
    } finally {
      setIsAsking(false);
    }
  }, [question, includePapers, includeVoiceNotes, includeHandwrittenNotes]);

  return (
    <div className={styles.ask}>
      <Box header="Ask My Research" icon={MessageSquareText}>
        <p className={styles.intro}>
          Search your paper library and notes, then get a grounded answer with
          citations.
        </p>

        <label className={styles.fieldLabel} htmlFor="ask-question">
          Question
        </label>
        <TextBox
          id="ask-question"
          mode="input"
          value={question}
          onChange={setQuestion}
          rows={4}
          isDisabled={isAsking}
          placeholder="What does my library say about hybrid retrieval?"
          ariaLabel="Research question"
        />

        <fieldset className={styles.sources} disabled={isAsking}>
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
            Handwritten notes
          </label>
        </fieldset>

        <div className={styles.actions}>
          <button
            type="button"
            className={styles.primaryButton}
            onClick={() => void ask()}
            disabled={!canAsk}
          >
            {isAsking ? 'Searching…' : 'Ask'}
          </button>
        </div>
      </Box>

      {error && (
        <div className={styles.error} role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {isAsking && (
        <Box header="Answer">
          <div className={styles.loading}>
            <Spinner />
            <p>Retrieving evidence and writing an answer…</p>
          </div>
        </Box>
      )}

      {!isAsking && result && (
        <>
          <Box header="Answer">
            {(result.insufficient_evidence || result.suggest_external_search) && (
              <p className={styles.caveat} role="status">
                {result.suggest_external_search
                  ? `Your library cannot support a field-wide or latest-progress claim (${coverageLabel(result.library_coverage)}). Closest matches are listed as citations.`
                  : `Low evidence from your library (${coverageLabel(result.library_coverage)}). This answer is scoped to what the library supports.`}
              </p>
            )}
            <p className={styles.answer}>{result.answer}</p>
            <p className={styles.modelMeta}>
              {result.model}
              {` · ${coverageLabel(result.library_coverage)}`}
              {` · ${result.query_kind === 'field_wide' ? 'field-wide question' : 'library question'}`}
            </p>
            {result.suggest_external_search && (
              <button
                type="button"
                className={styles.secondaryButton}
                disabled
                title="Coming soon"
              >
                Search outside library (coming soon)
              </button>
            )}
          </Box>

          <Box header="Citations">
            {result.citations.length === 0 ? (
              <p className={styles.empty}>No matching chunks were retrieved.</p>
            ) : (
              <ol className={styles.citations}>
                {result.citations.map((citation) => (
                  <li key={citation.chunk_id} className={styles.citation}>
                    <div className={styles.citationHeader}>
                      <span className={styles.index}>[{citation.index}]</span>
                      <span
                        className={`${styles.badge} ${badgeClass(citation.source_type)}`}
                      >
                        {sourceLabel(citation.source_type)}
                      </span>
                      <span className={styles.citationTitle}>
                        {citation.title}
                      </span>
                    </div>
                    <p className={styles.citationMeta}>{citationMeta(citation)}</p>
                    <p className={styles.snippet}>{citation.snippet}</p>
                  </li>
                ))}
              </ol>
            )}
          </Box>
        </>
      )}
    </div>
  );
}
