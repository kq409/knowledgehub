import { useMemo } from 'react';
import styles from './ComparePapers.module.css';
import type {
  CitationSourceType,
  CompareCitation,
  CompareResponse,
  CompareSynthesis,
} from '../types';
import { dimensionLabel } from '../types';

function sourceLabel(sourceType: CitationSourceType): string {
  if (sourceType === 'paper') {
    return 'Paper';
  }
  if (sourceType === 'voice') {
    return 'Voice note';
  }
  if (sourceType === 'web') {
    return 'Web';
  }
  return 'Handwritten note';
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

function citationMeta(citation: CompareCitation): string {
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
  return parts.join(' · ');
}

export function CompareTable({ result }: { result: CompareResponse }) {
  return (
    <>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Dimension</th>
              {result.papers.map((paper) => (
                <th key={paper.paper_id}>
                  {paper.title}
                  {paper.year != null ? ` (${paper.year})` : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.dimensions.map((dim) => (
              <tr key={dim}>
                <th>{dimensionLabel(dim)}</th>
                {result.papers.map((paper) => (
                  <td key={`${paper.paper_id}-${dim}`}>
                    {paper.values[dim] || '—'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={styles.modelMeta}>
        {result.model} · {result.prompt_version}
      </p>
    </>
  );
}

export function CompareSynthesisView({
  synthesis,
}: {
  synthesis: CompareSynthesis;
}) {
  return (
    <>
      <h3 className={styles.synthTitle}>Agreements</h3>
      <p className={styles.answer}>{synthesis.agreements || '—'}</p>
      <h3 className={styles.synthTitle}>Disagreements</h3>
      <p className={styles.answer}>{synthesis.disagreements || '—'}</p>
      <h3 className={styles.synthTitle}>Research gap</h3>
      <p className={styles.answer}>{synthesis.research_gap || '—'}</p>
    </>
  );
}

export function CompareCitationList({ result }: { result: CompareResponse }) {
  const byPaper = useMemo(() => {
    const map = new Map<string, CompareCitation[]>();
    for (const citation of result.citations) {
      const list = map.get(citation.paper_id) ?? [];
      list.push(citation);
      map.set(citation.paper_id, list);
    }
    return map;
  }, [result]);

  if (result.citations.length === 0) {
    return <p className={styles.empty}>No matching chunks were retrieved.</p>;
  }

  return (
    <>
      {result.papers.map((paper) => {
        const group = byPaper.get(paper.paper_id) ?? [];
        if (group.length === 0) {
          return null;
        }
        return (
          <div key={paper.paper_id} className={styles.citationGroup}>
            <h3 className={styles.synthTitle}>{paper.title}</h3>
            <ol className={styles.citations}>
              {group.map((citation) => (
                <li key={citation.chunk_id} className={styles.citation}>
                  <div className={styles.citationHeader}>
                    <span className={styles.index}>[{citation.index}]</span>
                    <span
                      className={`${styles.badge} ${badgeClass(
                        citation.source_type
                      )}`}
                    >
                      {sourceLabel(citation.source_type)}
                    </span>
                    <span className={styles.citationTitle}>
                      {citation.title}
                    </span>
                  </div>
                  {citationMeta(citation) && (
                    <p className={styles.citationMeta}>
                      {citationMeta(citation)}
                    </p>
                  )}
                  <p className={styles.snippet}>{citation.snippet}</p>
                </li>
              ))}
            </ol>
          </div>
        );
      })}
    </>
  );
}
