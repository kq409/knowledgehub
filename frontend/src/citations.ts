import type { ChatCitation } from './types';

const CITATION_TOKEN = /\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g;

function citationOrder(answer: string): number[] {
  const seen = new Set<number>();
  const order: number[] = [];
  for (const match of answer.matchAll(CITATION_TOKEN)) {
    const group = match[1];
    if (!group) {
      continue;
    }
    for (const part of group.split(',')) {
      const number = Number(part.trim());
      if (!seen.has(number)) {
        seen.add(number);
        order.push(number);
      }
    }
  }
  return order;
}

export function compactAnswerCitations(
  answer: string,
  citations: ChatCitation[]
): { answer: string; citations: ChatCitation[] } {
  const byIndex = new Map<number, ChatCitation>();
  for (const item of citations) {
    byIndex.set(item.index, item);
  }

  const used = citationOrder(answer).filter((number) => byIndex.has(number));
  if (used.length === 0) {
    return { answer, citations: [] };
  }

  const mapping = new Map<number, number>();
  used.forEach((old, index) => {
    mapping.set(old, index + 1);
  });

  const rewritten = answer.replace(CITATION_TOKEN, (token, raw: string) => {
    const numbers = raw.split(',').map((part) => Number(part.trim()));
    if (!numbers.some((number) => mapping.has(number))) {
      return token;
    }
    const remapped = numbers
      .map((number) => String(mapping.get(number) ?? number))
      .join(', ');
    return `[${remapped}]`;
  });

  return {
    answer: rewritten,
    citations: used.flatMap((old, index) => {
      const item = byIndex.get(old);
      return item ? [{ ...item, index: index + 1 }] : [];
    }),
  };
}
