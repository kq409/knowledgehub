import type { ComponentProps } from 'react';
import Markdown from 'react-markdown';
import type { Components, ExtraProps } from 'react-markdown';
import remarkBreaks from 'remark-breaks';
import styles from './MarkdownAnswer.module.css';

function MarkdownLink({
  node: _node,
  children,
  ...props
}: ComponentProps<'a'> & ExtraProps) {
  return (
    <a {...props} target="_blank" rel="noreferrer">
      {children}
    </a>
  );
}

const markdownComponents: Components = {
  a: MarkdownLink,
};

export function MarkdownAnswer({ text }: { text: string }) {
  return (
    <div className={styles.body}>
      <Markdown
        remarkPlugins={[remarkBreaks]}
        components={markdownComponents}
        disallowedElements={['img']}
      >
        {text}
      </Markdown>
    </div>
  );
}
