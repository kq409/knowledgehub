import styles from './Header.module.css';
import type { HeaderProps } from '../types';

export function Header(_props: HeaderProps) {
  return (
    <header className={styles.header}>
      <h1 className={styles.title}>ResearchPilot</h1>
      <p className={styles.subtitle}>
        Capture ideas, build a library, and ask your research
      </p>
    </header>
  );
}
