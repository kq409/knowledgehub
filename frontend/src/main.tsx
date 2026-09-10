import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Root element not found');
}

const originalFetch = window.fetch.bind(window);
const DEMO_USER_KEY = 'kh-demo-user';

window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
  const url =
    typeof input === 'string'
      ? input
      : input instanceof URL
        ? input.href
        : input.url;
  const headers = new Headers(init?.headers);
  if (
    !headers.has('X-User-Id') &&
    (url.startsWith('/') || url.includes('/api/'))
  ) {
    headers.set(
      'X-User-Id',
      window.localStorage.getItem(DEMO_USER_KEY) || 'alice'
    );
  }
  return originalFetch(input, { ...init, headers });
};

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
