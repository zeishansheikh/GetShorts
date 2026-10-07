import { Github } from 'lucide-react';

export const REPO_URL = 'https://github.com/zeishansheikh/GetShorts';

// Small "star us" ask, once per job, while the clips render
export default function StarBanner({ message = 'Enjoying GetShorts?' }) {
  return (
    <a
      href={REPO_URL}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center gap-2 px-3 py-2 rounded-input bg-paper3 border border-rule text-sm text-muted hover:text-ink transition-colors"
    >
      <Github size={14} className="shrink-0" />
      <span>{message} <span className="text-brass">Star us on GitHub ⭐</span></span>
    </a>
  );
}
