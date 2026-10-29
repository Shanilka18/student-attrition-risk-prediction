import React from 'react';
import { AlertTriangle } from 'lucide-react';

interface SyntheticBannerProps {
  bannerText?: string;
}

export const SyntheticBanner: React.FC<SyntheticBannerProps> = ({
  bannerText = 'SYNTHETIC DATA - pipeline validation only',
}) => {
  return (
    <div className="bg-amber-500/15 border-b border-amber-500/30 text-amber-900 dark:text-amber-200 px-4 py-2 text-xs font-semibold tracking-wide flex items-center justify-center gap-2 shadow-inner">
      <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 animate-pulse flex-shrink-0" />
      <span>{bannerText.toUpperCase()}</span>
      <span className="hidden md:inline font-normal text-amber-800/80 dark:text-amber-300/70">
        — Demonstrates technical pipeline integrity; not an empirical real-world claim.
      </span>
    </div>
  );
};
