import React, { useState } from 'react';
import { ShieldAlert, X } from 'lucide-react';

interface EthicsBannerProps {
  disclaimer?: string;
  isSynthetic?: boolean;
}

const STORAGE_KEY_DISMISSED = 'pri_ethics_banner_dismissed';

export const EthicsBanner: React.FC<EthicsBannerProps> = ({
  disclaimer = "The Psychosocial Risk Index (PRI) is an educational decision-support indicator developed for SLIIT Kandy research — not a clinical psychiatric evaluation or medical diagnosis.",
  isSynthetic = true,
}) => {
  const [isDismissed, setIsDismissed] = useState<boolean>(() => {
    try {
      return sessionStorage.getItem(STORAGE_KEY_DISMISSED) === 'true';
    } catch {
      return false;
    }
  });

  const handleDismiss = () => {
    setIsDismissed(true);
    try {
      sessionStorage.setItem(STORAGE_KEY_DISMISSED, 'true');
    } catch (e) {
      console.warn('Could not save banner dismissed state:', e);
    }
  };

  if (isDismissed) {
    return null;
  }

  return (
    <div className="bg-amber-950/50 border-b border-amber-500/30 text-amber-200/90 px-4 py-1.5 text-xs flex items-center justify-between gap-3 shadow-inner no-print select-none transition-all">
      <div className="flex items-center gap-2 min-w-0">
        <ShieldAlert className="w-3.5 h-3.5 text-amber-400 shrink-0" />
        <span className="font-bold text-amber-300 text-[11px] uppercase tracking-wider shrink-0">
          Academic Decision-Support:
        </span>
        <span className="truncate text-[11px] text-amber-200/90 font-medium">
          {disclaimer}
        </span>
      </div>

      <div className="flex items-center gap-2.5 shrink-0">
        {isSynthetic && (
          <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/30 uppercase tracking-wider">
            Benchmark Dataset
          </span>
        )}
        <button
          onClick={handleDismiss}
          className="p-1 rounded text-amber-400/80 hover:text-amber-200 hover:bg-amber-500/20 transition-colors"
          title="Dismiss banner"
          aria-label="Dismiss banner"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
