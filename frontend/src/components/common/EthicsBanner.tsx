import React from 'react';
import { AlertTriangle } from 'lucide-react';

interface EthicsBannerProps {
  disclaimer?: string;
  isSynthetic?: boolean;
}

export const EthicsBanner: React.FC<EthicsBannerProps> = ({
  disclaimer = "RESEARCH NOTICE: The Psychosocial Risk Index (PRI) is an academic decision-support indicator developed for educational research at SLIIT Kandy. It is not a clinical psychological evaluation or psychiatric diagnosis.",
  isSynthetic = true,
}) => {
  return (
    <div className="bg-amber-950/40 border-b border-amber-500/30 text-amber-200 px-4 py-2 text-xs flex flex-wrap items-center justify-between gap-2 shadow-inner no-print">
      <div className="flex items-center gap-2">
        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
        <span className="font-semibold text-amber-300 uppercase tracking-wider">Research Indicator Only:</span>
        <span className="text-amber-200/90">{disclaimer}</span>
      </div>
      {isSynthetic && (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 uppercase tracking-widest shrink-0">
          Synthetic Benchmark Dataset
        </span>
      )}
    </div>
  );
};
