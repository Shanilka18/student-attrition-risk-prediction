import React from 'react';
import { HelpCircle } from 'lucide-react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  delta?: string;
  deltaType?: 'positive' | 'negative' | 'neutral';
  tooltip?: string;
  icon?: React.ReactNode;
  highlight?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subValue,
  delta,
  deltaType = 'neutral',
  tooltip,
  icon,
  highlight = false,
}) => {
  return (
    <div
      className={`rounded-xl p-5 border transition-all ${
        highlight
          ? 'bg-indigo-950/20 border-indigo-500/40 shadow-sm shadow-indigo-950/30'
          : 'bg-slate-900/60 dark:bg-slate-900/60 border-slate-800'
      }`}
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="text-xs font-medium text-slate-400 flex items-center gap-1.5 uppercase tracking-wider">
          {label}
          {tooltip && (
            <span title={tooltip} className="cursor-help text-slate-500 hover:text-slate-300">
              <HelpCircle className="w-3.5 h-3.5" />
            </span>
          )}
        </span>
        {icon && <span className="text-slate-400">{icon}</span>}
      </div>

      <div className="flex items-baseline gap-2">
        <div className="text-2xl md:text-3xl font-bold font-mono tracking-tight text-slate-100">
          {value}
        </div>
        {delta && (
          <span
            className={`text-xs font-semibold px-1.5 py-0.5 rounded ${
              deltaType === 'positive'
                ? 'bg-emerald-500/15 text-emerald-400'
                : deltaType === 'negative'
                ? 'bg-rose-500/15 text-rose-400'
                : 'bg-slate-800 text-slate-300'
            }`}
          >
            {delta}
          </span>
        )}
      </div>

      {subValue && (
        <div className="text-xs text-slate-400 mt-1.5 font-sans leading-normal">
          {subValue}
        </div>
      )}
    </div>
  );
};
