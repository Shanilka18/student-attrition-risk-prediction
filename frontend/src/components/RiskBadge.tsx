import React from 'react';

interface RiskBadgeProps {
  band: 'Low' | 'Moderate' | 'High' | 'Critical' | string;
  score?: number;
  showScore?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({
  band,
  score,
  showScore = false,
  size = 'md'
}) => {
  const normalized = (band || 'Low').toLowerCase();

  let colorClasses = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
  let dotColor = 'bg-emerald-400';

  if (normalized === 'critical') {
    colorClasses = 'bg-rose-500/15 text-rose-400 border-rose-500/40 shadow-sm shadow-rose-950/50';
    dotColor = 'bg-rose-400 animate-pulse';
  } else if (normalized === 'high') {
    colorClasses = 'bg-amber-500/15 text-amber-400 border-amber-500/40';
    dotColor = 'bg-amber-400';
  } else if (normalized === 'moderate') {
    colorClasses = 'bg-sky-500/15 text-sky-400 border-sky-500/30';
    dotColor = 'bg-sky-400';
  }

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-xs px-2.5 py-1',
    lg: 'text-sm px-3.5 py-1.5 font-semibold',
  }[size];

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-medium rounded-full border ${colorClasses} ${sizeClasses}`}
      title={`Clinical Risk Tier: ${band} (${score !== undefined ? score.toFixed(1) : ''}/100)`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
      <span>{band}</span>
      {showScore && score !== undefined && (
        <span className="font-mono opacity-80 border-l border-current/20 pl-1.5 ml-0.5">
          {score.toFixed(1)}
        </span>
      )}
    </span>
  );
};
