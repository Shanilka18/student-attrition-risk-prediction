import { StatusTag } from './Tag';
import { cn } from '@/utils/cn';

export interface GapBarProps {
  gap: number;
  status: string;
  maxExpectedGap?: number;
  className?: string;
}

export function GapBar({
  gap,
  status,
  maxExpectedGap = 0.25,
  className,
}: GapBarProps) {
  const normStatus = status.toLowerCase();
  
  // Normalized percentage width bounded between 0% and 100%
  const widthPct = Math.min(100, Math.max(4, Math.round((Math.abs(gap) / maxExpectedGap) * 100)));
  
  // Tone according to generalization gap risk tokens
  let barColor = 'bg-risk-low';
  if (normStatus.includes('slight') || (gap > 0.05 && gap <= 0.10)) {
    barColor = 'bg-stone';
  } else if (normStatus.includes('overfit') || gap > 0.10) {
    barColor = 'bg-risk-high';
  }

  const formattedGap = gap >= 0 ? `+${gap.toFixed(3)}` : gap.toFixed(3);

  return (
    <div className={cn('flex items-center gap-2 min-w-[140px]', className)} data-testid="gap-bar">
      <div className="flex-1 bg-paper border border-line h-2 rounded-sm overflow-hidden" title={`Generalisation gap: ${formattedGap}`}>
        <div
          className={cn('h-full transition-all duration-300', barColor)}
          style={{ width: `${widthPct}%` }}
          role="progressbar"
          aria-valuenow={gap}
          aria-valuemin={0}
          aria-valuemax={maxExpectedGap}
          aria-label={`Generalisation gap ${formattedGap}`}
        />
      </div>
      <span className="text-caption font-mono tabular-nums text-muted shrink-0 w-12 text-right">
        {formattedGap}
      </span>
      <StatusTag status={status} />
    </div>
  );
}
