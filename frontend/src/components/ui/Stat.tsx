import React from 'react';
import { cn } from '@/utils/cn';

export interface StatProps extends React.HTMLAttributes<HTMLDivElement> {
  label: React.ReactNode;
  value: React.ReactNode;
  unit?: string;
  note?: React.ReactNode;
  sampleSize?: number;
  indicator?: 'neutral' | 'positive' | 'negative' | 'warning';
}

export function Stat({
  className,
  label,
  value,
  unit,
  note,
  sampleSize,
  indicator = 'neutral',
  ...props
}: StatProps) {
  const indicatorColors = {
    neutral: 'text-muted',
    positive: 'text-risk-low',
    negative: 'text-risk-high',
    warning: 'text-risk-moderate',
  };

  return (
    <div
      className={cn(
        'bg-surface border border-line rounded-md p-5 flex flex-col justify-between shadow-none',
        className
      )}
      {...props}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-caption font-medium text-muted uppercase tracking-wider">
          {label}
        </span>
        {sampleSize !== undefined && (
          <span className="text-caption text-muted tabular-nums">
            N = {sampleSize.toLocaleString()}
          </span>
        )}
      </div>

      <div className="mt-2.5 flex items-baseline gap-1.5">
        <span className="font-serif text-[26px] sm:text-[28px] lg:text-[28px] xl:text-[32px] leading-tight font-semibold text-ink tabular-nums tracking-tight break-words">
          {value}
        </span>
        {unit && (
          <span className="text-caption text-muted font-sans font-normal">
            {unit}
          </span>
        )}
      </div>

      {note && (
        <div className={cn('mt-2 text-caption flex items-center gap-1.5', indicatorColors[indicator])}>
          {note}
        </div>
      )}
    </div>
  );
}
