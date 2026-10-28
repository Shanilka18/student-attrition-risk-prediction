import React from 'react';
import { getGlossaryEntry, type GlossaryEntry } from '@/content/glossary';
import { Tooltip } from './Tooltip';
import { cn } from '@/utils/cn';

export interface TermProps extends React.HTMLAttributes<HTMLSpanElement> {
  termKey: string;
  customLabel?: React.ReactNode;
  showFormalFirst?: boolean;
  open?: boolean;
}

export function Term({
  termKey,
  customLabel,
  showFormalFirst = false,
  open,
  className,
  ...props
}: TermProps) {
  const entry: GlossaryEntry = getGlossaryEntry(termKey);
  const displayText = customLabel || (showFormalFirst ? entry.formalName : entry.plainLabel);

  const tooltipContent = (
    <div className="space-y-1 text-left">
      <div className="font-serif font-semibold text-surface">
        {entry.formalName}
      </div>
      <div className="text-[12px] leading-relaxed text-[#DDD8CE]">
        {entry.summary}
      </div>
      {entry.unit && (
        <div className="text-[11px] text-line pt-0.5">
          Measured in: <span className="tabular-nums font-mono">{entry.unit}</span>
        </div>
      )}
    </div>
  );

  return (
    <Tooltip content={tooltipContent} open={open}>
      <span
        tabIndex={0}
        role="button"
        aria-label={`${entry.plainLabel}: ${entry.formalName}`}
        className={cn(
          'inline-flex items-baseline border-b border-dotted border-muted cursor-help text-ink hover:text-forest transition-colors duration-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-forest',
          className
        )}
        {...props}
      >
        {displayText}
      </span>
    </Tooltip>
  );
}
