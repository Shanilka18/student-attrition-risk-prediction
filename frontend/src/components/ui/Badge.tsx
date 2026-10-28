import React from 'react';
import { cn } from '@/utils/cn';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'forest' | 'amber' | 'crimson' | 'neutral' | string;
  size?: 'sm' | 'md';
}

export function Badge({
  className,
  variant = 'neutral',
  size = 'md',
  children,
  ...props
}: BadgeProps) {
  const styles: Record<string, string> = {
    forest: 'bg-forest-subtle text-forest border border-forest/30',
    amber: 'bg-amber-subtle text-amber-text border border-amber/30',
    crimson: 'bg-crimson-subtle text-crimson border border-crimson/30',
    neutral: 'bg-paper text-muted border border-border',
  };

  const activeStyle = styles[variant] || styles.neutral;
  const sizeStyle = size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-caption';

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 font-medium rounded font-sans',
        sizeStyle,
        activeStyle,
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}
