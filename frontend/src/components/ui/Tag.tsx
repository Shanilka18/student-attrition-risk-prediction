import React from 'react';
import { cn } from '@/utils/cn';

export type TagVariant =
  | 'good-fit'
  | 'slight-overfit'
  | 'overfit'
  | 'risk-low'
  | 'risk-moderate'
  | 'risk-high'
  | 'consistent'
  | 'declining'
  | 'recovering'
  | 'variable'
  | 'neutral'
  | 'primary';

export interface TagProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: TagVariant;
  size?: 'sm' | 'md';
  icon?: React.ReactNode;
}

export function Tag({
  className,
  variant = 'neutral',
  size = 'md',
  icon,
  children,
  ...props
}: TagProps) {
  const variantStyles: Record<TagVariant, { bg: string; dot: string }> = {
    // Overfit status
    'good-fit': {
      bg: 'bg-risk-low-subtle text-risk-low border-risk-low/30',
      dot: 'bg-risk-low',
    },
    'slight-overfit': {
      bg: 'bg-risk-moderate-subtle text-[#7A5400] border-risk-moderate/30',
      dot: 'bg-risk-moderate',
    },
    overfit: {
      bg: 'bg-risk-high-subtle text-risk-high border-risk-high/30',
      dot: 'bg-risk-high',
    },

    // Risk bands (Low / Moderate / High)
    'risk-low': {
      bg: 'bg-risk-low-subtle text-risk-low border-risk-low/30',
      dot: 'bg-risk-low',
    },
    'risk-moderate': {
      bg: 'bg-risk-moderate-subtle text-[#7A5400] border-risk-moderate/30',
      dot: 'bg-risk-moderate',
    },
    'risk-high': {
      bg: 'bg-risk-high-subtle text-risk-high border-risk-high/30',
      dot: 'bg-risk-high',
    },

    // Longitudinal patterns
    consistent: {
      bg: 'bg-forest-subtle text-forest border-forest/30',
      dot: 'bg-forest',
    },
    declining: {
      bg: 'bg-clay-subtle text-clay border-clay/30',
      dot: 'bg-clay',
    },
    recovering: {
      bg: 'bg-info-subtle text-info border-info/30',
      dot: 'bg-info',
    },
    variable: {
      bg: 'bg-stone-subtle text-stone border-stone/30',
      dot: 'bg-stone',
    },

    // General purpose
    neutral: {
      bg: 'bg-paper text-ink border-line',
      dot: 'bg-muted',
    },
    primary: {
      bg: 'bg-forest-subtle text-forest border-forest/30',
      dot: 'bg-forest',
    },
  };

  const sizes = {
    sm: 'text-[11px] leading-tight px-1.5 py-0.5 gap-1',
    md: 'text-[12.5px] leading-tight px-2 py-0.5 gap-1.5',
  };

  const style = variantStyles[variant] || variantStyles.neutral;

  return (
    <span
      className={cn(
        'inline-flex items-center font-sans font-medium border rounded select-none',
        style.bg,
        sizes[size],
        className
      )}
      {...props}
    >
      {icon ? (
        <span className="shrink-0">{icon}</span>
      ) : (
        <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', style.dot)} aria-hidden="true" />
      )}
      <span>{children}</span>
    </span>
  );
}

/**
 * Convenience helper to render the right tag from a string status value.
 */
export function StatusTag({ status }: { status: string }) {
  const norm = status.toLowerCase().trim();
  if (norm.includes('good')) {
    return <Tag variant="good-fit">Good fit</Tag>;
  }
  if (norm.includes('slight')) {
    return <Tag variant="slight-overfit">Slight overfit</Tag>;
  }
  if (norm.includes('overfit')) {
    return <Tag variant="overfit">Overfit</Tag>;
  }
  if (norm === 'low') {
    return <Tag variant="risk-low">Low risk</Tag>;
  }
  if (norm === 'moderate') {
    return <Tag variant="risk-moderate">Moderate risk</Tag>;
  }
  if (norm === 'high') {
    return <Tag variant="risk-high">High risk</Tag>;
  }
  if (norm === 'consistent') {
    return <Tag variant="consistent">Consistent</Tag>;
  }
  if (norm === 'declining') {
    return <Tag variant="declining">Declining</Tag>;
  }
  if (norm === 'recovering') {
    return <Tag variant="recovering">Recovering</Tag>;
  }
  if (norm === 'variable' || norm === 'sporadic') {
    return <Tag variant="variable">Variable</Tag>;
  }
  return <Tag variant="neutral">{status}</Tag>;
}
