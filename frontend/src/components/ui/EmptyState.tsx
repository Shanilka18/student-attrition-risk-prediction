import React from 'react';
import { Inbox } from 'lucide-react';
import { cn } from '@/utils/cn';

export interface EmptyStateProps {
  title: string;
  description: string;
  icon?: React.ReactNode;
  action?: React.ReactNode;
  secondaryAction?: React.ReactNode;
  className?: string;
  bordered?: boolean;
}

export function EmptyState({
  title,
  description,
  icon,
  action,
  secondaryAction,
  className,
  bordered = true,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center text-center p-8 sm:p-12',
        bordered && 'bg-surface border border-line rounded-md',
        className
      )}
    >
      <div className="w-12 h-12 rounded-md bg-paper border border-line flex items-center justify-center text-muted mb-4">
        {icon || <Inbox className="w-6 h-6 stroke-1" />}
      </div>
      <h3 className="font-serif text-section-title font-semibold text-ink mb-1.5">
        {title}
      </h3>
      <p className="text-body text-muted max-w-md mx-auto mb-6">
        {description}
      </p>
      {(action || secondaryAction) && (
        <div className="flex flex-wrap items-center justify-center gap-3">
          {action}
          {secondaryAction}
        </div>
      )}
    </div>
  );
}
