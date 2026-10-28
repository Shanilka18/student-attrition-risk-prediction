import React, { forwardRef } from 'react';
import { cn } from '@/utils/cn';

export interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  'aria-label': string;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ className, variant = 'ghost', size = 'md', children, ...props }, ref) => {
    const baseStyles =
      'inline-flex items-center justify-center font-sans transition-colors duration-150 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forest focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 select-none';

    const variants = {
      primary: 'bg-forest text-surface hover:bg-forest-hover active:bg-forest-active',
      secondary: 'bg-surface text-ink border border-line hover:bg-paper active:bg-line/40',
      ghost: 'bg-transparent text-muted hover:text-ink hover:bg-paper active:bg-line/40',
    };

    const sizes = {
      sm: 'w-7 h-7 text-sm',
      md: 'w-8 h-8 text-base',
      lg: 'w-10 h-10 text-lg',
    };

    return (
      <button
        ref={ref}
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        {...props}
      >
        {children}
      </button>
    );
  }
);

IconButton.displayName = 'IconButton';
