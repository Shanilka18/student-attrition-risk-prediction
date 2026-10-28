import React, { forwardRef } from 'react';
import { cn } from '@/utils/cn';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'secondary',
      size = 'md',
      loading = false,
      disabled,
      icon,
      children,
      ...props
    },
    ref
  ) => {
    const baseStyles =
      'inline-flex items-center justify-center font-sans font-medium transition-colors duration-150 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forest focus-visible:ring-offset-2 disabled:cursor-not-allowed select-none';

    const variants = {
      primary:
        'bg-forest text-surface hover:bg-forest-hover active:bg-forest-active border border-forest disabled:bg-forest/50 disabled:border-forest/50',
      secondary:
        'bg-surface text-ink border border-line hover:bg-paper active:bg-line/40 disabled:bg-surface/50 disabled:text-muted disabled:border-line/60',
      ghost:
        'bg-transparent text-ink hover:bg-forest-subtle hover:text-forest active:bg-forest-subtle/80 disabled:text-muted',
      danger:
        'bg-surface text-risk-high border border-risk-high/40 hover:bg-risk-high-subtle active:border-risk-high disabled:border-line disabled:text-muted',
    };

    const sizes = {
      sm: 'h-8 px-3 text-[12.5px] leading-tight gap-1.5',
      md: 'h-9 px-4 text-[14px] leading-tight gap-2',
      lg: 'h-10 px-5 text-[15px] leading-normal gap-2.5',
    };

    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        aria-busy={loading}
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        {...props}
      >
        {loading ? (
          <span className="inline-block w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
        ) : (
          icon
        )}
        <span>{children}</span>
      </button>
    );
  }
);

Button.displayName = 'Button';
