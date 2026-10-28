import React from 'react';
import { cn } from '@/utils/cn';

export interface CardProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'title'> {
  as?: React.ElementType;
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
}

export function Card({
  as: Component = 'div',
  title,
  subtitle,
  action,
  className,
  children,
  ...props
}: CardProps) {
  if (title) {
    return (
      <Component
        className={cn(
          'bg-surface border border-line rounded-md shadow-none overflow-hidden',
          className
        )}
        {...props}
      >
        <CardHeader className="flex items-center justify-between gap-4">
          <div>
            <CardTitle>{title}</CardTitle>
            {subtitle && <CardDescription>{subtitle}</CardDescription>}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </CardHeader>
        <CardContent>{children}</CardContent>
      </Component>
    );
  }

  return (
    <Component
      className={cn(
        'bg-surface border border-line rounded-md shadow-none overflow-hidden',
        className
      )}
      {...props}
    >
      {children}
    </Component>
  );
}

export function CardHeader({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        'px-5 py-4 border-b border-line flex items-center justify-between gap-4',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardTitle({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn('font-serif text-section-title text-ink font-semibold tracking-tight', className)}
      {...props}
    >
      {children}
    </h3>
  );
}

export function CardDescription({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p className={cn('text-caption text-muted mt-0.5', className)} {...props}>
      {children}
    </p>
  );
}

export function CardContent({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('p-5', className)} {...props}>
      {children}
    </div>
  );
}

export function CardFooter({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        'px-5 py-3 border-t border-line bg-[#FAFAF8] flex items-center justify-between text-caption text-muted',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}
