import React from 'react';
import { cn } from '@/utils/cn';

export interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  width?: string | number;
  height?: string | number;
}

export function Skeleton({ className, style, width, height, ...props }: SkeletonProps) {
  return (
    <div
      className={cn('animate-pulse rounded bg-[#EAE6DD]', className)}
      style={{
        width,
        height,
        ...style,
      }}
      aria-hidden="true"
      {...props}
    />
  );
}

export function StatSkeleton() {
  return (
    <div className="bg-surface border border-line rounded-md p-5 flex flex-col justify-between shadow-none space-y-4">
      <Skeleton className="h-4 w-28" />
      <Skeleton className="h-9 w-20" />
      <Skeleton className="h-3.5 w-36" />
    </div>
  );
}

export function TableSkeleton({ rows = 5, cols = 4 }: { rows?: number; cols?: number }) {
  return (
    <div className="border border-line rounded-md bg-surface overflow-hidden">
      <div className="h-10 bg-[#FAF8F5] border-b border-line flex items-center px-4 gap-4">
        {Array.from({ length: cols }).map((_, i) => (
          <Skeleton key={i} className="h-4 flex-1" />
        ))}
      </div>
      <div className="divide-y divide-line p-2 space-y-2">
        {Array.from({ length: rows }).map((_, r) => (
          <div key={r} className="flex items-center px-2 py-3 gap-4">
            {Array.from({ length: cols }).map((_, c) => (
              <Skeleton key={c} className="h-4 flex-1" />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
