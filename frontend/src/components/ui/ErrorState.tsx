import { useState } from 'react';
import { AlertCircle, RefreshCw, ChevronDown, ChevronRight } from 'lucide-react';
import { Button } from './Button';
import { cn } from '@/utils/cn';

export interface ErrorStateProps {
  title?: string;
  message?: string;
  error?: Error | unknown;
  onRetry?: () => void;
  className?: string;
}

export function ErrorState({
  title = 'Information temporarily unavailable',
  message = 'We encountered an unexpected connection issue while retrieving these records. Your data is safe.',
  error,
  onRetry,
  className,
}: ErrorStateProps) {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const errorMessage = error instanceof Error ? error.message : typeof error === 'string' ? error : null;

  return (
    <div
      className={cn(
        'bg-surface border border-line rounded-md p-8 sm:p-10 flex flex-col items-center justify-center text-center',
        className
      )}
    >
      <div className="w-12 h-12 rounded-md bg-risk-high-subtle border border-risk-high/30 flex items-center justify-center text-risk-high mb-4">
        <AlertCircle className="w-6 h-6 stroke-1.5" />
      </div>

      <h3 className="font-serif text-section-title font-semibold text-ink mb-1.5">
        {title}
      </h3>
      <p className="text-body text-muted max-w-md mx-auto mb-6">
        {message}
      </p>

      {onRetry && (
        <Button
          variant="primary"
          onClick={onRetry}
          icon={<RefreshCw className="w-3.5 h-3.5" />}
          className="mb-4"
        >
          Try again
        </Button>
      )}

      {errorMessage && (
        <div className="mt-2 w-full max-w-md text-left">
          <button
            type="button"
            onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
            className="flex items-center gap-1.5 text-caption text-muted hover:text-ink font-medium select-none"
          >
            {showTechnicalDetails ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
            <span>Technical details</span>
          </button>

          {showTechnicalDetails && (
            <div className="mt-2 p-3 bg-paper border border-line rounded text-[12px] font-mono text-ink overflow-x-auto whitespace-pre-wrap break-all">
              {errorMessage}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
