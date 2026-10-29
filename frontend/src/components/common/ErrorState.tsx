import React from 'react';
import { AlertCircle, RefreshCw, Inbox } from 'lucide-react';

interface ErrorStateProps {
  message?: string;
  onRetry?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  message = 'Failed to load endpoint data from FastAPI service.',
  onRetry,
}) => {
  return (
    <div className="bg-rose-50/50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900/50 rounded-xl p-8 text-center space-y-3">
      <AlertCircle className="w-8 h-8 text-rose-500 mx-auto" />
      <h4 className="text-sm font-semibold text-rose-900 dark:text-rose-200">
        Service Communication Error
      </h4>
      <p className="text-xs text-rose-700/80 dark:text-rose-300/70 max-w-md mx-auto">
        {message}
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-rose-700 dark:text-rose-300 bg-rose-100 dark:bg-rose-900/40 hover:bg-rose-200 dark:hover:bg-rose-900/60 rounded-md transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Retry Request
        </button>
      )}
    </div>
  );
};

export const EmptyState: React.FC<{ title?: string; message?: string }> = ({
  title = 'No Data Available',
  message = 'There are currently no records to display.',
}) => {
  return (
    <div className="bg-slate-50 dark:bg-slate-900/40 border border-dashed border-slate-300 dark:border-slate-800 rounded-xl p-8 text-center space-y-2">
      <Inbox className="w-8 h-8 text-slate-400 mx-auto" />
      <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300">{title}</h4>
      <p className="text-xs text-slate-500 dark:text-slate-500 max-w-sm mx-auto">{message}</p>
    </div>
  );
};
