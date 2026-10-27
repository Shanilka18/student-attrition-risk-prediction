import React from 'react';
import { Loader2, AlertTriangle, Inbox, RefreshCw } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = 'Loading research telemetry...',
}) => (
  <div className="flex flex-col items-center justify-center p-12 text-center">
    <Loader2 className="w-8 h-8 text-indigo-400 animate-spin mb-3" />
    <span className="text-sm font-medium text-slate-300">{message}</span>
    <span className="text-xs text-slate-500 mt-1">Fetching live models & audit records</span>
  </div>
);

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({ message, onRetry }) => (
  <div className="rounded-xl border border-rose-500/40 bg-rose-950/20 p-6 my-4 text-center">
    <AlertTriangle className="w-8 h-8 text-rose-400 mx-auto mb-2" />
    <h4 className="text-sm font-semibold text-rose-300">Service Communication Warning</h4>
    <p className="text-xs text-slate-300 mt-1 max-w-md mx-auto">{message}</p>
    {onRetry && (
      <button
        onClick={onRetry}
        className="mt-4 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-rose-600 hover:bg-rose-500 text-white transition-colors"
      >
        <RefreshCw className="w-3.5 h-3.5" />
        Retry Request
      </button>
    )}
  </div>
);

interface EmptyStateProps {
  title: string;
  description: string;
  action?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ title, description, action }) => (
  <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-10 text-center">
    <Inbox className="w-10 h-10 text-slate-500 mx-auto mb-2" />
    <h4 className="text-sm font-semibold text-slate-200">{title}</h4>
    <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">{description}</p>
    {action && <div className="mt-4">{action}</div>}
  </div>
);
