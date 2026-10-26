import React from 'react';
import { Loader2, AlertCircle } from 'lucide-react';

export const LoadingState: React.FC<{ message?: string }> = ({ message = 'Loading live telemetry...' }) => (
  <div className="flex flex-col items-center justify-center min-h-[300px] p-8 text-center">
    <Loader2 className="w-8 h-8 text-indigo-400 animate-spin mb-3" />
    <p className="text-xs font-medium text-slate-400 tracking-wide">{message}</p>
  </div>
);

export const ErrorState: React.FC<{ message: string; onRetry?: () => void }> = ({
  message,
  onRetry,
}) => (
  <div className="flex flex-col items-center justify-center min-h-[300px] p-8 text-center bg-rose-950/10 border border-rose-500/20 rounded-xl my-4">
    <AlertCircle className="w-8 h-8 text-rose-400 mb-2" />
    <h4 className="text-sm font-semibold text-rose-300">Unable to load data</h4>
    <p className="text-xs text-rose-300/80 mt-1 max-w-md">{message}</p>
    {onRetry && (
      <button
        onClick={onRetry}
        className="mt-4 px-3 py-1.5 rounded-lg text-xs font-medium bg-rose-500/20 text-rose-200 border border-rose-500/30 hover:bg-rose-500/30 transition-colors"
      >
        Retry Request
      </button>
    )}
  </div>
);
