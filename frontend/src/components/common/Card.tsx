import React from 'react';
import { HelpCircle, Download } from 'lucide-react';

interface CardProps {
  title?: React.ReactNode;
  subtitle?: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
  aboutText?: string;
  onExportPng?: () => void;
}

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  badge,
  actions,
  className = '',
  children,
  aboutText,
  onExportPng,
}) => {
  return (
    <div
      className={`bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 rounded-xl shadow-sm hover:shadow transition-shadow overflow-hidden ${className}`}
    >
      {(title || subtitle || badge || actions || aboutText || onExportPng) && (
        <div className="px-5 py-4 border-b border-slate-100 dark:border-slate-800/80 flex items-center justify-between gap-3">
          <div className="space-y-0.5">
            <div className="flex items-center gap-2">
              {typeof title === 'string' ? (
                <h3 className="text-base font-semibold text-slate-800 dark:text-slate-100">
                  {title}
                </h3>
              ) : (
                title
              )}
              {badge}
              {aboutText && (
                <div className="relative group inline-block">
                  <HelpCircle className="w-4 h-4 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 cursor-pointer" />
                  <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 hidden group-hover:block z-50 w-64 p-2 text-xs bg-slate-900 dark:bg-slate-800 text-slate-100 rounded-lg shadow-lg border border-slate-700 pointer-events-none">
                    {aboutText}
                  </div>
                </div>
              )}
            </div>
            {subtitle && (
              <p className="text-xs text-slate-500 dark:text-slate-400">{subtitle}</p>
            )}
          </div>
          <div className="flex items-center gap-2">
            {onExportPng && (
              <button
                onClick={onExportPng}
                title="Export Chart as PNG"
                className="p-1.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              >
                <Download className="w-4 h-4" />
              </button>
            )}
            {actions}
          </div>
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
};
