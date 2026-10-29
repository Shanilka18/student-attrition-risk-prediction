import React from 'react';
import {
  Sun,
  Moon,
  Activity,
  Layers,
  BarChart3,
  Trophy,
  ShieldAlert,
  BrainCircuit,
  GitBranch,
  UploadCloud,
} from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';

export type PageId =
  | 'overview'
  | 'data-cleaning'
  | 'model-zoo'
  | 'leaderboard'
  | 'final-model'
  | 'shap'
  | 'causal'
  | 'upload-predict';

interface HeaderProps {
  activePage: PageId;
  onPageChange: (page: PageId) => void;
  isBackendOnline: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activePage,
  onPageChange,
  isBackendOnline,
}) => {
  const { theme, toggleTheme } = useTheme();

  const navItems: { id: PageId; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <Layers className="w-4 h-4" /> },
    { id: 'data-cleaning', label: 'Data & Cleaning', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'model-zoo', label: 'Model Zoo', icon: <Activity className="w-4 h-4" /> },
    { id: 'leaderboard', label: 'Leaderboard', icon: <Trophy className="w-4 h-4 text-amber-500" /> },
    { id: 'final-model', label: 'ERI Model', icon: <ShieldAlert className="w-4 h-4 text-blue-500" /> },
    { id: 'shap', label: 'SHAP (XAI)', icon: <BrainCircuit className="w-4 h-4 text-purple-500" /> },
    { id: 'causal', label: 'Causal Inference', icon: <GitBranch className="w-4 h-4 text-emerald-500" /> },
    { id: 'upload-predict', label: 'Upload & Score', icon: <UploadCloud className="w-4 h-4 text-cyan-500" /> },
  ];

  return (
    <header className="sticky top-0 z-40 bg-white/95 dark:bg-slate-900/95 backdrop-blur border-b border-slate-200 dark:border-slate-800 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand & Project ID */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-900 via-blue-800 to-indigo-950 text-white flex items-center justify-center font-bold text-lg shadow-md border border-blue-700/50">
              ERI
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-sm sm:text-base text-slate-900 dark:text-white tracking-tight">
                  Environmental & Financial Risk Intelligence
                </span>
                <span className="text-[10px] font-semibold uppercase tracking-wider bg-blue-100 dark:bg-blue-950 text-blue-800 dark:text-blue-300 px-2 py-0.5 rounded-full border border-blue-200 dark:border-blue-800">
                  KND-CDAP-IT-05
                </span>
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400">
                SLIIT CDAP Attrition Prevention Research Platform
              </p>
            </div>
          </div>

          {/* Right Controls: Backend Status & Theme Toggle */}
          <div className="flex items-center gap-3">
            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
                isBackendOnline
                  ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-300'
                  : 'bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-800 text-rose-700 dark:text-rose-300'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isBackendOnline ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
                }`}
              />
              <span className="hidden sm:inline">
                {isBackendOnline ? 'FastAPI 8000 Live' : 'Backend Offline'}
              </span>
            </div>

            <button
              onClick={toggleTheme}
              aria-label="Toggle Theme"
              className="p-2 text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
            >
              {theme === 'dark' ? (
                <Sun className="w-4 h-4 text-amber-400" />
              ) : (
                <Moon className="w-4 h-4 text-slate-700" />
              )}
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 overflow-x-auto no-scrollbar py-1 border-t border-slate-100 dark:border-slate-800/80">
          {navItems.map((item) => {
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onPageChange(item.id)}
                className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
