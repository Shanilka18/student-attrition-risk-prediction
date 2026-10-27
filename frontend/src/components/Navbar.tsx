import React from 'react';
import {
  GraduationCap,
  ShieldAlert,
  Server,
  Sun,
  Moon,
  Tv,
  Layers,
  Database,
  Trophy,
  Sliders,
  GitCompare,
  Clock,
  Sparkles,
  UploadCloud,
  BookOpen,
  Download,
} from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import type { HealthResponse, MetaResponse } from '../api/client';

export type PageId =
  | 'overview'
  | 'data'
  | 'leaderboard'
  | 'model-detail'
  | 'compare'
  | 'early-warning'
  | 'explainability'
  | 'predict'
  | 'methodology'
  | 'export';

interface NavbarProps {
  currentPage: PageId;
  onPageChange: (page: PageId) => void;
  health: HealthResponse | null;
  meta: MetaResponse | null;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentPage,
  onPageChange,
  health,
  meta,
}) => {
  const { theme, toggleTheme, presentationMode, togglePresentationMode } = useTheme();

  const navItems: Array<{ id: PageId; label: string; icon: React.ReactNode }> = [
    { id: 'overview', label: 'Overview', icon: <Layers className="w-4 h-4" /> },
    { id: 'data', label: 'Data & Cohort', icon: <Database className="w-4 h-4" /> },
    { id: 'leaderboard', label: 'Leaderboard', icon: <Trophy className="w-4 h-4" /> },
    { id: 'model-detail', label: 'Model Detail', icon: <Sliders className="w-4 h-4" /> },
    { id: 'compare', label: 'Compare', icon: <GitCompare className="w-4 h-4" /> },
    { id: 'early-warning', label: 'Early Warning', icon: <Clock className="w-4 h-4" /> },
    { id: 'explainability', label: 'Explainability', icon: <Sparkles className="w-4 h-4" /> },
    { id: 'predict', label: 'PREDICT (Upload)', icon: <UploadCloud className="w-4 h-4" /> },
    { id: 'methodology', label: 'Methodology', icon: <BookOpen className="w-4 h-4" /> },
    { id: 'export', label: 'Export Center', icon: <Download className="w-4 h-4" /> },
  ];

  const isSynthetic = meta?.data_mode === 'SYNTHETIC' || true;

  return (
    <header className="sticky top-0 z-50 bg-slate-950/95 dark:bg-slate-950/95 backdrop-blur border-b border-slate-800">
      {/* 1. Persistent Synthetic Provenance Banner */}
      {isSynthetic && (
        <div className="bg-amber-950/50 border-b border-amber-600/40 px-4 py-1.5 flex items-center justify-between text-amber-200 text-xs">
          <div className="flex items-center gap-2 max-w-6xl mx-auto w-full">
            <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0" />
            <span className="font-semibold uppercase tracking-wider text-amber-300">
              [Models trained on SYNTHETIC data]:
            </span>
            <span className="truncate hidden sm:inline">
              Simulated Sri Lankan cohort for methodological validation pending institutional records clearance.
            </span>
            <span className="truncate sm:hidden">
              Synthetic cohort validation mode.
            </span>
          </div>
        </div>
      )}

      {/* 2. Top Title Bar & System Diagnostics */}
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
            <GraduationCap className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-base md:text-lg tracking-tight text-white">
                Academic Intelligence Framework
              </h1>
              <span className="hidden md:inline-block px-2 py-0.5 text-[10px] font-mono rounded bg-slate-800 text-slate-300 border border-slate-700">
                SLIIT CoEAI
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Student ID: <span className="font-mono text-slate-200 font-medium">IT23164680</span> | Project: <span className="font-mono text-slate-300">KND_CDAP_IT_05</span>
            </p>
          </div>
        </div>

        {/* Controls: Health, Presentation, Theme */}
        <div className="flex items-center gap-2">
          {/* Health Status Pill */}
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <Server className={`w-3.5 h-3.5 ${health?.status === 'healthy' ? 'text-emerald-400' : 'text-amber-400'}`} />
            <span className="text-slate-400">Backend:</span>
            <span className={`font-mono ${health?.status === 'healthy' ? 'text-emerald-400' : 'text-amber-400'}`}>
              {health?.status || 'Connecting...'}
            </span>
          </div>

          {/* Presentation Mode Toggle */}
          <button
            onClick={togglePresentationMode}
            title={presentationMode ? 'Exit Presentation Mode' : 'Enter Presentation Mode (Projector Optimization)'}
            className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              presentationMode
                ? 'bg-indigo-600 text-white border-indigo-500 shadow-sm shadow-indigo-500/50'
                : 'bg-slate-900 text-slate-300 border-slate-800 hover:bg-slate-800'
            }`}
          >
            <Tv className="w-3.5 h-3.5" />
            <span className="hidden md:inline">Presentation Mode</span>
          </button>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} theme`}
            className="p-1.5 rounded-lg bg-slate-900 text-slate-300 border border-slate-800 hover:bg-slate-800 transition-colors"
          >
            {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-300" /> : <Moon className="w-4 h-4 text-indigo-400" />}
          </button>
        </div>
      </div>

      {/* 3. Navigation Tabs Bar */}
      <nav className="max-w-7xl mx-auto px-4 flex gap-1 overflow-x-auto no-scrollbar border-t border-slate-800/80">
        {navItems.map(item => {
          const isActive = currentPage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onPageChange(item.id)}
              className={`flex items-center gap-2 px-3 py-2 text-xs md:text-sm font-medium whitespace-nowrap border-b-2 transition-all ${
                isActive
                  ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
              }`}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
    </header>
  );
};
