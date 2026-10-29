import React from 'react';
import {
  LayoutDashboard,
  Database,
  Trophy,
  Award,
  Network,
  UploadCloud,
  UserCheck,
  BookOpen,
  SlidersHorizontal,
  ChevronRight,
} from 'lucide-react';
import { ResearchPrototypeBadge } from '../common/Badges';

export type PageId =
  | 'overview'
  | 'reliability'
  | 'leaderboard'
  | 'champion'
  | 'explainability'
  | 'score'
  | 'student'
  | 'methodology';

interface SidebarProps {
  currentPage: PageId;
  onSelectPage: (page: PageId) => void;
  presentationMode: boolean;
  onTogglePresentation: () => void;
}

const navItems: Array<{ id: PageId; label: string; icon: React.FC<{ className?: string }>; badge?: string }> = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'reliability', label: 'Data & Reliability', icon: Database },
  { id: 'leaderboard', label: 'Leaderboard', icon: Trophy },
  { id: 'champion', label: 'Champion & Evaluation', icon: Award },
  { id: 'explainability', label: 'Explainability (SHAP)', icon: Network },
  { id: 'score', label: 'Score My Data', icon: UploadCloud },
  { id: 'student', label: 'Individual Student View', icon: UserCheck },
  { id: 'methodology', label: 'Methodology & Docs', icon: BookOpen },
];

export const Sidebar: React.FC<SidebarProps> = ({
  currentPage,
  onSelectPage,
  presentationMode,
  onTogglePresentation,
}) => {
  return (
    <aside className="w-64 bg-navy-900 border-r border-navy-700/80 flex flex-col shrink-0 no-print select-none">
      {/* Brand Header */}
      <div className="p-5 border-b border-navy-700/80">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20 text-white font-black text-lg tracking-wider">
            Ψ
          </div>
          <div>
            <h1 className="font-bold text-sm tracking-tight text-white leading-tight">
              Attrition Intelligence
            </h1>
            <p className="text-[11px] text-slate-400 font-medium">Psychosocial Risk Index (PRI)</p>
          </div>
        </div>
        <div className="mt-3 flex items-center justify-between">
          <ResearchPrototypeBadge />
          <span className="text-[10px] text-slate-400 font-mono">v1.0.0</span>
        </div>
      </div>

      {/* Nav List */}
      <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
        <div className="px-3 py-2 text-[10px] font-semibold tracking-wider text-slate-400 uppercase">
          Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectPage(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                isActive
                  ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40 shadow-sm'
                  : 'text-slate-300 hover:text-white hover:bg-navy-800/60'
              }`}
            >
              <div className="flex items-center gap-2.5">
                <Icon className={`w-4 h-4 ${isActive ? 'text-indigo-400' : 'text-slate-400'}`} />
                <span>{item.label}</span>
              </div>
              <div className="flex items-center gap-1.5">
                {item.badge && (
                  <span className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    {item.badge}
                  </span>
                )}
                {isActive && <ChevronRight className="w-3.5 h-3.5 text-indigo-400" />}
              </div>
            </button>
          );
        })}
      </nav>

      {/* Bottom Controls */}
      <div className="p-3 border-t border-navy-700/80 bg-navy-950/40 space-y-2">
        <button
          onClick={onTogglePresentation}
          className={`w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium border transition-colors ${
            presentationMode
              ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
              : 'bg-navy-800 text-slate-300 border-navy-700 hover:bg-navy-750'
          }`}
          title="Toggles high-contrast presentation mode with larger typography and clean slide layouts"
        >
          <SlidersHorizontal className="w-3.5 h-3.5" />
          <span>{presentationMode ? 'Exit Presentation Mode' : 'Presentation Mode'}</span>
        </button>

        <div className="px-2 py-1 text-[10px] text-slate-400 text-center">
          SLIIT Kandy Research • CDAP IT05
        </div>
      </div>
    </aside>
  );
};
