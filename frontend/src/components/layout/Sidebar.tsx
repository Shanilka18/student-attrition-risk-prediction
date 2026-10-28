import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Trophy,
  Activity,
  Sliders,
  FileSpreadsheet,
  Database,
  Boxes,
  X,
} from 'lucide-react';
import { cn } from '@/utils/cn';

export interface NavItem {
  to: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

export const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/leaderboard', label: 'Model Leaderboard', icon: Trophy },
  { to: '/patterns', label: 'Engagement Patterns', icon: Activity },
  { to: '/drivers', label: 'What Drives Risk', icon: Sliders },
  { to: '/score', label: 'Score Students', icon: FileSpreadsheet },
  { to: '/train', label: 'Train on New Data', icon: Database },
];

export interface SidebarProps {
  isOpenMobile?: boolean;
  onCloseMobile?: () => void;
}

export function Sidebar({ isOpenMobile, onCloseMobile }: SidebarProps) {
  const isDev = import.meta.env.DEV;

  const content = (
    <div className="flex flex-col h-full bg-surface border-r border-line w-sidebar">
      {/* Product Name Header */}
      <div className="p-5 border-b border-line flex items-center justify-between">
        <div>
          <h1 className="font-serif text-[19px] leading-tight font-semibold text-ink tracking-tight">
            Engagement Index
          </h1>
          <p className="text-[11px] text-muted tracking-wider uppercase mt-0.5 font-medium">
            Student Intelligence
          </p>
        </div>
        {onCloseMobile && (
          <button
            onClick={onCloseMobile}
            className="lg:hidden p-1 rounded text-muted hover:text-ink hover:bg-paper"
            aria-label="Close menu"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 p-3 space-y-1 overflow-y-auto" aria-label="Main Navigation">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={onCloseMobile}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-3 py-2 text-table font-medium rounded-md transition-colors duration-150 select-none',
                  isActive
                    ? 'bg-forest-subtle text-forest font-semibold'
                    : 'text-muted hover:text-ink hover:bg-[#FAF8F5]'
                )
              }
            >
              <Icon className="w-4 h-4 shrink-0 stroke-[1.75]" />
              <span>{item.label}</span>
            </NavLink>
          );
        })}

        {/* Development Component Showcase Route */}
        {isDev && (
          <div className="pt-4 mt-4 border-t border-line">
            <div className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-wider text-muted">
              Development
            </div>
            <NavLink
              to="/_components"
              onClick={onCloseMobile}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-3 py-2 text-table font-medium rounded-md transition-colors duration-150 select-none',
                  isActive
                    ? 'bg-forest-subtle text-forest font-semibold'
                    : 'text-muted hover:text-ink hover:bg-[#FAF8F5]'
                )
              }
            >
              <Boxes className="w-4 h-4 shrink-0 stroke-[1.75]" />
              <span>Design System</span>
            </NavLink>
          </div>
        )}
      </nav>

      {/* Institutional Metadata Footer */}
      <div className="p-4 border-t border-line bg-[#FAF8F5] text-[11px] text-muted space-y-0.5">
        <div className="font-medium text-ink">Component KND_CDAP_IT_05</div>
        <div>Research analytics platform</div>
        <div className="tabular-nums text-[10px]">Version 1.0.0 (Offline)</div>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Persistent Sidebar */}
      <aside className="hidden lg:block shrink-0 sticky top-0 h-screen z-20">
        {content}
      </aside>

      {/* Mobile Drawer Overlay */}
      {isOpenMobile && (
        <div className="lg:hidden fixed inset-0 z-50 flex">
          <div
            className="fixed inset-0 bg-ink/40 backdrop-blur-[1px] animate-in fade-in"
            onClick={onCloseMobile}
            aria-hidden="true"
          />
          <div className="relative z-50 animate-in slide-in-from-left duration-150 h-full">
            {content}
          </div>
        </div>
      )}
    </>
  );
}
