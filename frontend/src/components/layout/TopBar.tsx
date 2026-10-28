import { Menu, ShieldCheck } from 'lucide-react';
import { useStatus } from '@/api/queries';
import { Tooltip } from '@/components/ui/Tooltip';
import { IconButton } from '@/components/ui/IconButton';

export interface TopBarProps {
  onOpenMobileMenu: () => void;
}

export function TopBar({ onOpenMobileMenu }: TopBarProps) {
  const { data: status } = useStatus();

  const championName = status?.champion?.display_name || 'k-Nearest Neighbours';
  const optimalThreshold = status?.champion?.optimal_threshold ?? 0.40;
  const valF1 = status?.champion?.cv_f1;
  
  const isRealData = status?.data_source === 'survey';
  const dataSourceLabel = isRealData
    ? 'Training data: Your survey data'
    : 'Training data: Sample survey responses (generated)';

  return (
    <header className="h-14 border-b border-line bg-surface sticky top-0 z-10 px-4 sm:px-6 flex items-center justify-between gap-4">
      {/* Left: Mobile Menu Toggle & System Identity */}
      <div className="flex items-center gap-3">
        <IconButton
          variant="ghost"
          size="sm"
          aria-label="Open navigation menu"
          onClick={onOpenMobileMenu}
          className="lg:hidden"
        >
          <Menu className="w-5 h-5 text-ink" />
        </IconButton>

        <div className="hidden sm:flex items-center gap-2 text-caption text-muted">
          <span className="font-serif font-semibold text-ink">University Attrition Analytics</span>
          <span className="text-line">•</span>
          <span>Semester Evaluation</span>
        </div>
      </div>

      {/* Right: Active Model Badge & Quiet Data Source Label */}
      <div className="flex items-center gap-3">
        {/* Active Model Indicator */}
        <Tooltip
          content={
            <div className="text-left space-y-1">
              <div className="font-serif font-semibold text-surface">Active Champion Model</div>
              <div className="text-[12px] text-[#DDD8CE]">
                Selected via 15-fold cross-validation with overfitting penalty.
              </div>
              <div className="text-[11px] text-line pt-1">
                Threshold: <span className="tabular-nums font-mono">{optimalThreshold.toFixed(2)}</span>
                {valF1 !== undefined && (
                  <> • Val F1: <span className="tabular-nums font-mono">{(valF1 * 100).toFixed(1)}%</span></>
                )}
              </div>
            </div>
          }
        >
          <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 bg-paper border border-line rounded text-caption text-ink cursor-help select-none">
            <ShieldCheck className="w-3.5 h-3.5 text-forest shrink-0" />
            <span className="text-muted">Model:</span>
            <span className="font-medium text-ink truncate max-w-[160px]">{championName}</span>
          </div>
        </Tooltip>

        {/* Quiet Data-Source Label (Rule 10) */}
        <div
          className="text-caption text-muted bg-[#FAF8F5] border border-line px-2.5 py-1 rounded text-xs select-none"
          title="Data origin for current model weights"
        >
          {dataSourceLabel}
        </div>
      </div>
    </header>
  );
}
