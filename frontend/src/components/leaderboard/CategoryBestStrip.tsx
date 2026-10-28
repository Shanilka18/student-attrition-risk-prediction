import { Trophy, Check } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import type { LeaderboardRow } from '@/api/types';
import { cn } from '@/utils/cn';

interface CategoryBestStripProps {
  categoriesBest?: Record<string, LeaderboardRow>;
  selectedCategory: string | null;
  onSelectCategory: (category: string | null) => void;
  championId?: string;
  maxScore?: number;
}

const CATEGORY_ORDER = [
  { key: 'baseline', label: 'Baseline' },
  { key: 'linear', label: 'Linear' },
  { key: 'classical', label: 'Classical / Shallow' },
  { key: 'boosting', label: 'Boosting & Trees' },
  { key: 'deep', label: 'Deep Learning' },
  { key: 'ensemble', label: 'Ensembles' },
  { key: 'hybrid', label: 'Proposed Hybrid' },
];

export function CategoryBestStrip({
  categoriesBest = {},
  selectedCategory,
  onSelectCategory,
  championId,
  maxScore = 1.0,
}: CategoryBestStripProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-caption text-muted">
        <span className="font-medium text-ink uppercase tracking-wider text-[11px]">
          Best Model per Category (15-Fold Outer Cross-Validation)
        </span>
        {selectedCategory && (
          <button
            type="button"
            onClick={() => onSelectCategory(null)}
            className="text-forest hover:underline focus:outline-none"
          >
            Clear category filter
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-3">
        {CATEGORY_ORDER.map(({ key, label }) => {
          const row = categoriesBest[key];
          const isSelected = selectedCategory === key;
          const isChamp = row?.model_id === championId || row?.is_champion;

          if (!row) {
            return (
              <Card
                key={key}
                className="p-3 opacity-60 bg-paper/40 border-dashed"
              >
                <div className="text-[11px] font-medium text-muted uppercase tracking-wider truncate">
                  {label}
                </div>
                <div className="text-body text-muted italic mt-2">None</div>
              </Card>
            );
          }

          // Calculate percentage width for mini bar
          const score = row.leaderboard_score;
          const scorePct = Math.min(100, Math.max(10, Math.round((score / maxScore) * 100)));

          return (
            <Card
              key={key}
              onClick={() => onSelectCategory(isSelected ? null : key)}
              className={cn(
                'p-3 cursor-pointer transition-all duration-150 hover:border-forest text-left relative flex flex-col justify-between',
                isSelected && 'border-forest ring-1 ring-forest bg-forest-subtle/30',
                isChamp && 'border-forest/60'
              )}
            >
              <div>
                <div className="flex items-center justify-between gap-1 mb-1">
                  <span className="text-[11px] font-semibold text-muted uppercase tracking-wider truncate">
                    {label}
                  </span>
                  {isChamp && (
                    <span
                      title="Overall Champion Model"
                      className="inline-flex items-center text-[10px] bg-forest text-surface font-semibold px-1 py-0.2 rounded"
                    >
                      <Trophy className="w-2.5 h-2.5 mr-0.5" />
                      Champ
                    </span>
                  )}
                </div>

                <div
                  className="font-medium text-ink text-ui line-clamp-1"
                  title={row.display_name}
                >
                  {row.display_name}
                </div>
              </div>

              <div className="mt-3 pt-2 border-t border-line/60">
                <div className="flex items-baseline justify-between">
                  <span className="font-serif text-h2 font-semibold text-ink tabular-nums">
                    {score.toFixed(3)}
                  </span>
                  {isSelected && (
                    <span className="text-forest text-[11px] font-medium flex items-center">
                      <Check className="w-3 h-3 mr-0.5" /> Filtered
                    </span>
                  )}
                </div>

                {/* Mini progress bar */}
                <div className="w-full bg-paper border border-line h-1 rounded-sm mt-1.5 overflow-hidden">
                  <div
                    className={cn(
                      'h-full',
                      isChamp ? 'bg-forest' : 'bg-muted'
                    )}
                    style={{ width: `${scorePct}%` }}
                  />
                </div>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
