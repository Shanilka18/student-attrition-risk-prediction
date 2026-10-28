import { useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useLeaderboard, useCategoriesBest, useStatus, usePatterns } from '@/api/queries';
import { Skeleton } from '@/components/ui/Skeleton';
import { ErrorState } from '@/components/ui/ErrorState';
import { ChampionBanner } from '@/components/leaderboard/ChampionBanner';
import { CategoryBestStrip } from '@/components/leaderboard/CategoryBestStrip';
import { LeaderboardTable } from '@/components/leaderboard/LeaderboardTable';
import { LeaderboardCharts } from '@/components/leaderboard/LeaderboardCharts';
import { StateAblationCard } from '@/components/leaderboard/StateAblationCard';
import { cn } from '@/utils/cn';

export function LeaderboardPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  // Track selection: 'classification' | 'regression'
  const trackParam = searchParams.get('track');
  const track: 'classification' | 'regression' =
    trackParam === 'regression' ? 'regression' : 'classification';

  // Category selection filter
  const selectedCategory = searchParams.get('category') || null;

  // Selected models for comparison (up to 4)
  const [selectedModels, setSelectedModels] = useState<string[]>([]);

  // Queries
  const {
    data: leaderboardData,
    isLoading: isLeaderboardLoading,
    error: leaderboardError,
    refetch: refetchLeaderboard,
  } = useLeaderboard(track);

  const {
    data: categoriesBest,
  } = useCategoriesBest(track);

  const { data: statusData } = useStatus();
  const { data: patternsData } = usePatterns();

  // Handlers
  const handleTrackChange = (newTrack: 'classification' | 'regression') => {
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set('track', newTrack);
    // Reset category filter on track switch if needed
    nextParams.delete('category');
    setSearchParams(nextParams);
    setSelectedModels([]);
  };

  const handleSelectCategory = (category: string | null) => {
    const nextParams = new URLSearchParams(searchParams);
    if (category) {
      nextParams.set('category', category);
    } else {
      nextParams.delete('category');
    }
    setSearchParams(nextParams);
  };

  const handleToggleSelectModel = (modelId: string) => {
    setSelectedModels((prev) => {
      if (prev.includes(modelId)) {
        return prev.filter((id) => id !== modelId);
      }
      if (prev.length >= 4) {
        return prev;
      }
      return [...prev, modelId];
    });
  };

  const handleCompare = () => {
    if (selectedModels.length > 0) {
      navigate(`/compare?ids=${selectedModels.join(',')}&track=${track}`);
    }
  };

  // Find champion row in leaderboard
  const championRow = leaderboardData?.rows.find(
    (r) => r.model_id === leaderboardData.champion_id || r.is_champion
  );

  return (
    <div className="space-y-8 max-w-7xl mx-auto" data-testid="leaderboard-page">
      {/* 1. Header & Track Segmented Control */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-line pb-5">
        <div className="space-y-1.5 max-w-2xl">
          <h1 className="font-serif text-page-title text-ink font-semibold">
            Model leaderboard
          </h1>
          <p className="text-body text-muted leading-relaxed">
            We trained many kinds of models on the same survey data and ranked them fairly. The top model is used to calculate the Engagement Index.
          </p>

          <div className="flex items-center gap-2 pt-1 text-caption text-muted flex-wrap">
            <span className="font-medium text-ink bg-paper px-2 py-0.5 rounded border border-line">
              {statusData?.data_source_badge || 'Training data: Sample survey responses (generated)'}
            </span>
            <span>·</span>
            <span className="tabular-nums">
              {statusData?.n_students_train?.toLocaleString() || '1,600'} training students ({statusData?.n_students_total?.toLocaleString() || '2,000'} total cohort)
            </span>
            <span>·</span>
            <span className="tabular-nums font-medium text-ink">
              {leaderboardData?.total_models || 36} candidate models evaluated
            </span>
          </div>
        </div>

        {/* Track switch (segmented control) */}
        <div className="flex items-center bg-paper border border-line p-1 rounded-md shrink-0 self-start md:self-auto" role="tablist" aria-label="Prediction Track">
          <button
            type="button"
            role="tab"
            aria-selected={track === 'classification'}
            onClick={() => handleTrackChange('classification')}
            className={cn(
              'px-3.5 py-1.5 text-ui rounded font-medium transition-all duration-150',
              track === 'classification'
                ? 'bg-surface text-ink font-semibold shadow-sm border border-line'
                : 'text-muted hover:text-ink'
            )}
          >
            At-risk or not (Classification)
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={track === 'regression'}
            onClick={() => handleTrackChange('regression')}
            className={cn(
              'px-3.5 py-1.5 text-ui rounded font-medium transition-all duration-150',
              track === 'regression'
                ? 'bg-surface text-ink font-semibold shadow-sm border border-line'
                : 'text-muted hover:text-ink'
            )}
          >
            Risk score 0–100 (Regression)
          </button>
        </div>
      </div>

      {/* Loading & Error States */}
      {isLeaderboardLoading && (
        <div className="space-y-6">
          <Skeleton className="h-44 w-full rounded" />
          <div className="grid grid-cols-7 gap-3">
            {Array.from({ length: 7 }).map((_, i) => (
              <Skeleton key={i} className="h-24 w-full rounded" />
            ))}
          </div>
          <Skeleton className="h-96 w-full rounded" />
        </div>
      )}

      {leaderboardError && (
        <ErrorState
          title="Failed loading leaderboard benchmark"
          message="Could not read real evaluation artifacts from the active run directory. Please verify that the pipeline artifacts exist on disk."
          onRetry={() => refetchLeaderboard()}
        />
      )}

      {leaderboardData && (
        <>
          {/* 2. Champion Banner */}
          <ChampionBanner
            championRow={championRow}
            status={statusData}
            track={track}
          />

          {/* 3. Best in Each Category Strip */}
          <CategoryBestStrip
            categoriesBest={categoriesBest}
            selectedCategory={selectedCategory}
            onSelectCategory={handleSelectCategory}
            championId={leaderboardData.champion_id}
            maxScore={track === 'classification' ? 1.0 : 0.6}
          />

          {/* 4. Main Leaderboard Table */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="font-serif text-h2 text-ink font-semibold">
                  Complete Cross-Validation Benchmark
                </h2>
                <p className="text-body text-muted">
                  Ranked by 15-fold outer cross-validation validation score. Click any row to inspect training curves, error matrices, and split details.
                </p>
              </div>
            </div>

            <LeaderboardTable
              rows={leaderboardData.rows}
              track={track}
              selectedCategory={selectedCategory}
              onSelectCategory={handleSelectCategory}
              selectedModels={selectedModels}
              onToggleSelectModel={handleToggleSelectModel}
              onClearSelectedModels={() => setSelectedModels([])}
              onCompare={handleCompare}
            />
          </div>

          {/* 5. Empirical Performance Charts */}
          <LeaderboardCharts
            rows={leaderboardData.rows}
            track={track}
            totalStudents={statusData?.n_students_train || 1600}
          />

          {/* 6. "With and without engagement states" Card */}
          <StateAblationCard
            rows={leaderboardData.rows}
            patterns={patternsData}
            track={track}
          />
        </>
      )}
    </div>
  );
}
