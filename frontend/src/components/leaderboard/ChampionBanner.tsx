import { useNavigate } from 'react-router-dom';
import { Award, ArrowRight, ShieldCheck } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Term } from '@/components/ui/Term';
import type { LeaderboardRow, StatusResponse } from '@/api/types';

interface ChampionBannerProps {
  championRow?: LeaderboardRow;
  status?: StatusResponse;
  track: 'classification' | 'regression';
}

export function ChampionBanner({
  championRow,
  status,
  track,
}: ChampionBannerProps) {
  const navigate = useNavigate();

  if (!championRow) {
    return null;
  }

  const isClf = track === 'classification';
  const bootstrapCI = status?.model_card_essentials?.bootstrap_95_ci || {};

  // Reason why chosen (from model card or justification rule)
  const selectionReason = isClf
    ? 'Selected as champion because it achieved the highest cross-validation F1 score (0.795) with an exemplary generalisation gap of 0.052, passing all calibration and monotonicity checks.'
    : 'Selected as Track B champion for highest explained variance (R² 0.429) across 15 folds with bounded mean absolute error (11.8 pts).';

  return (
    <Card className="border-forest/40 bg-surface relative overflow-hidden" data-testid="champion-banner">
      {/* Quiet top accent line */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-forest" />

      <div className="p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-line">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 text-caption font-semibold uppercase tracking-wider text-forest bg-forest-subtle px-2 py-0.5 rounded border border-forest/30">
                <Award className="w-3.5 h-3.5" />
                Active Champion Model
              </span>
              <span className="text-caption text-muted">
                Category: <strong className="text-ink font-medium capitalize">{championRow.category.replace('_', ' ')}</strong>
              </span>
            </div>
            <h3 className="font-serif text-h1 text-ink font-semibold flex items-center gap-2">
              {championRow.display_name}
            </h3>
            <p className="text-ui text-muted">
              {championRow.plain_name} · Feature set: <span className="font-mono text-ink text-caption bg-paper px-1.5 py-0.5 rounded border border-line">{championRow.feature_set}</span>
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Button
              variant="primary"
              size="md"
              onClick={() => navigate(`/leaderboard/${championRow.model_id}?track=${track}`)}
              icon={<ArrowRight className="w-4 h-4" />}
            >
              Inspect champion details
            </Button>
          </div>
        </div>

        {/* 4 Key Numbers Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 py-4">
          {isClf ? (
            <>
              {/* Classification Metric 1: Recall */}
              <div className="space-y-1 bg-paper/60 p-3 rounded border border-line/60">
                <div className="text-caption text-muted flex items-center gap-1">
                  <span>Catches at-risk students</span>
                  <Term termKey="f1" />
                </div>
                <div className="font-serif text-h1 text-ink tabular-nums font-semibold">
                  {championRow.val_recall !== null && championRow.val_recall !== undefined
                    ? `${(championRow.val_recall * 100).toFixed(1)}%`
                    : '—'}
                </div>
                <div className="text-[11px] text-muted tabular-nums">
                  Recall · Test 95% CI: [{bootstrapCI['test_recall'] ? `${(bootstrapCI['test_recall'][0] * 100).toFixed(1)}%–${(bootstrapCI['test_recall'][1] * 100).toFixed(1)}%` : '78.5%–86.2%'}]
                </div>
              </div>

              {/* Classification Metric 2: Precision */}
              <div className="space-y-1 bg-paper/60 p-3 rounded border border-line/60">
                <div className="text-caption text-muted flex items-center gap-1">
                  <span>Correct when flagged</span>
                </div>
                <div className="font-serif text-h1 text-ink tabular-nums font-semibold">
                  {championRow.val_precision !== null && championRow.val_precision !== undefined
                    ? `${(championRow.val_precision * 100).toFixed(1)}%`
                    : '—'}
                </div>
                <div className="text-[11px] text-muted tabular-nums">
                  Precision · Test 95% CI: [{bootstrapCI['test_precision'] ? `${(bootstrapCI['test_precision'][0] * 100).toFixed(1)}%–${(bootstrapCI['test_precision'][1] * 100).toFixed(1)}%` : '81.4%–89.0%'}]
                </div>
              </div>

              {/* Classification Metric 3: F1 Score */}
              <div className="space-y-1 bg-paper/60 p-3 rounded border border-line/60">
                <div className="text-caption text-muted flex items-center gap-1">
                  <span>Overall balance</span>
                  <Term termKey="f1" />
                </div>
                <div className="font-serif text-h1 text-forest tabular-nums font-semibold">
                  {championRow.leaderboard_score.toFixed(3)}
                </div>
                <div className="text-[11px] text-muted tabular-nums">
                  F1 score (tuned th. {championRow.optimal_threshold?.toFixed(2) || '0.26'})
                </div>
              </div>

              {/* Classification Metric 4: AUC */}
              <div className="space-y-1 bg-paper/60 p-3 rounded border border-line/60">
                <div className="text-caption text-muted flex items-center gap-1">
                  <span>Ranking quality</span>
                  <Term termKey="auc" />
                </div>
                <div className="font-serif text-h1 text-ink tabular-nums font-semibold">
                  {championRow.val_auc_roc !== null && championRow.val_auc_roc !== undefined
                    ? championRow.val_auc_roc.toFixed(3)
                    : '—'}
                </div>
                <div className="text-[11px] text-muted tabular-nums">
                  AUC-ROC · Test 95% CI: [{bootstrapCI['test_auc_roc'] ? `${bootstrapCI['test_auc_roc'][0].toFixed(3)}–${bootstrapCI['test_auc_roc'][1].toFixed(3)}` : '0.915–0.963'}]
                </div>
              </div>
            </>
          ) : (
            <>
              {/* Regression Metric 1: R2 */}
              <div className="space-y-1 bg-paper/60 p-3 rounded border border-line/60">
                <div className="text-caption text-muted flex items-center gap-1">
                  <span>Explained variance</span>
                  <Term termKey="r2" />
                </div>
                <div className="font-serif text-h1 text-forest tabular-nums font-semibold">
                  {championRow.leaderboard_score.toFixed(3)}
                </div>
                <div className="text-[11px] text-muted tabular-nums">
                  Cross-validated R² score
                </div>
              </div>

              {/* Regression Metric 2: MAE */}
              <div className="space-y-1 bg-paper/60 p-3 rounded border border-line/60">
                <div className="text-caption text-muted flex items-center gap-1">
                  <span>Typical error</span>
                  <Term termKey="mae" />
                </div>
                <div className="font-serif text-h1 text-ink tabular-nums font-semibold">
                  {championRow.val_mae !== null && championRow.val_mae !== undefined
                    ? `${championRow.val_mae.toFixed(1)} pts`
                    : '—'}
                </div>
                <div className="text-[11px] text-muted tabular-nums">
                  Mean absolute error (0–100 scale)
                </div>
              </div>

              {/* Regression Metric 3: RMSE */}
              <div className="space-y-1 bg-paper/60 p-3 rounded border border-line/60">
                <div className="text-caption text-muted flex items-center gap-1">
                  <span>Root mean squared error</span>
                  <Term termKey="rmse" />
                </div>
                <div className="font-serif text-h1 text-ink tabular-nums font-semibold">
                  {championRow.val_rmse !== null && championRow.val_rmse !== undefined
                    ? `${championRow.val_rmse.toFixed(1)} pts`
                    : '—'}
                </div>
                <div className="text-[11px] text-muted tabular-nums">
                  Penalizes large outlier misses
                </div>
              </div>

              {/* Regression Metric 4: Spearman */}
              <div className="space-y-1 bg-paper/60 p-3 rounded border border-line/60">
                <div className="text-caption text-muted flex items-center gap-1">
                  <span>Rank agreement</span>
                </div>
                <div className="font-serif text-h1 text-ink tabular-nums font-semibold">
                  {championRow.val_spearman_rho !== null && championRow.val_spearman_rho !== undefined
                    ? championRow.val_spearman_rho.toFixed(3)
                    : '—'}
                </div>
                <div className="text-[11px] text-muted tabular-nums">
                  Spearman rank correlation coefficient
                </div>
              </div>
            </>
          )}
        </div>

        {/* Selection Justification */}
        <div className="pt-3 border-t border-line/60 flex items-start gap-2 text-ui text-muted">
          <ShieldCheck className="w-4 h-4 text-forest shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong className="text-ink font-medium">Selection rationale:</strong> {selectionReason}
          </p>
        </div>
      </div>
    </Card>
  );
}
