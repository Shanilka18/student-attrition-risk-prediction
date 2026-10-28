import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { TooltipProvider } from '@/components/ui/Tooltip';
import { LeaderboardTable } from '@/components/leaderboard/LeaderboardTable';
import type { LeaderboardRow } from '@/api/types';

const MOCK_ROWS: LeaderboardRow[] = [
  {
    rank: 1,
    model_id: 'knn',
    display_name: 'K-Nearest Neighbours (Remediated)',
    plain_name: 'Nearest neighbor classifier',
    category: 'classical',
    feature_set: 'agg',
    leaderboard_score: 0.7952,
    generalization_gap: 0.0412,
    overfit_status: 'Good fit',
    val_f1_optimal: 0.7952,
    val_auc_roc: 0.9412,
    val_balanced_accuracy: 0.8845,
    val_mcc: 0.7612,
    test_f1: 0.7952,
    test_auc_roc: 0.9412,
    training_time_s: 1.45,
    inference_time_ms: 0.12,
    is_champion: true,
    is_category_best: true,
    is_remediated: true,
  },
  {
    rank: 2,
    model_id: 'gru',
    display_name: 'Gated Recurrent Unit (GRU)',
    plain_name: 'Recurrent neural network',
    category: 'deep',
    feature_set: 'sequence',
    leaderboard_score: 0.7814,
    generalization_gap: 0.0498,
    overfit_status: 'Good fit',
    val_f1_optimal: 0.7814,
    val_auc_roc: 0.9321,
    val_balanced_accuracy: 0.8710,
    val_mcc: 0.7420,
    test_f1: 0.7814,
    test_auc_roc: 0.9321,
    training_time_s: 14.82,
    inference_time_ms: 1.45,
    is_champion: false,
    is_category_best: true,
    is_remediated: false,
  },
  {
    rank: 3,
    model_id: 'xgboost_hybrid',
    display_name: 'XGBoost with Engagement States',
    plain_name: 'Tree ensemble with engagement states',
    category: 'hybrid',
    feature_set: 'agg_states',
    leaderboard_score: 0.7745,
    generalization_gap: 0.0621,
    overfit_status: 'Slight overfit',
    val_f1_optimal: 0.7745,
    val_auc_roc: 0.9250,
    val_balanced_accuracy: 0.8650,
    val_mcc: 0.7315,
    test_f1: null,
    test_auc_roc: null,
    training_time_s: 4.21,
    inference_time_ms: 0.35,
    is_champion: false,
    is_category_best: true,
    is_remediated: false,
  },
];

describe('LeaderboardTable Component', () => {
  it('renders table where the first row corresponds to the rank 1 model from data', () => {
    render(
      <BrowserRouter>
        <TooltipProvider>
          <LeaderboardTable
            rows={MOCK_ROWS}
            track="classification"
            selectedCategory={null}
            onSelectCategory={vi.fn()}
            selectedModels={[]}
            onToggleSelectModel={vi.fn()}
            onClearSelectedModels={vi.fn()}
            onCompare={vi.fn()}
          />
        </TooltipProvider>
      </BrowserRouter>
    );

    const table = screen.getByTestId('leaderboard-table');
    expect(table).toBeInTheDocument();

    const rows = table.querySelectorAll('tbody tr');
    expect(rows.length).toBe(3);

    // First row must contain rank 1 model display name and champion badge
    expect(rows[0]).toHaveTextContent('K-Nearest Neighbours (Remediated)');
    expect(rows[0]).toHaveTextContent('0.795');
    expect(rows[0]).toHaveTextContent('Champion');
    expect(rows[0]).toHaveTextContent('Remediated');
  });

  it('filters rows when searching by algorithm name', () => {
    render(
      <BrowserRouter>
        <TooltipProvider>
          <LeaderboardTable
            rows={MOCK_ROWS}
            track="classification"
            selectedCategory={null}
            onSelectCategory={vi.fn()}
            selectedModels={[]}
            onToggleSelectModel={vi.fn()}
            onClearSelectedModels={vi.fn()}
            onCompare={vi.fn()}
          />
        </TooltipProvider>
      </BrowserRouter>
    );

    const searchInput = screen.getByLabelText(/filter by model or algorithm/i);
    fireEvent.change(searchInput, { target: { value: 'XGBoost' } });

    const table = screen.getByTestId('leaderboard-table');
    const rows = table.querySelectorAll('tbody tr');
    expect(rows.length).toBe(1);
    expect(rows[0]).toHaveTextContent('XGBoost with Engagement States');
  });

  it('triggers comparison callbacks when selection checkbox is clicked', () => {
    const handleToggle = vi.fn();
    render(
      <BrowserRouter>
        <TooltipProvider>
          <LeaderboardTable
            rows={MOCK_ROWS}
            track="classification"
            selectedCategory={null}
            onSelectCategory={vi.fn()}
            selectedModels={['knn']}
            onToggleSelectModel={handleToggle}
            onClearSelectedModels={vi.fn()}
            onCompare={vi.fn()}
          />
        </TooltipProvider>
      </BrowserRouter>
    );

    // The selection bar should show "1 of 4 models selected"
    expect(screen.getByText(/1 of 4 models selected/i)).toBeInTheDocument();

    const checkboxes = screen.getAllByRole('checkbox');
    // First checkbox in header or first row
    fireEvent.click(checkboxes[1]);
    expect(handleToggle).toHaveBeenCalledWith('gru');
  });
});
