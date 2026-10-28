import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
  type VisibilityState,
} from '@tanstack/react-table';
import {
  Search,
  Filter,
  Download,
  SlidersHorizontal,
  ChevronUp,
  ChevronDown,
  ChevronsUpDown,
  ArrowRight,
  X,
  Trophy,
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { GapBar } from '@/components/ui/GapBar';
import { Tooltip } from '@/components/ui/Tooltip';
import type { LeaderboardRow } from '@/api/types';
import { cn } from '@/utils/cn';

interface LeaderboardTableProps {
  rows: LeaderboardRow[];
  track: 'classification' | 'regression';
  selectedCategory: string | null;
  onSelectCategory: (category: string | null) => void;
  selectedModels: string[];
  onToggleSelectModel: (modelId: string) => void;
  onClearSelectedModels: () => void;
  onCompare: () => void;
}

export function LeaderboardTable({
  rows,
  track,
  selectedCategory,
  onSelectCategory,
  selectedModels,
  onToggleSelectModel,
  onClearSelectedModels,
  onCompare,
}: LeaderboardTableProps) {
  const navigate = useNavigate();
  const isClf = track === 'classification';

  // Table local state
  const [globalFilter, setGlobalFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [sorting, setSorting] = useState<SortingState>([
    { id: 'leaderboard_score', desc: true },
  ]);
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});
  const [showColumnChooser, setShowColumnChooser] = useState(false);

  // Filter rows by category and overfit status
  const filteredRows = useMemo(() => {
    return rows.filter((row) => {
      if (selectedCategory && row.category !== selectedCategory) {
        return false;
      }
      if (statusFilter !== 'all') {
        const norm = row.overfit_status.toLowerCase();
        if (statusFilter === 'good' && !norm.includes('good')) return false;
        if (statusFilter === 'slight' && !norm.includes('slight')) return false;
        if (statusFilter === 'overfit' && (!norm.includes('overfit') || norm.includes('slight'))) return false;
      }
      return true;
    });
  }, [rows, selectedCategory, statusFilter]);

  // Define columns
  const columns = useMemo<ColumnDef<LeaderboardRow>[]>(() => {
    return [
      {
        id: 'select',
        header: () => <span className="sr-only">Select</span>,
        cell: ({ row }) => {
          const modelId = row.original.model_id;
          const isChecked = selectedModels.includes(modelId);
          const isMaxSelected = selectedModels.length >= 4 && !isChecked;

          return (
            <div className="flex items-center justify-center" onClick={(e) => e.stopPropagation()}>
              <input
                type="checkbox"
                aria-label={`Select ${row.original.display_name} for comparison`}
                checked={isChecked}
                disabled={isMaxSelected}
                onChange={() => onToggleSelectModel(modelId)}
                className="w-4 h-4 text-forest rounded border-line focus:ring-forest cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
              />
            </div>
          );
        },
        enableSorting: false,
        size: 36,
      },
      {
        accessorKey: 'rank',
        header: 'Rank',
        cell: ({ row }) => (
          <span className="font-mono text-muted text-caption font-medium tabular-nums">
            #{row.original.rank}
          </span>
        ),
        size: 54,
      },
      {
        accessorKey: 'display_name',
        header: 'Model',
        cell: ({ row }) => {
          const r = row.original;
          return (
            <div className="space-y-0.5">
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="font-medium text-ink hover:text-forest transition-colors">
                  {r.display_name}
                </span>
                {r.is_champion && (
                  <span className="inline-flex items-center text-[10px] bg-forest text-surface font-semibold px-1 py-0.2 rounded" title="Champion Model">
                    <Trophy className="w-2.5 h-2.5 mr-0.5" />
                    Champion
                  </span>
                )}
                {r.is_remediated && (
                  <Tooltip content="Regularisation and capacity reduction applied to constrain over-fitting">
                    <span className="inline-flex items-center text-[10px] bg-stone-subtle text-stone border border-stone/30 font-medium px-1.5 py-0.2 rounded">
                      Remediated
                    </span>
                  </Tooltip>
                )}
              </div>
              <div className="text-caption text-muted truncate max-w-xs">
                {r.plain_name}
              </div>
            </div>
          );
        },
        minSize: 220,
      },
      {
        accessorKey: 'category',
        header: 'Category',
        cell: ({ row }) => (
          <span className="text-caption capitalize text-muted bg-paper px-2 py-0.5 rounded border border-line whitespace-nowrap">
            {row.original.category.replace('_', ' ')}
          </span>
        ),
        size: 130,
      },
      {
        accessorKey: 'leaderboard_score',
        header: () => (isClf ? 'Leaderboard (F1)' : 'Leaderboard (R²)'),
        cell: ({ row }) => (
          <span className="font-serif text-body font-semibold text-ink tabular-nums">
            {row.original.leaderboard_score.toFixed(3)}
          </span>
        ),
        size: 110,
      },
      ...(isClf
        ? [
            {
              id: 'auc_roc',
              accessorKey: 'val_auc_roc',
              header: 'AUC-ROC',
              cell: ({ row }: any) => (
                <span className="font-sans text-ui text-ink tabular-nums">
                  {row.original.val_auc_roc !== null && row.original.val_auc_roc !== undefined
                    ? row.original.val_auc_roc.toFixed(3)
                    : '—'}
                </span>
              ),
              size: 90,
            },
            {
              id: 'bal_acc',
              accessorKey: 'val_balanced_accuracy',
              header: 'Bal. Acc.',
              cell: ({ row }: any) => (
                <span className="font-sans text-ui text-muted tabular-nums">
                  {row.original.val_balanced_accuracy !== null && row.original.val_balanced_accuracy !== undefined
                    ? `${(row.original.val_balanced_accuracy * 100).toFixed(1)}%`
                    : '—'}
                </span>
              ),
              size: 90,
            },
            {
              id: 'mcc',
              accessorKey: 'val_mcc',
              header: 'MCC',
              cell: ({ row }: any) => (
                <span className="font-sans text-ui text-muted tabular-nums">
                  {row.original.val_mcc !== null && row.original.val_mcc !== undefined
                    ? row.original.val_mcc.toFixed(3)
                    : '—'}
                </span>
              ),
              size: 80,
            },
          ]
        : [
            {
              id: 'rmse',
              accessorKey: 'val_rmse',
              header: 'RMSE',
              cell: ({ row }: any) => (
                <span className="font-sans text-ui text-ink tabular-nums">
                  {row.original.val_rmse !== null && row.original.val_rmse !== undefined
                    ? `${row.original.val_rmse.toFixed(1)} pts`
                    : '—'}
                </span>
              ),
              size: 90,
            },
            {
              id: 'mae',
              accessorKey: 'val_mae',
              header: 'MAE',
              cell: ({ row }: any) => (
                <span className="font-sans text-ui text-muted tabular-nums">
                  {row.original.val_mae !== null && row.original.val_mae !== undefined
                    ? `${row.original.val_mae.toFixed(1)} pts`
                    : '—'}
                </span>
              ),
              size: 90,
            },
            {
              id: 'spearman',
              accessorKey: 'val_spearman_rho',
              header: 'Spearman',
              cell: ({ row }: any) => (
                <span className="font-sans text-ui text-muted tabular-nums">
                  {row.original.val_spearman_rho !== null && row.original.val_spearman_rho !== undefined
                    ? row.original.val_spearman_rho.toFixed(3)
                    : '—'}
                </span>
              ),
              size: 90,
            },
          ]),
      {
        accessorKey: 'generalization_gap',
        header: 'Train–CV Gap',
        cell: ({ row }) => (
          <GapBar
            gap={row.original.generalization_gap}
            status={row.original.overfit_status}
          />
        ),
        minSize: 180,
      },
      {
        id: 'test_score',
        header: 'Test Score',
        cell: ({ row }) => {
          const r = row.original;
          const score = isClf ? r.test_f1 : r.test_r2;
          if (score !== null && score !== undefined) {
            return (
              <span className="font-serif font-semibold text-forest text-ui tabular-nums">
                {score.toFixed(3)}
              </span>
            );
          }
          return (
            <Tooltip content="Only the finalists are evaluated on the untouched holdout test set">
              <span className="text-muted font-mono cursor-help">
                —
              </span>
            </Tooltip>
          );
        },
        size: 100,
      },
      {
        accessorKey: 'training_time_s',
        header: 'Train Time',
        cell: ({ row }) => (
          <span className="font-mono text-caption text-muted tabular-nums">
            {row.original.training_time_s < 1
              ? `${(row.original.training_time_s * 1000).toFixed(0)} ms`
              : `${row.original.training_time_s.toFixed(1)} s`}
          </span>
        ),
        size: 90,
      },
    ];
  }, [track, isClf, selectedModels, onToggleSelectModel]);

  const table = useReactTable({
    data: filteredRows,
    columns,
    state: {
      sorting,
      globalFilter,
      columnVisibility,
    },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    onColumnVisibilityChange: setColumnVisibility,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: {
      pagination: {
        pageSize: 50,
      },
    },
  });

  // Export CSV
  const handleExportCsv = () => {
    const headers = [
      'Rank',
      'Model ID',
      'Display Name',
      'Category',
      'Feature Set',
      'Leaderboard Score',
      'Generalization Gap',
      'Overfit Status',
      isClf ? 'CV F1' : 'CV R2',
      isClf ? 'CV AUC' : 'CV RMSE',
      isClf ? 'Test F1' : 'Test R2',
      'Training Time (s)',
    ];

    const csvRows = filteredRows.map((r) => [
      r.rank,
      r.model_id,
      `"${r.display_name.replace(/"/g, '""')}"`,
      r.category,
      r.feature_set,
      r.leaderboard_score.toFixed(4),
      r.generalization_gap.toFixed(4),
      r.overfit_status,
      isClf ? (r.val_f1_optimal?.toFixed(4) || '') : (r.val_r2?.toFixed(4) || ''),
      isClf ? (r.val_auc_roc?.toFixed(4) || '') : (r.val_rmse?.toFixed(4) || ''),
      isClf ? (r.test_f1?.toFixed(4) || '') : (r.test_r2?.toFixed(4) || ''),
      r.training_time_s.toFixed(2),
    ]);

    const csvContent = [headers.join(','), ...csvRows.map((e) => e.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `leaderboard_${track}_${new Date().toISOString().split('T')[0]}.csv`);
    link.click();
  };

  return (
    <Card className="overflow-hidden" data-testid="leaderboard-table-card">
      {/* Controls Bar */}
      <div className="p-4 border-b border-line flex flex-wrap items-center justify-between gap-3 bg-surface">
        <div className="flex flex-wrap items-center gap-3 flex-1 min-w-[260px]">
          {/* Search box */}
          <div className="relative flex-1 max-w-xs">
            <Search className="w-4 h-4 text-muted absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="text"
              value={globalFilter ?? ''}
              onChange={(e) => setGlobalFilter(e.target.value)}
              aria-label="Filter by model or algorithm"
              title="Filter by model or algorithm"
              className="w-full pl-9 pr-3 py-1.5 text-body bg-paper border border-line rounded text-ink focus:outline-none focus:border-forest"
            />
          </div>

          {/* Status filter */}
          <div className="flex items-center gap-1.5 text-caption text-muted">
            <Filter className="w-3.5 h-3.5" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-paper border border-line rounded px-2 py-1 text-ink text-caption focus:outline-none focus:border-forest"
              aria-label="Filter by generalization status"
            >
              <option value="all">All statuses</option>
              <option value="good">Good fit</option>
              <option value="slight">Slight overfit</option>
              <option value="overfit">Overfit</option>
            </select>
          </div>

          {/* Active category filter chip */}
          {selectedCategory && (
            <button
              type="button"
              onClick={() => onSelectCategory(null)}
              className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-forest-subtle text-forest text-caption font-medium border border-forest/30 hover:bg-forest/10 transition-colors"
              title="Click to clear category filter"
            >
              <span>Category: {selectedCategory.replace('_', ' ')}</span>
              <X className="w-3 h-3" />
            </button>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Column chooser */}
          <div className="relative">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setShowColumnChooser((prev) => !prev)}
              icon={<SlidersHorizontal className="w-3.5 h-3.5" />}
            >
              Columns
            </Button>
            {showColumnChooser && (
              <div className="absolute right-0 mt-1 w-48 bg-surface border border-line rounded shadow-popover p-2 z-20 space-y-1 text-ui">
                <div className="text-caption font-semibold text-muted px-2 py-1 border-b border-line mb-1">
                  Toggle Columns
                </div>
                {table.getAllLeafColumns().map((column) => {
                  if (column.id === 'select' || column.id === 'display_name') return null;
                  return (
                    <label
                      key={column.id}
                      className="flex items-center gap-2 px-2 py-1 hover:bg-paper rounded cursor-pointer text-caption"
                    >
                      <input
                        type="checkbox"
                        checked={column.getIsVisible()}
                        onChange={column.getToggleVisibilityHandler()}
                        className="rounded text-forest focus:ring-forest"
                      />
                      <span className="capitalize">{column.id.replace('_', ' ')}</span>
                    </label>
                  );
                })}
              </div>
            )}
          </div>

          {/* Export CSV */}
          <Button
            variant="secondary"
            size="sm"
            onClick={handleExportCsv}
            icon={<Download className="w-3.5 h-3.5" />}
          >
            Export CSV
          </Button>
        </div>
      </div>

      {/* Floating Selection Bar */}
      {selectedModels.length > 0 && (
        <div className="bg-forest-subtle border-b border-forest/30 px-4 py-2.5 flex items-center justify-between text-ui">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-forest">
              {selectedModels.length} of 4 models selected
            </span>
            <span className="text-caption text-muted">
              (Choose up to 4 models to compare side-by-side)
            </span>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={onClearSelectedModels}
              icon={<X className="w-3.5 h-3.5" />}
            >
              Clear
            </Button>
            <Button
              variant="primary"
              size="sm"
              onClick={onCompare}
              icon={<ArrowRight className="w-3.5 h-3.5" />}
            >
              Compare selected ({selectedModels.length})
            </Button>
          </div>
        </div>
      )}

      {/* Table Container */}
      <div className="overflow-x-auto max-h-[640px] relative">
        <table className="w-full text-left text-body border-collapse" data-testid="leaderboard-table">
          <thead className="bg-paper sticky top-0 z-10 border-b border-line text-caption font-semibold text-muted">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  const isSorted = header.column.getIsSorted();
                  return (
                    <th
                      key={header.id}
                      style={{ width: header.getSize() !== 150 ? header.getSize() : undefined }}
                      className={cn(
                        'px-3 py-2.5 font-medium select-none',
                        header.column.getCanSort() && 'cursor-pointer hover:text-ink'
                      )}
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      <div className="flex items-center gap-1">
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {header.column.getCanSort() && (
                          <span className="text-muted shrink-0">
                            {isSorted === 'asc' ? (
                              <ChevronUp className="w-3.5 h-3.5 text-forest" />
                            ) : isSorted === 'desc' ? (
                              <ChevronDown className="w-3.5 h-3.5 text-forest" />
                            ) : (
                              <ChevronsUpDown className="w-3 h-3 opacity-30" />
                            )}
                          </span>
                        )}
                      </div>
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>

          <tbody className="divide-y divide-line bg-surface">
            {table.getRowModel().rows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-4 py-8 text-center text-muted">
                  No models match the active filter criteria.
                </td>
              </tr>
            ) : (
              table.getRowModel().rows.map((row) => {
                const isSelected = selectedModels.includes(row.original.model_id);
                return (
                  <tr
                    key={row.id}
                    onClick={() => navigate(`/leaderboard/${row.original.model_id}?track=${track}`)}
                    className={cn(
                      'hover:bg-paper/70 cursor-pointer transition-colors',
                      isSelected && 'bg-forest-subtle/20'
                    )}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} className="px-3 py-2.5 align-middle">
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="px-4 py-3 border-t border-line flex items-center justify-between text-caption text-muted bg-paper/30">
        <div>
          Showing {table.getRowModel().rows.length} of {rows.length} models
        </div>
        <div className="flex items-center gap-2">
          <span>
            Page {table.getState().pagination.pageIndex + 1} of{' '}
            {Math.max(1, table.getPageCount())}
          </span>
          <div className="flex items-center gap-1">
            <Button
              variant="secondary"
              size="sm"
              disabled={!table.getCanPreviousPage()}
              onClick={() => table.previousPage()}
            >
              Previous
            </Button>
            <Button
              variant="secondary"
              size="sm"
              disabled={!table.getCanNextPage()}
              onClick={() => table.nextPage()}
            >
              Next
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
}
