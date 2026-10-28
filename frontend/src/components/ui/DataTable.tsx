import { useState } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  getFilteredRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
  type VisibilityState,
} from '@tanstack/react-table';
import { ArrowUpDown, ArrowUp, ArrowDown, Download, Search, SlidersHorizontal, ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from './Button';
import { IconButton } from './IconButton';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { cn } from '@/utils/cn';

export interface DataTableProps<TData> {
  data: TData[];
  columns: ColumnDef<TData, any>[];
  searchPrompt?: string;
  defaultPageSize?: number;
  filename?: string;
  enableExport?: boolean;
  enableColumnVisibility?: boolean;
  enableSearch?: boolean;
  onRowClick?: (row: TData) => void;
  className?: string;
}

export function DataTable<TData>({
  data,
  columns,
  searchPrompt = 'Search records...',
  defaultPageSize = 10,
  filename = 'export.csv',
  enableExport = true,
  enableColumnVisibility = true,
  enableSearch = true,
  onRowClick,
  className,
}: DataTableProps<TData>) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState('');
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});

  const table = useReactTable({
    data,
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
    getPaginationRowModel: getPaginationRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    initialState: {
      pagination: {
        pageSize: defaultPageSize,
      },
    },
  });

  // Client-side CSV export
  const handleExportCSV = () => {
    const visibleCols = table.getVisibleLeafColumns();
    const headers = visibleCols.map((c) => c.id || (typeof c.columnDef.header === 'string' ? c.columnDef.header : 'col'));
    const rows = table.getFilteredRowModel().rows.map((row) =>
      visibleCols.map((c) => {
        const val = row.getValue(c.id);
        if (val === null || val === undefined) return '';
        const str = String(val);
        return str.includes(',') ? `"${str.replace(/"/g, '""')}"` : str;
      })
    );

    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
  };

  return (
    <div className={cn('space-y-3', className)}>
      {/* Top Toolbar */}
      {(enableSearch || enableColumnVisibility || enableExport) && (
        <div className="flex flex-wrap items-center justify-between gap-3">
          {enableSearch && (
            <div className="relative flex-1 min-w-[200px] max-w-sm">
              <Search className="w-4 h-4 text-muted absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="text"
                value={globalFilter ?? ''}
                onChange={(e) => setGlobalFilter(e.target.value)}
                aria-label={searchPrompt}
                title={searchPrompt}
                className="w-full h-9 pl-9 pr-3 text-table bg-surface border border-line rounded-md text-ink focus:outline-none focus:ring-2 focus:ring-forest focus:border-forest"
              />
            </div>
          )}

          <div className="flex items-center gap-2 ml-auto">
            {enableColumnVisibility && (
              <DropdownMenu.Root>
                <DropdownMenu.Trigger asChild>
                  <Button variant="secondary" size="sm" icon={<SlidersHorizontal className="w-3.5 h-3.5" />}>
                    Columns
                  </Button>
                </DropdownMenu.Trigger>
                <DropdownMenu.Portal>
                  <DropdownMenu.Content
                    align="end"
                    className="z-50 min-w-[160px] bg-surface border border-line rounded-md p-1.5 shadow-menu text-table text-ink"
                  >
                    <div className="px-2 py-1 text-caption text-muted font-medium border-b border-line mb-1">
                      Toggle Columns
                    </div>
                    {table
                      .getAllLeafColumns()
                      .filter((col) => col.getCanHide())
                      .map((column) => {
                        return (
                          <DropdownMenu.CheckboxItem
                            key={column.id}
                            checked={column.getIsVisible()}
                            onCheckedChange={(val) => column.toggleVisibility(Boolean(val))}
                            className="flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer hover:bg-paper text-table select-none outline-none focus:bg-paper"
                          >
                            <input
                              type="checkbox"
                              checked={column.getIsVisible()}
                              readOnly
                              className="accent-forest rounded"
                            />
                            <span>{column.id}</span>
                          </DropdownMenu.CheckboxItem>
                        );
                      })}
                  </DropdownMenu.Content>
                </DropdownMenu.Portal>
              </DropdownMenu.Root>
            )}

            {enableExport && (
              <Button
                variant="secondary"
                size="sm"
                onClick={handleExportCSV}
                icon={<Download className="w-3.5 h-3.5" />}
              >
                Export CSV
              </Button>
            )}
          </div>
        </div>
      )}

      {/* Table Container */}
      <div className="border border-line rounded-md bg-surface overflow-x-auto shadow-none">
        <table className="w-full text-table border-collapse">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="border-b border-line bg-[#FAF8F5]">
                {headerGroup.headers.map((header) => {
                  const canSort = header.column.getCanSort();
                  const isSorted = header.column.getIsSorted();
                  return (
                    <th
                      key={header.id}
                      colSpan={header.colSpan}
                      onClick={header.column.getToggleSortingHandler()}
                      className={cn(
                        'px-3.5 py-2.5 text-left text-caption font-semibold text-muted tracking-wider select-none whitespace-nowrap',
                        canSort && 'cursor-pointer hover:text-ink hover:bg-paper/60 transition-colors'
                      )}
                    >
                      <div className="flex items-center gap-1.5">
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {canSort && (
                          <span className="text-muted shrink-0">
                            {isSorted === 'asc' ? (
                              <ArrowUp className="w-3.5 h-3.5 text-forest" />
                            ) : isSorted === 'desc' ? (
                              <ArrowDown className="w-3.5 h-3.5 text-forest" />
                            ) : (
                              <ArrowUpDown className="w-3 h-3 opacity-40" />
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
          <tbody className="divide-y divide-line">
            {table.getRowModel().rows.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <tr
                  key={row.id}
                  onClick={() => onRowClick?.(row.original)}
                  className={cn(
                    'transition-colors duration-75',
                    onRowClick ? 'cursor-pointer hover:bg-paper' : 'hover:bg-[#FAF8F5]'
                  )}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="px-3.5 py-2 text-ink whitespace-nowrap">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={columns.length} className="px-4 py-8 text-center text-muted text-caption">
                  No records match your filter criteria.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-caption text-muted px-1">
        <div className="flex items-center gap-2">
          <span>Rows per page:</span>
          <select
            value={table.getState().pagination.pageSize}
            onChange={(e) => table.setPageSize(Number(e.target.value))}
            className="h-7 px-2 bg-surface border border-line rounded text-caption text-ink focus:outline-none focus:ring-1 focus:ring-forest"
          >
            {[10, 20, 30, 50].map((pageSize) => (
              <option key={pageSize} value={pageSize}>
                {pageSize}
              </option>
            ))}
          </select>
          <span className="tabular-nums">
            Showing{' '}
            {table.getState().pagination.pageIndex * table.getState().pagination.pageSize + 1}–
            {Math.min(
              (table.getState().pagination.pageIndex + 1) * table.getState().pagination.pageSize,
              table.getFilteredRowModel().rows.length
            )}{' '}
            of {table.getFilteredRowModel().rows.length.toLocaleString()} entries
          </span>
        </div>

        <div className="flex items-center gap-1">
          <IconButton
            variant="secondary"
            size="sm"
            aria-label="Previous page"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            <ChevronLeft className="w-4 h-4" />
          </IconButton>
          <span className="px-2 tabular-nums font-medium text-ink">
            Page {table.getState().pagination.pageIndex + 1} of {Math.max(1, table.getPageCount())}
          </span>
          <IconButton
            variant="secondary"
            size="sm"
            aria-label="Next page"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            <ChevronRight className="w-4 h-4" />
          </IconButton>
        </div>
      </div>
    </div>
  );
}
