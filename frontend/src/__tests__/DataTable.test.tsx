import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DataTable } from '@/components/ui/DataTable';
import type { ColumnDef } from '@tanstack/react-table';

interface Item {
  id: string;
  name: string;
  score: number;
}

const TEST_DATA: Item[] = [
  { id: '1', name: 'Alpha', score: 85 },
  { id: '2', name: 'Gamma', score: 62 },
  { id: '3', name: 'Delta', score: 94 },
];

const TEST_COLUMNS: ColumnDef<Item, any>[] = [
  {
    accessorKey: 'name',
    header: 'Name',
  },
  {
    accessorKey: 'score',
    header: 'Score',
    cell: ({ row }) => <span className="tabular-nums">{row.original.score}</span>,
  },
];

describe('DataTable Component', () => {
  it('renders all column headers and data rows', () => {
    render(<DataTable data={TEST_DATA} columns={TEST_COLUMNS} />);
    expect(screen.getByText('Name')).toBeInTheDocument();
    expect(screen.getByText('Score')).toBeInTheDocument();
    expect(screen.getByText('Alpha')).toBeInTheDocument();
    expect(screen.getByText('Gamma')).toBeInTheDocument();
    expect(screen.getByText('Delta')).toBeInTheDocument();
  });

  it('sorts rows when column header is clicked', () => {
    render(<DataTable data={TEST_DATA} columns={TEST_COLUMNS} />);
    const scoreHeader = screen.getByText('Score');

    // Initial order: Alpha, Gamma, Delta
    const getRenderedNames = () =>
      screen
        .getAllByRole('row')
        .slice(1) // skip thead
        .map((r) => r.children[0].textContent);

    expect(getRenderedNames()).toEqual(['Alpha', 'Gamma', 'Delta']);

    // Click score header -> Sorts descending first (highest score first: Delta 94, Alpha 85, Gamma 62)
    fireEvent.click(scoreHeader);
    expect(getRenderedNames()).toEqual(['Delta', 'Alpha', 'Gamma']);

    // Click score header again -> Sorts ascending (lowest score first: Gamma 62, Alpha 85, Delta 94)
    fireEvent.click(scoreHeader);
    expect(getRenderedNames()).toEqual(['Gamma', 'Alpha', 'Delta']);
  });

  it('filters rows based on search input', () => {
    render(<DataTable data={TEST_DATA} columns={TEST_COLUMNS} searchPrompt="Search names..." />);
    const searchInput = screen.getByRole('textbox');

    fireEvent.change(searchInput, { target: { value: 'Del' } });
    expect(screen.getByText('Delta')).toBeInTheDocument();
    expect(screen.queryByText('Alpha')).not.toBeInTheDocument();
    expect(screen.queryByText('Gamma')).not.toBeInTheDocument();
  });
});
