import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { GapBar } from '@/components/ui/GapBar';

describe('GapBar Component', () => {
  it('renders positive gap with plus sign and good fit status', () => {
    render(<GapBar gap={0.035} status="Good fit" />);
    
    expect(screen.getByText('+0.035')).toBeInTheDocument();
    expect(screen.getByText('Good fit')).toBeInTheDocument();
    
    const progress = screen.getByRole('progressbar');
    expect(progress).toHaveAttribute('aria-valuenow', '0.035');
    expect(progress).toHaveAttribute('aria-label', 'Generalisation gap +0.035');
    expect(progress).toHaveClass('bg-risk-low');
  });

  it('renders slight overfit with stone warning tone', () => {
    render(<GapBar gap={0.075} status="Slight overfit" />);
    
    expect(screen.getByText('+0.075')).toBeInTheDocument();
    expect(screen.getByText('Slight overfit')).toBeInTheDocument();
    
    const progress = screen.getByRole('progressbar');
    expect(progress).toHaveClass('bg-stone');
  });

  it('renders severe overfit with high risk token styling', () => {
    render(<GapBar gap={0.165} status="Overfit" />);
    
    expect(screen.getByText('+0.165')).toBeInTheDocument();
    expect(screen.getByText('Overfit')).toBeInTheDocument();
    
    const progress = screen.getByRole('progressbar');
    expect(progress).toHaveClass('bg-risk-high');
  });

  it('handles negative gap correctly', () => {
    render(<GapBar gap={-0.012} status="Good fit" />);
    
    expect(screen.getByText('-0.012')).toBeInTheDocument();
    const progress = screen.getByRole('progressbar');
    expect(progress).toHaveAttribute('aria-valuenow', '-0.012');
  });
});
