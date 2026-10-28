import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from '@/components/ui/Button';

describe('Button Component', () => {
  it('renders children and default secondary variant correctly', () => {
    render(<Button>Click Me</Button>);
    const btn = screen.getByRole('button', { name: /click me/i });
    expect(btn).toBeInTheDocument();
    expect(btn).toHaveClass('border-line');
  });

  it('renders primary variant with forest styling', () => {
    render(<Button variant="primary">Submit</Button>);
    const btn = screen.getByRole('button', { name: /submit/i });
    expect(btn).toHaveClass('bg-forest');
    expect(btn).toHaveClass('text-surface');
  });

  it('disables button and displays loading indicator when loading is true', () => {
    const handleClick = vi.fn();
    render(
      <Button loading onClick={handleClick}>
        Saving
      </Button>
    );
    const btn = screen.getByRole('button');
    expect(btn).toBeDisabled();
    expect(btn).toHaveAttribute('aria-busy', 'true');
    fireEvent.click(btn);
    expect(handleClick).not.toHaveBeenCalled();
  });

  it('handles click events when enabled', () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Action</Button>);
    const btn = screen.getByRole('button', { name: /action/i });
    fireEvent.click(btn);
    expect(handleClick).toHaveBeenCalledTimes(1);
  });
});
