import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatusTag, Tag } from '@/components/ui/Tag';

describe('StatusTag Component', () => {
  it('renders "Good fit" tag with low-risk subtle styling', () => {
    const { container } = render(<StatusTag status="Good fit" />);
    expect(screen.getByText('Good fit')).toBeInTheDocument();
    expect(container.firstChild).toHaveClass('text-risk-low');
  });

  it('renders "Slight overfit" tag with warning tone', () => {
    const { container } = render(<StatusTag status="Slight overfit" />);
    expect(screen.getByText('Slight overfit')).toBeInTheDocument();
    expect(container.firstChild).toHaveClass('text-[#7A5400]');
  });

  it('renders "Overfit" tag with high-risk tone', () => {
    const { container } = render(<StatusTag status="Overfit" />);
    expect(screen.getByText('Overfit')).toBeInTheDocument();
    expect(container.firstChild).toHaveClass('text-risk-high');
  });

  it('renders risk tiers correctly', () => {
    const { unmount } = render(<StatusTag status="low" />);
    expect(screen.getByText('Low risk')).toBeInTheDocument();
    unmount();

    render(<StatusTag status="moderate" />);
    expect(screen.getByText('Moderate risk')).toBeInTheDocument();
  });

  it('renders longitudinal patterns', () => {
    const { unmount } = render(<StatusTag status="consistent" />);
    expect(screen.getByText('Consistent')).toBeInTheDocument();
    unmount();

    render(<StatusTag status="declining" />);
    expect(screen.getByText('Declining')).toBeInTheDocument();
  });

  it('renders custom neutral fallback for arbitrary text', () => {
    render(<StatusTag status="Custom Status" />);
    expect(screen.getByText('Custom Status')).toBeInTheDocument();
  });
});

describe('Tag Component', () => {
  it('renders children and indicator dot by default', () => {
    const { container } = render(<Tag variant="neutral">Default Tag</Tag>);
    expect(screen.getByText('Default Tag')).toBeInTheDocument();
    const dot = container.querySelector('span[aria-hidden="true"]');
    expect(dot).toBeInTheDocument();
  });

  it('renders with custom icon instead of dot', () => {
    const { container } = render(
      <Tag icon={<span data-testid="custom-icon">*</span>}>With Icon</Tag>
    );
    expect(screen.getByTestId('custom-icon')).toBeInTheDocument();
    const dot = container.querySelector('span[aria-hidden="true"]');
    expect(dot).not.toBeInTheDocument();
  });
});
