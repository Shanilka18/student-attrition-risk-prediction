import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Term } from '@/components/ui/Term';
import { TooltipProvider } from '@/components/ui/Tooltip';

describe('Term Component', () => {
  it('renders plain-English label by default', () => {
    render(
      <TooltipProvider>
        <Term termKey="f1" />
      </TooltipProvider>
    );
    // Plain label for F1 in glossary is "At-risk balance score"
    expect(screen.getByText('At-risk balance score')).toBeInTheDocument();
  });

  it('renders custom label when provided', () => {
    render(
      <TooltipProvider>
        <Term termKey="auc" customLabel="Custom Metric Name" />
      </TooltipProvider>
    );
    expect(screen.getByText('Custom Metric Name')).toBeInTheDocument();
  });

  it('includes formal academic name in aria-label for accessibility', () => {
    render(
      <TooltipProvider>
        <Term termKey="mei" />
      </TooltipProvider>
    );

    const termElement = screen.getByRole('button');
    expect(termElement).toHaveAttribute(
      'aria-label',
      'Engagement risk score: Multimodal Engagement Index (MEI)'
    );
  });

  it('renders formal academic name and summary in tooltip when opened', () => {
    render(
      <TooltipProvider>
        <Term termKey="mei" open={true} />
      </TooltipProvider>
    );

    expect(screen.getByText('Multimodal Engagement Index (MEI)')).toBeInTheDocument();
    expect(
      screen.getByText(/A continuous scale from 0 to 100 estimating how likely a student is/i)
    ).toBeInTheDocument();
  });
});
