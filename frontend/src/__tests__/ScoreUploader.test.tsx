import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ScoreUploader } from '@/components/score/ScoreUploader';

describe('ScoreUploader', () => {
  it('shows an accessible error for an unsupported file', () => {
    const onFileSelect = vi.fn();
    render(
      <ScoreUploader
        onFileSelect={onFileSelect}
        onOpenSingleScore={vi.fn()}
      />,
    );

    const input = screen.getByLabelText('Upload survey file') as HTMLInputElement;
    const file = new File(['invalid'], 'survey.pdf', { type: 'application/pdf' });
    fireEvent.change(input, { target: { files: [file] } });

    expect(screen.getByRole('alert')).toHaveTextContent(
      'Choose a CSV or Excel survey spreadsheet',
    );
    expect(onFileSelect).not.toHaveBeenCalled();
  });
});
