import { describe, it, expect } from 'vitest';
import React from 'react';
import { render, screen } from '@testing-library/react';
import { SyntheticBanner } from '../components/common/SyntheticBanner';
import { ERIGauge } from '../components/charts/ShapWaterfall';
import { Card } from '../components/common/Card';
import { ThemeProvider } from '../context/ThemeContext';

describe('Frontend Component Tests', () => {
  it('renders synthetic banner with mandatory warning text', () => {
    render(<SyntheticBanner bannerText="SYNTHETIC DATA - pipeline validation only" />);
    expect(screen.getByText(/SYNTHETIC DATA - PIPELINE VALIDATION ONLY/i)).toBeTruthy();
  });

  it('renders ERI gauge with score and risk band', () => {
    render(<ERIGauge score={48.5} band="Moderate" />);
    expect(screen.getByText(/48.5/i)).toBeTruthy();
    expect(screen.getByText(/Moderate Risk Tier/i)).toBeTruthy();
  });

  it('renders Card with title, subtitle, and about popover', () => {
    render(
      <Card
        title="Test Card Title"
        subtitle="Test Card Subtitle"
        aboutText="Explains the chart purpose."
      >
        <div>Card Inner Content</div>
      </Card>
    );
    expect(screen.getByText(/Test Card Title/i)).toBeTruthy();
    expect(screen.getByText(/Test Card Subtitle/i)).toBeTruthy();
    expect(screen.getByText(/Card Inner Content/i)).toBeTruthy();
  });
});
