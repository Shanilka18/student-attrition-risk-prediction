import { describe, expect, it } from 'vitest';
import { getRiskBandDefinitionLabel } from '@/content/riskBands';

describe('getRiskBandDefinitionLabel', () => {
  it('uses the active model band boundaries when available', () => {
    expect(
      getRiskBandDefinitionLabel(
        { min: 5, max: 29, label: 'Low Risk' },
        'Low Risk'
      )
    ).toBe('Low Risk (5–29)');
  });

  it('does not invent boundaries when the model definition is missing', () => {
    expect(getRiskBandDefinitionLabel(undefined, 'Low Risk')).toBe('Low Risk');
  });
});
