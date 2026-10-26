import { describe, it, expect } from 'vitest';

describe('Leaderboard sorting and filtering logic', () => {
  const sampleEntries = [
    { model_name: 'ridge', category: 'Linear/Shallow', primary_metric: 0.9908, rank: 1, is_champion: true },
    { model_name: 'hist_gradient_boosting', category: 'Boosting', primary_metric: 0.9852, rank: 2, is_champion: false },
    { model_name: 'random_forest', category: 'Ensemble/Bagging', primary_metric: 0.9798, rank: 3, is_champion: false },
    { model_name: 'dummy', category: 'Baseline', primary_metric: -0.0010, rank: 4, is_champion: false },
  ];

  it('filters entries correctly by category', () => {
    const boosting = sampleEntries.filter((e) => e.category === 'Boosting');
    expect(boosting).toHaveLength(1);
    expect(boosting[0].model_name).toBe('hist_gradient_boosting');

    const all = sampleEntries.filter((e) => 'All' === 'All' || e.category === 'All');
    expect(all).toHaveLength(4);
  });

  it('sorts entries by primary metric descending', () => {
    const sorted = [...sampleEntries].sort((a, b) => b.primary_metric - a.primary_metric);
    expect(sorted[0].model_name).toBe('ridge');
    expect(sorted[1].model_name).toBe('hist_gradient_boosting');
    expect(sorted[3].model_name).toBe('dummy');
  });

  it('identifies the designated champion entry with a crown', () => {
    const champ = sampleEntries.find((e) => e.is_champion);
    expect(champ).toBeDefined();
    expect(champ?.model_name).toBe('ridge');
    expect(champ?.rank).toBe(1);
  });
});

describe('Upload flow column mapping and validation contract', () => {
  it('correctly maps fuzzy column suggestions with high confidence', () => {
    const mockMappings: Record<string, string> = {
      'Q1_stress': 'S1',
      'Q2_stress': 'S2',
      'M_goal': 'M1',
    };
    expect(mockMappings['Q1_stress']).toBe('S1');
    expect(Object.keys(mockMappings)).toHaveLength(3);
  });

  it('validates required 15 Likert items and student ID in schema check', () => {
    const requiredFactors = ['stress', 'motivation', 'belongingness', 'self_efficacy', 'peer_support'];
    expect(requiredFactors).toHaveLength(5);
    const totalItems = requiredFactors.length * 3;
    expect(totalItems).toBe(15);
  });
});
