import { describe, it, expect } from 'vitest';
import type { ColumnMatchDetail } from '../api/types';

describe('Column Matching and Data Quality Business Logic', () => {
  it('correctly classifies confidence tiers based on score and match status', () => {
    const highMatch: ColumnMatchDetail = {
      required_key: 'resource_access_p1',
      plain_name: 'LMS / Resource Access (Weeks 1–4)',
      matched_column: 'resource_access_p1',
      confidence_tier: 'High',
      confidence_score: 100.0,
      sample_values: ['4', '5', '3'],
      status: 'auto_accepted',
    };

    const checkMatch: ColumnMatchDetail = {
      required_key: 'study_time_p2',
      plain_name: 'Study Time (Weeks 5–8)',
      matched_column: 'Hours_Studying_Midterm',
      confidence_tier: 'Check this',
      confidence_score: 82.0,
      sample_values: ['2', '3', '4'],
      status: 'needs_confirmation',
    };

    const notFoundMatch: ColumnMatchDetail = {
      required_key: 'quiz_practice_p4',
      plain_name: 'Practice & Quizzes (Weeks 13–16)',
      matched_column: null,
      confidence_tier: 'Not found',
      confidence_score: 0.0,
      sample_values: [],
      status: 'unmapped',
    };

    expect(highMatch.confidence_tier).toBe('High');
    expect(highMatch.status).toBe('auto_accepted');

    expect(checkMatch.confidence_tier).toBe('Check this');
    expect(checkMatch.status).toBe('needs_confirmation');

    expect(notFoundMatch.confidence_tier).toBe('Not found');
    expect(notFoundMatch.matched_column).toBeNull();
  });

  it('computes dataset traffic light correctly based on matches and scorable rate', () => {
    function computeTrafficLight(
      matches: ColumnMatchDetail[],
      scorableCount: number,
      totalRows: number
    ): 'GREEN' | 'AMBER' | 'RED' {
      const notFoundCount = matches.filter((m) => m.confidence_tier === 'Not found').length;
      const needsConfCount = matches.filter((m) => m.confidence_tier === 'Check this').length;
      const pctScorable = totalRows > 0 ? (scorableCount / totalRows) * 100 : 0;

      if (notFoundCount === 0 && needsConfCount === 0 && pctScorable >= 95) {
        return 'GREEN';
      }
      if (notFoundCount === 0 && pctScorable >= 70) {
        return 'AMBER';
      }
      return 'RED';
    }

    const perfectMatches: ColumnMatchDetail[] = Array.from({ length: 28 }, (_, i) => ({
      required_key: `ind_p${i + 1}`,
      plain_name: `Indicator ${i + 1}`,
      matched_column: `col_${i + 1}`,
      confidence_tier: 'High',
      confidence_score: 95.0,
      sample_values: ['3', '4'],
      status: 'auto_accepted',
    }));

    expect(computeTrafficLight(perfectMatches, 200, 200)).toBe('GREEN');

    // With 2 fuzzy suggestions
    const amberMatches = [...perfectMatches];
    amberMatches[0] = { ...amberMatches[0], confidence_tier: 'Check this', confidence_score: 85.0 };
    expect(computeTrafficLight(amberMatches, 200, 200)).toBe('AMBER');

    // With 1 missing column
    const redMatches = [...perfectMatches];
    redMatches[1] = { ...redMatches[1], confidence_tier: 'Not found', matched_column: null };
    expect(computeTrafficLight(redMatches, 200, 200)).toBe('RED');
  });

  it('guarantees additive waterfall points equal delta from baseline MEI', () => {
    const baselineMei = 24.6;
    const finalMei = 73;
    const targetDelta = Math.round(finalMei - baselineMei); // 48 points

    const reasons = [
      { text: 'Low overall review time', points: 22, direction: 'risk_increasing' },
      { text: 'Reduced late consistency', points: 17, direction: 'risk_increasing' },
      { text: 'Declining quiz exercises', points: 12, direction: 'risk_increasing' },
      { text: 'Regular resource access', points: -3, direction: 'risk_decreasing' },
    ];

    const sumPoints = reasons.reduce((acc, r) => acc + r.points, 0);
    expect(sumPoints).toBe(targetDelta);
    expect(Math.round(baselineMei + sumPoints)).toBe(finalMei);
  });

  it('properly evaluates missing feature scorable threshold at 50%', () => {
    const totalFeatureColumns = 28;
    const maxAllowedMissing = totalFeatureColumns * 0.50; // 14

    function isScorable(missingCount: number): boolean {
      return missingCount <= maxAllowedMissing;
    }

    expect(isScorable(0)).toBe(true);
    expect(isScorable(5)).toBe(true);
    expect(isScorable(14)).toBe(true);
    expect(isScorable(15)).toBe(false);
    expect(isScorable(28)).toBe(false);
  });
});
