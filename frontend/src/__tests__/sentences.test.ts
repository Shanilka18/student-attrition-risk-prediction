import { describe, it, expect } from 'vitest';
import {
  getPatternSummarySentence,
  getAblationVerdictSentence,
  getDependenceInsightSentence,
  getShapConsistencySentence,
  getGroupSizeWarningSentence,
  getBandActionSentence,
} from '../content/sentences';

describe('Data Storytelling Sentence Templates', () => {
  it('generates pattern summary sentences with CI correctly', () => {
    const s1 = getPatternSummarySentence({
      patternName: 'Consistent',
      atRiskPct: 19.4,
      ciLow: 16.8,
      ciHigh: 22.2,
    });
    expect(s1).toContain('steady engagement');
    expect(s1).toContain('19.4%');
    expect(s1).toContain('16.8%–22.2%');

    const s2 = getPatternSummarySentence({
      patternName: 'Declining',
      atRiskPct: 49.5,
      ciLow: 42.8,
      ciHigh: 56.3,
    });
    expect(s2).toContain('dropped off sharply');
    expect(s2).toContain('49.5%');

    const s3 = getPatternSummarySentence({
      patternName: 'Sporadic',
      atRiskPct: 27.6,
    });
    expect(s3).toContain('irregular study activity');
    expect(s3).toContain('27.6%');
    expect(s3).not.toContain('CI:');
  });

  it('generates ablation verdicts based on statistical significance', () => {
    const sig = getAblationVerdictSentence({
      base_f1: 0.732,
      states_f1: 0.785,
      p_value: 0.041,
      is_significant: true,
    });
    expect(sig).toContain('+5.3 percentage points');
    expect(sig).toContain('statistically significant improvement');
    expect(sig).toContain('p = 0.041');

    const nonSig = getAblationVerdictSentence({
      base_f1: 0.75,
      states_f1: 0.755,
      p_value: 0.21,
      is_significant: false,
    });
    expect(nonSig).toContain('+0.5 percentage points');
    expect(nonSig).toContain('did not reach statistical significance');
    expect(nonSig).toContain('p = 0.210');
  });

  it('generates dependence insight sentences accurately', () => {
    const up = getDependenceInsightSentence('Self-Study Hours', 8.4, 'Fewer study hours');
    expect(up).toBe('Fewer study hours in self-study hours are associated with up to +8.4 points higher risk on average.');

    const down = getDependenceInsightSentence('Attendance Rate', -6.2, 'Higher attendance');
    expect(down).toBe('Higher attendance in attendance rate are associated with -6.2 points lower risk on average.');

    const neutral = getDependenceInsightSentence('Peer Collaboration', 0);
    expect(neutral).toContain('neutral association');
  });

  it('generates SHAP consistency descriptions', () => {
    const high = getShapConsistencySentence({
      mean_spearman: 0.884,
      std_spearman: 0.031,
      kendall_w: 0.892,
      mean_top10_jaccard: 0.84,
    });
    expect(high).toContain('highly consistent');
    expect(high).toContain('88.4%');
    expect(high).toContain('0.884 ± 0.031');
    expect(high).toContain('84.0% top-10');

    const mod = getShapConsistencySentence({
      mean_spearman: 0.65,
      std_spearman: 0.05,
      kendall_w: 0.68,
      mean_top10_jaccard: 0.7,
    });
    expect(mod).toContain('moderate consistency');
  });

  it('generates sample size warnings for subgroups', () => {
    const small = getGroupSizeWarningSentence(117, 200);
    expect(small).toContain('directional indicators');
    expect(small).toContain('117 students');

    const large = getGroupSizeWarningSentence(834, 200);
    expect(large).toContain('robust sample size');
  });

  it('returns band action guidance', () => {
    expect(getBandActionSentence('low')).toContain('Routine monitoring');
    expect(getBandActionSentence('moderate')).toContain('Proactive advisor outreach');
    expect(getBandActionSentence('high')).toContain('Immediate student affairs');
  });
});
