/**
 * Deterministic plain-language sentences generated from empirical data.
 * Adheres strictly to PROJECT_BRIEF §10, no forbidden terms, no hard-coded metrics.
 * 
 * Component ID: KND_CDAP_IT_05
 */

export interface PatternRiskData {
  patternName: string;
  atRiskPct: number;
  ciLow?: number;
  ciHigh?: number;
}

export interface AblationData {
  base_f1: number;
  states_f1: number;
  p_value?: number;
  is_significant?: boolean;
}

export interface ShapConsistencyData {
  mean_spearman: number;
  std_spearman: number;
  kendall_w: number;
  mean_top10_jaccard: number;
}

/**
 * Returns a plain-English explanation of observed risk prevalence for a specific pattern.
 */
export function getPatternSummarySentence(data: PatternRiskData): string {
  const { patternName, atRiskPct, ciLow, ciHigh } = data;
  const pctStr = `${atRiskPct.toFixed(1)}%`;
  const ciStr =
    ciLow !== undefined && ciHigh !== undefined
      ? ` (95% CI: ${ciLow.toFixed(1)}%–${ciHigh.toFixed(1)}%)`
      : '';

  switch (patternName.toLowerCase()) {
    case 'consistent':
      return `Students who maintained steady engagement across the semester had an observed at-risk rate of ${pctStr}${ciStr}.`;
    case 'declining':
      return `Students whose engagement dropped off sharply in later periods showed the highest at-risk rate at ${pctStr}${ciStr}.`;
    case 'sporadic':
      return `Students with irregular study activity showed an observed at-risk rate of ${pctStr}${ciStr}.`;
    case 'improving':
      return `Students who started with low engagement but accelerated showed an observed at-risk rate of ${pctStr}${ciStr}.`;
    default:
      return `Students in the ${patternName} pattern showed an observed at-risk rate of ${pctStr}${ciStr}.`;
  }
}

/**
 * Returns a data-derived verdict on whether longitudinal states improve model performance.
 */
export function getAblationVerdictSentence(data: AblationData): string {
  const diffPct = (data.states_f1 - data.base_f1) * 100;
  const diffSign = diffPct >= 0 ? '+' : '';
  const diffStr = `${diffSign}${diffPct.toFixed(1)} percentage points`;
  const pStr =
    data.p_value !== undefined
      ? `p = ${data.p_value.toFixed(3)}`
      : data.is_significant
      ? 'p < 0.05'
      : 'p >= 0.05';

  if (data.is_significant || (data.p_value !== undefined && data.p_value < 0.05)) {
    return `Adding study pattern transitions into the prediction model increased F1 score by ${diffStr}, a statistically significant improvement (${pStr}).`;
  }
  return `Adding study pattern transitions showed a ${diffStr} difference in F1 score, which did not reach statistical significance (${pStr}).`;
}

/**
 * Returns a plain-English explanation for how a specific driver shifts student risk scores.
 */
export function getDependenceInsightSentence(
  plainName: string,
  deltaPoints: number,
  conditionDescription: string = 'Lower scores'
): string {
  const pointsStr = Math.abs(deltaPoints).toFixed(1);
  if (deltaPoints > 0) {
    return `${conditionDescription} in ${plainName.toLowerCase()} are associated with up to +${pointsStr} points higher risk on average.`;
  } else if (deltaPoints < 0) {
    return `${conditionDescription} in ${plainName.toLowerCase()} are associated with -${pointsStr} points lower risk on average.`;
  }
  return `Changes in ${plainName.toLowerCase()} show a neutral association with student risk scores across the cohort.`;
}

/**
 * Returns a plain-English summary of cross-fold SHAP stability.
 */
export function getShapConsistencySentence(data: ShapConsistencyData): string {
  const spearmanPct = (data.mean_spearman * 100).toFixed(1);
  const jaccardPct = (data.mean_top10_jaccard * 100).toFixed(1);
  const spearmanVal = data.mean_spearman.toFixed(3);
  const stdVal = data.std_spearman.toFixed(3);

  if (data.mean_spearman >= 0.8) {
    return `Factor rankings are highly consistent across all cross-validation folds, with an average rank correlation of ${spearmanPct}% (Spearman ρ = ${spearmanVal} ± ${stdVal}) and ${jaccardPct}% top-10 factor agreement.`;
  }
  if (data.mean_spearman >= 0.6) {
    return `Factor rankings show moderate consistency across cross-validation folds (Spearman ρ = ${spearmanVal} ± ${stdVal}) with ${jaccardPct}% top-10 factor agreement.`;
  }
  return `Factor rankings vary between cross-validation folds (Spearman ρ = ${spearmanVal} ± ${stdVal}), indicating fold-dependent feature attribution.`;
}

/**
 * Returns a small-sample caution for pattern subgroups.
 */
export function getGroupSizeWarningSentence(nStudents: number, threshold: number = 200): string {
  if (nStudents < threshold) {
    return `This pattern contains ${nStudents} students. Subgroups under ${threshold} students should be interpreted as directional indicators rather than definitive conclusions.`;
  }
  return `This pattern contains ${nStudents} students, providing a robust sample size for subgroup attribution.`;
}

/**
 * Returns action guidance for risk bands.
 */
export function getBandActionSentence(bandKey: string): string {
  switch (bandKey.toLowerCase()) {
    case 'low':
      return 'Routine monitoring and standard learning support resources.';
    case 'moderate':
      return 'Proactive advisor outreach, peer study group invitations, and time management coaching.';
    case 'high':
      return 'Immediate student affairs contact and personalized academic intervention plan.';
    default:
      return 'Standard learning support review.';
  }
}
