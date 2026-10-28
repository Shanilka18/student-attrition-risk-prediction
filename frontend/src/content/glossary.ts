/**
 * Plain-English glossary for university lecturers, advisors, and administrators.
 * Translates technical machine learning concepts into clear, accessible language.
 */

export interface GlossaryEntry {
  key: string;
  plainLabel: string;
  formalName: string;
  summary: string;
  unit?: string;
  ideal?: 'higher' | 'lower' | 'target';
}

export const GLOSSARY: Record<string, GlossaryEntry> = {
  mei: {
    key: 'mei',
    plainLabel: 'Engagement risk score',
    formalName: 'Multimodal Engagement Index (MEI)',
    summary: 'A continuous scale from 0 to 100 estimating how likely a student is to consider leaving their course, based on changes in their study routines.',
    unit: 'points (0–100)',
    ideal: 'lower',
  },
  band: {
    key: 'band',
    plainLabel: 'Risk tier',
    formalName: 'Risk classification band',
    summary: 'A triage category (Low, Moderate, or High) assigned to each student to help advisors prioritize outreach without stigmatizing individuals.',
  },
  pattern: {
    key: 'pattern',
    plainLabel: 'Study routine trend',
    formalName: 'Longitudinal engagement trajectory',
    summary: 'The overall direction of a student’s participation across all four university periods (Consistent, Declining, Recovering, or Variable).',
  },
  level: {
    key: 'level',
    plainLabel: 'Overall engagement level',
    formalName: 'Mean engagement tier',
    summary: 'The student’s average participation level across the semester (High, Medium, or Low).',
  },
  f1: {
    key: 'f1',
    plainLabel: 'At-risk balance score',
    formalName: 'F1 score (positive class)',
    summary: 'Measures how well the model balances finding students who genuinely need help without raising too many false alarms.',
    unit: '0.00–1.00',
    ideal: 'higher',
  },
  auc: {
    key: 'auc',
    plainLabel: 'Ranking accuracy',
    formalName: 'Area under the ROC curve (AUC-ROC)',
    summary: 'The likelihood that a randomly chosen student who needs help receives a higher risk score than a student who is thriving.',
    unit: '0.00–1.00',
    ideal: 'higher',
  },
  mcc: {
    key: 'mcc',
    plainLabel: 'Overall prediction agreement',
    formalName: 'Matthews correlation coefficient (MCC)',
    summary: 'A robust measure of agreement between predictions and reality that remains reliable even when at-risk students are a small minority.',
    unit: '-1.00 to +1.00',
    ideal: 'higher',
  },
  balanced_accuracy: {
    key: 'balanced_accuracy',
    plainLabel: 'Fair-share accuracy',
    formalName: 'Balanced accuracy',
    summary: 'The average accuracy across both at-risk students and safe students, preventing the model from ignoring the smaller group.',
    unit: '0.00–1.00',
    ideal: 'higher',
  },
  calibration: {
    key: 'calibration',
    plainLabel: 'Probability reliability',
    formalName: 'Expected calibration error (ECE)',
    summary: 'Checks whether a predicted 70% risk score really means that 7 out of 10 students with that score face academic difficulty.',
    ideal: 'lower',
  },
  generalisation_gap: {
    key: 'generalisation_gap',
    plainLabel: 'Over-practice gap',
    formalName: 'Generalisation gap (train minus validation score)',
    summary: 'The drop in score between familiar training examples and unseen students; small gaps indicate the model will perform reliably on future classes.',
    unit: 'difference',
    ideal: 'lower',
  },
  r2: {
    key: 'r2',
    plainLabel: 'Survey variance explained',
    formalName: 'Coefficient of determination (R²)',
    summary: 'The share of differences in students’ continuous attrition intentions that our engagement questionnaire successfully accounts for.',
    unit: '0.00–1.00',
    ideal: 'higher',
  },
  rmse: {
    key: 'rmse',
    plainLabel: 'Typical score error',
    formalName: 'Root mean squared error (RMSE)',
    summary: 'The typical difference between the predicted 0–100 score and the student’s actual self-reported intent score, penalising large mistakes.',
    unit: 'points (0–100)',
    ideal: 'lower',
  },
  mae: {
    key: 'mae',
    plainLabel: 'Average points off',
    formalName: 'Mean absolute error (MAE)',
    summary: 'The average number of points by which the model’s continuous score differs from the student’s self-reported intent.',
    unit: 'points (0–100)',
    ideal: 'lower',
  },
  shap: {
    key: 'shap',
    plainLabel: 'Feature contribution points',
    formalName: 'Shapley additive explanations (SHAP)',
    summary: 'A fair mathematical method that divides a student’s final score into the exact points contributed by each survey answer.',
    unit: 'points',
  },
  hmm_state: {
    key: 'hmm_state',
    plainLabel: 'Study habit phase',
    formalName: 'Hidden Markov Model state',
    summary: 'An underlying mindset or study mode (such as Exemplary Dedication or Critical Disengagement) identified from participation answers.',
  },
  optimal_threshold: {
    key: 'optimal_threshold',
    plainLabel: 'Action cutoff',
    formalName: 'Tuned classification threshold',
    summary: 'The exact probability cutoff chosen on validation data to decide when an advisory outreach flag should be raised.',
    unit: '0.00–1.00',
  },
};

/**
 * Retrieve glossary entry with fallback.
 */
export function getGlossaryEntry(key: string): GlossaryEntry {
  const normKey = key.toLowerCase().replace(/[^a-z0-9_]/g, '_');
  if (GLOSSARY[normKey]) {
    return GLOSSARY[normKey];
  }
  return {
    key: normKey,
    plainLabel: key.replace(/_/g, ' '),
    formalName: key.replace(/_/g, ' '),
    summary: `Technical indicator for ${key.replace(/_/g, ' ')}.`,
  };
}
