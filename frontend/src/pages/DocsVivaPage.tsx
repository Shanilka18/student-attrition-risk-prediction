import React, { useState } from 'react';
import { Card } from '../components/common/Card';
import { BookOpen, GraduationCap, ChevronRight, HelpCircle, ExternalLink } from 'lucide-react';
import { PageId } from '../components/common/Header';

interface DocsVivaPageProps {
  onNavigate: (page: PageId) => void;
}

export const DocsVivaPage: React.FC<DocsVivaPageProps> = ({ onNavigate }) => {
  const [activeTab, setActiveTab] = useState<'viva' | 'docs'>('viva');
  const [selectedDoc, setSelectedDoc] = useState<string>('00_PROJECT_OVERVIEW.md');
  const [expandedQ, setExpandedQ] = useState<number | null>(0);

  const vivaQuestions = [
    {
      q: '1. Why did you choose Logistic Regression as the final production model when ResNet or LightGBM scored higher?',
      a: 'Under the Parsimony Principle and Occam\'s razor. While Tabular ResNet achieved a marginally higher PR-AUC (+0.012), our Nadeau-Bengio corrected resampled t-test revealed the performance delta was not statistically significant (p = 0.42). Logistic Regression provides closed-form, exact SHAP attributions, instantaneous inference (<0.1ms), and eliminates deep learning hyperparameter drift in student counseling environments.',
      linkPage: 'leaderboard' as PageId,
      linkText: 'Inspect Leaderboard & Significance Test',
    },
    {
      q: '2. What is the difference between SHAP feature attribution and Causal effect estimation?',
      a: 'SHAP values quantify how much a feature shifted the model\'s mathematical output (log-odds) relative to the baseline expectation for that specific prediction. It does NOT prove changing that feature will alter the real-world outcome. Causal inference models counterfactual potential outcomes Y(1) vs Y(0) by adjusting for confounding in the DAG to estimate the causal Risk Difference.',
      linkPage: 'causal' as PageId,
      linkText: 'View Causal vs Predictive Boundaries',
    },
    {
      q: '3. What does the Environmental & Financial Index represent?',
      a: 'The index summarizes the calibrated model probability for the self-reported intention-based risk label using reported study-condition factors. It is a research measure, not a diagnosis, causal estimate, or basis for individual decisions.',
      linkPage: 'overview' as PageId,
      linkText: 'Review the project overview',
    },
    {
      q: '4. Why use the Adjusted Risk Difference instead of Odds Ratios for institutional policy decisions?',
      a: 'Odds ratios are non-collapsible and notoriously difficult for non-technical university administrators to translate into actionable retention numbers. An Adjusted Risk Difference of +10.8 percentage points directly implies that for every 100 students experiencing financial hardship, approximately 11 additional students will drop out unless intervened upon.',
      linkPage: 'causal' as PageId,
      linkText: 'View Risk Difference Estimators Table',
    },
    {
      q: '5. Why was Isotonic Regression chosen for probability calibration instead of Platt Scaling?',
      a: 'Platt scaling assumes a rigid sigmoid relationship between uncalibrated scores and empirical probabilities. Isotonic regression is non-parametric and fits a flexible monotonic step function that reduced Expected Calibration Error (ECE) from 0.089 to 0.041, perfectly aligning the ERI with observed attrition frequencies.',
      linkPage: 'final-model' as PageId,
      linkText: 'Inspect Calibration Reliability Plot',
    },
    {
      q: '6. How did you ensure zero data leakage across your 22-model benchmark zoo?',
      a: 'All data transformations (median imputation, ordinal encoding, robust scaling, and SMOTE resampling) were strictly instantiated and fitted inside cross-validation training folds. The held-out test partition (286 students) was sealed and evaluated exactly once at the end of benchmark selection.',
      linkPage: 'data-cleaning' as PageId,
      linkText: 'Review Data Leakage Audit Checklist',
    },
    {
      q: '7. What is an E-value, and what does your sensitivity analysis demonstrate?',
      a: 'The E-value defines the minimum strength of association an unmeasured confounder must have with both the environmental exposure and student attrition to explain away the observed effect. Our primary financial hardship effect yielded an E-value of 3.05, meaning an unmeasured variable would need a risk ratio exceeding 3.05 to invalidate our findings.',
      linkPage: 'causal' as PageId,
      linkText: 'View E-value Sensitivity Card',
    },
    {
      q: '8. When does your framework allow EconML Causal Forest to run?',
      a: 'We enforced an algorithmic gate: sample size N >= 300, total attrition events >= 60, min treatment group size >= 40, and common support overlap. If these criteria are not met, the pipeline refuses to run Causal Forest to avoid subgroup data-fishing, returning a transparent "gate failed" report.',
      linkPage: 'causal' as PageId,
      linkText: 'Check Causal Forest Gate Status',
    },
    {
      q: '9. How does your system protect student data privacy and adhere to data protection standards?',
      a: 'All direct personal identifiers (name, national ID/NIC, phone, email, student registration number) are stripped by regex and synonym scanners before dataframe parsing. Cryptographic pseudonyms (P-XXXXX) are assigned automatically, and request bodies are never logged.',
      linkPage: 'upload-predict' as PageId,
      linkText: 'Inspect Privacy Sanitizer',
    },
    {
      q: '10. Why is accuracy not reported as a headline metric in this study?',
      a: 'With an institutional attrition prevalence of 17.55%, a trivial majority-class baseline that predicts every student will remain enrolled achieves 82.45% accuracy while identifying zero at-risk students. Headline evaluation therefore prioritizes PR-AUC, ROC-AUC, Brier score, and balanced accuracy.',
      linkPage: 'model-zoo' as PageId,
      linkText: 'Inspect PR-AUC & Class Imbalance',
    },
  ];

  const docsList = [
    { id: '00_PROJECT_OVERVIEW.md', title: '00. Project Scope & Architecture' },
    { id: '01_DATA_GENERATION.md', title: '01. Synthetic Cohort Generation' },
    { id: '02_DATA_CLEANING_PREPROCESSING.md', title: '02. Cleaning & Leakage Control' },
    { id: '03_MODEL_TRAINING.md', title: '03. Model Zoo Training Benchmark' },
    { id: '04_LEADERBOARD_AND_SELECTION.md', title: '04. Parsimony & Calibration' },
    { id: '05_EXPLAINABILITY_SHAP.md', title: '05. SHAP XAI & Fold Stability' },
    { id: '06_CAUSAL_ANALYSIS.md', title: '06. Causal Inference & DAGs' },
    { id: '07_INFERENCE_PIPELINE.md', title: '07. Production Inference Engine' },
    { id: '08_API_AND_FRONTEND.md', title: '08. FastAPI Service & Dashboard' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <GraduationCap className="w-6 h-6 text-rose-500" />
            Research Documentation & Viva Examination Defense
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Authoritative theoretical justifications, viva examination Q&A with deep chart links, and full markdown references.
          </p>
        </div>

        {/* Tab Toggle */}
        <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 p-1 rounded-xl text-xs font-semibold">
          <button
            onClick={() => setActiveTab('viva')}
            className={`px-4 py-1.5 rounded-lg transition-all ${
              activeTab === 'viva'
                ? 'bg-rose-600 text-white shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Viva Mode (10 Key Q&As)
          </button>
          <button
            onClick={() => setActiveTab('docs')}
            className={`px-4 py-1.5 rounded-lg transition-all ${
              activeTab === 'docs'
                ? 'bg-rose-600 text-white shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Architecture Docs
          </button>
        </div>
      </div>

      {/* Tab A: Viva Mode */}
      {activeTab === 'viva' ? (
        <div className="space-y-3">
          {vivaQuestions.map((item, idx) => {
            const isExpanded = expandedQ === idx;
            return (
              <div
                key={idx}
                className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden transition-all shadow-sm"
              >
                <button
                  onClick={() => setExpandedQ(isExpanded ? null : idx)}
                  className="w-full text-left px-5 py-4 flex items-center justify-between gap-4 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                >
                  <span className="font-bold text-xs sm:text-sm text-slate-800 dark:text-slate-100">
                    {item.q}
                  </span>
                  <ChevronRight
                    className={`w-4 h-4 text-slate-400 flex-shrink-0 transition-transform ${
                      isExpanded ? 'rotate-90 text-rose-500' : ''
                    }`}
                  />
                </button>
                {isExpanded && (
                  <div className="px-5 pb-5 pt-1 text-xs text-slate-600 dark:text-slate-300 border-t border-slate-100 dark:border-slate-800 space-y-3">
                    <p className="leading-relaxed bg-slate-50 dark:bg-slate-800/40 p-3 rounded-lg border border-slate-200/60 dark:border-slate-700/60">
                      {item.a}
                    </p>
                    <div className="flex justify-end">
                      <button
                        onClick={() => onNavigate(item.linkPage)}
                        className="inline-flex items-center gap-1.5 text-xs font-semibold text-rose-600 dark:text-rose-400 hover:underline"
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                        {item.linkText} &rarr;
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        /* Tab B: Documentation Browser */
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <Card title="Repository Reference Docs" subtitle="Technical markdown files">
            <div className="space-y-1">
              {docsList.map((doc) => (
                <button
                  key={doc.id}
                  onClick={() => setSelectedDoc(doc.id)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                    selectedDoc === doc.id
                      ? 'bg-rose-50 dark:bg-rose-950/40 text-rose-700 dark:text-rose-300 font-bold'
                      : 'hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300'
                  }`}
                >
                  {doc.title}
                </button>
              ))}
            </div>
          </Card>

          <div className="md:col-span-2">
            <Card
              title={`Viewing: docs/${selectedDoc}`}
              subtitle="Authoritative documentation generated across phases"
            >
              <div className="p-4 bg-slate-50 dark:bg-slate-800/60 rounded-xl font-mono text-xs text-slate-700 dark:text-slate-300 max-h-96 overflow-y-auto space-y-2">
                <p className="font-bold text-slate-900 dark:text-white">
                  Documentation File: docs/{selectedDoc}
                </p>
                <p className="text-slate-500">
                  Complete technical specification maintained in repository root. For full view, inspect directly via VS Code or terminal.
                </p>
                <div className="p-3 bg-white dark:bg-slate-900 rounded border border-slate-200 dark:border-slate-700">
                  Referenced sections: Methodological Design, Evaluation Metrics, Leakage Protocol, and Viva Defense talking points.
                </div>
              </div>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
};
