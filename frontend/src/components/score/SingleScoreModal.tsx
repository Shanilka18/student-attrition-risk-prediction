import { useState } from 'react';
import { X, TrendingDown, TrendingUp, ArrowRight, ShieldCheck } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { useScoreSingleStudent } from '@/api/queries';
import type { SingleScoreResponse } from '@/api/types';

interface SingleScoreModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const QUESTIONS = [
  {
    id: 'resource_access',
    title: 'LMS / Resource Access',
    question: 'How often did you access course materials and online resources?',
    options: ['0 days', '1–2 days', '3–4 days', '5–6 days', 'Every day'],
  },
  {
    id: 'study_time',
    title: 'Study Time',
    question: 'How many hours per week did you dedicate to studying course material?',
    options: ['<2 hours', '2–5 hours', '6–10 hours', '11–15 hours', '>15 hours'],
  },
  {
    id: 'review_time',
    title: 'Review Time',
    question: 'How many hours per week did you spend reviewing lecture notes and recordings?',
    options: ['<1 hour', '1–2 hours', '3–4 hours', '5–6 hours', '>6 hours'],
  },
  {
    id: 'study_consistency',
    title: 'Study Consistency',
    question: 'How consistent was your study schedule throughout this period?',
    options: ['Not consistent', 'Slightly consistent', 'Moderately consistent', 'Very consistent', 'Extremely consistent'],
  },
  {
    id: 'participation',
    title: 'Class Participation',
    question: 'How actively did you participate in class discussions, forums, and Q&A?',
    options: ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],
  },
  {
    id: 'assignment_engagement',
    title: 'Assignment Engagement',
    question: 'When did you typically begin working on course assignments?',
    options: ['Last minute', 'A few days before', 'About a week before', 'Early in period', 'Well in advance'],
  },
  {
    id: 'quiz_practice',
    title: 'Practice & Quizzes',
    question: 'How many practice quizzes or self-assessment exercises did you attempt?',
    options: ['0 quizzes', '1–2 quizzes', '3–4 quizzes', '5 quizzes', '>5 quizzes'],
  },
];

const PERIODS = [
  { p: 1, label: 'Weeks 1–4' },
  { p: 2, label: 'Weeks 5–8' },
  { p: 3, label: 'Weeks 9–12' },
  { p: 4, label: 'Weeks 13–16' },
];

export function SingleScoreModal({ isOpen, onClose }: SingleScoreModalProps) {
  const [studentId, setStudentId] = useState('STUDENT_SAMPLE_01');
  const [answers, setAnswers] = useState<Record<string, number>>(() => {
    const init: Record<string, number> = {};
    for (const q of QUESTIONS) {
      for (const p of PERIODS) {
        init[`${q.id}_p${p.p}`] = 4; // default high engagement
      }
    }
    return init;
  });

  const [activePeriod, setActivePeriod] = useState<number>(1);
  const [result, setResult] = useState<SingleScoreResponse | null>(null);

  const scoreMutation = useScoreSingleStudent();

  if (!isOpen) return null;

  const handleSelect = (questionId: string, period: number, val: number) => {
    setAnswers((prev) => ({
      ...prev,
      [`${questionId}_p${period}`]: val,
    }));
  };

  const handleApplyPreset = (type: 'high' | 'declining' | 'disengaged') => {
    const newAnswers: Record<string, number> = {};
    for (const q of QUESTIONS) {
      for (const { p } of PERIODS) {
        if (type === 'high') {
          newAnswers[`${q.id}_p${p}`] = 4 + (p % 2 === 0 ? 1 : 0);
        } else if (type === 'declining') {
          // Drops from 4 down to 1
          newAnswers[`${q.id}_p${p}`] = Math.max(1, 5 - p);
        } else {
          // Low across board
          newAnswers[`${q.id}_p${p}`] = 1;
        }
      }
    }
    setAnswers(newAnswers);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await scoreMutation.mutateAsync({
        student_id: studentId.trim() || 'STUDENT_INLINE',
        answers,
      });
      setResult(res);
    } catch (err: any) {
      alert(`Scoring failed: ${err.message}`);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink/40 backdrop-blur-none">
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-surface rounded-lg border border-border shadow-md overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-paper">
          <div>
            <h3 className="font-serif text-section font-semibold text-ink">
              Score an Individual Student
            </h3>
            <p className="text-small text-muted mt-0.5">
              Enter survey ratings across the 4 semester periods to view the calculated Engagement Index and waterfall driver reasons.
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-muted hover:text-ink rounded-md transition-colors"
            aria-label="Close modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body Container */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {!result ? (
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Presets and Identifier Bar */}
              <div className="flex flex-wrap items-center justify-between gap-4 p-4 bg-paper rounded-md border border-border">
                <div className="flex items-center gap-2">
                  <label htmlFor="student-id-input" className="text-caption font-semibold text-ink">
                    Student ID:
                  </label>
                  <input
                    id="student-id-input"
                    type="text"
                    value={studentId}
                    onChange={(e) => setStudentId(e.target.value)}
                    className="px-3 py-1.5 text-caption bg-surface border border-border rounded-md font-mono focus:outline-none focus:border-forest"
                  />
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-caption text-muted font-medium">Quick Profiles:</span>
                  <button
                    type="button"
                    onClick={() => handleApplyPreset('high')}
                    className="px-2.5 py-1 text-small bg-forest-subtle text-forest hover:bg-forest/15 rounded border border-forest/30 font-medium"
                  >
                    High Engagement
                  </button>
                  <button
                    type="button"
                    onClick={() => handleApplyPreset('declining')}
                    className="px-2.5 py-1 text-small bg-amber-subtle text-amber-text hover:bg-amber/15 rounded border border-amber/30 font-medium"
                  >
                    Declining Pace
                  </button>
                  <button
                    type="button"
                    onClick={() => handleApplyPreset('disengaged')}
                    className="px-2.5 py-1 text-small bg-crimson-subtle text-crimson hover:bg-crimson/15 rounded border border-crimson/30 font-medium"
                  >
                    High Risk
                  </button>
                </div>
              </div>

              {/* Period Tabs */}
              <div className="flex border-b border-border">
                {PERIODS.map(({ p, label }) => {
                  const isActive = activePeriod === p;
                  return (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setActivePeriod(p)}
                      className={`flex-1 py-2.5 text-caption font-medium text-center border-b-2 transition-colors ${
                        isActive
                          ? 'border-forest text-forest font-semibold bg-forest-subtle/30'
                          : 'border-transparent text-muted hover:text-ink hover:border-border-strong'
                      }`}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>

              {/* Questionnaire Grid for Active Period */}
              <div className="space-y-4">
                {QUESTIONS.map((q) => {
                  const currentVal = answers[`${q.id}_p${activePeriod}`] || 3;
                  return (
                    <div
                      key={q.id}
                      className="p-4 rounded-md border border-border bg-surface hover:border-border-strong transition-colors"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                        <span className="text-caption font-semibold text-ink">{q.title}</span>
                        <span className="text-small text-muted">{q.question}</span>
                      </div>

                      {/* 5-Option Radio Row */}
                      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                        {q.options.map((optText, optIdx) => {
                          const ratingVal = optIdx + 1;
                          const isSelected = currentVal === ratingVal;
                          return (
                            <button
                              key={ratingVal}
                              type="button"
                              onClick={() => handleSelect(q.id, activePeriod, ratingVal)}
                              className={`px-3 py-2 text-small rounded border text-left transition-colors flex items-center justify-between ${
                                isSelected
                                  ? 'border-forest bg-forest-subtle text-forest font-semibold'
                                  : 'border-border bg-paper hover:bg-surface text-ink'
                              }`}
                            >
                              <span className="truncate pr-1">{optText}</span>
                              <span className="text-[11px] font-mono text-muted shrink-0">
                                ({ratingVal})
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="flex items-center justify-end gap-3 pt-4 border-t border-border">
                <Button variant="secondary" onClick={onClose} type="button">
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  type="submit"
                  disabled={scoreMutation.isPending}
                  icon={<ArrowRight className="w-4 h-4" />}
                >
                  {scoreMutation.isPending ? 'Calculating...' : 'Calculate Engagement Index'}
                </Button>
              </div>
            </form>
          ) : (
            /* Results View with Waterfall */
            <div className="space-y-6">
              {/* Score Headline */}
              <div className="p-6 bg-paper border border-border rounded-lg flex flex-wrap items-center justify-between gap-6">
                <div>
                  <div className="text-small text-muted font-medium uppercase tracking-wider">
                    Calculated Result for {result.student_id}
                  </div>
                  <div className="flex items-baseline gap-3 mt-1">
                    <span className="font-serif text-hero font-semibold text-ink tabular-nums">
                      {result.mei}
                    </span>
                    <span className="text-body text-muted">/ 100</span>
                    <Badge
                      variant={
                        result.band === 'High'
                          ? 'crimson'
                          : result.band === 'Moderate'
                          ? 'amber'
                          : 'forest'
                      }
                      size="md"
                    >
                      {result.band} Risk
                    </Badge>
                    <Badge variant={result.flag === 'Yes' ? 'crimson' : 'neutral'} size="md">
                      {result.flag === 'Yes' ? 'At-Risk Flagged' : 'Normal Engagement'}
                    </Badge>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-caption border-l border-border pl-6">
                  <div>
                    <span className="text-muted block">Study Pattern:</span>
                    <span className="font-semibold text-ink">{result.pattern}</span>
                  </div>
                  <div>
                    <span className="text-muted block">Engagement Level:</span>
                    <span className="font-semibold text-ink">{result.level}</span>
                  </div>
                </div>
              </div>

              {/* Waterfall Point Breakdown */}
              <Card className="border-border">
                <CardHeader>
                  <CardTitle className="text-section">Waterfall Driver Explanation</CardTitle>
                  <CardDescription>
                    Points attribution starting from the cohort baseline average score of {result.baseline_mei.toFixed(1)}.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {/* Baseline row */}
                  <div className="flex items-center justify-between p-3 bg-paper rounded border border-border text-caption">
                    <span className="font-medium text-ink">Cohort Baseline Average MEI</span>
                    <span className="font-mono font-semibold text-muted">
                      {result.baseline_mei.toFixed(1)} pts
                    </span>
                  </div>

                  {/* Additive Reasons */}
                  {result.reasons.map((r, i) => {
                    const isRisk = r.direction === 'risk_increasing';
                    return (
                      <div
                        key={i}
                        className={`flex items-center justify-between p-3 rounded border text-caption ${
                          isRisk
                            ? 'bg-crimson-subtle/50 border-crimson/20'
                            : 'bg-forest-subtle/50 border-forest/20'
                        }`}
                      >
                        <div className="flex items-center gap-2.5">
                          {isRisk ? (
                            <TrendingUp className="w-4 h-4 text-crimson shrink-0" />
                          ) : (
                            <TrendingDown className="w-4 h-4 text-forest shrink-0" />
                          )}
                          <span className="text-ink">{r.text}</span>
                        </div>
                        <span
                          className={`font-mono font-semibold shrink-0 ${
                            isRisk ? 'text-crimson' : 'text-forest'
                          }`}
                        >
                          {r.points > 0 ? `+${r.points}` : r.points} pts
                        </span>
                      </div>
                    );
                  })}

                  {/* Final Score Row */}
                  <div className="flex items-center justify-between p-3 bg-forest-subtle border border-forest/30 rounded text-caption font-semibold">
                    <span className="text-ink">Final Student Engagement Index</span>
                    <span className="font-mono text-forest text-body">{result.mei} pts</span>
                  </div>
                </CardContent>
              </Card>

              {/* Ethical usage note */}
              <div className="p-4 bg-paper rounded-md border border-border flex items-start gap-3">
                <ShieldCheck className="w-5 h-5 text-forest shrink-0 mt-0.5" />
                <div className="text-small text-muted leading-relaxed">
                  <span className="font-semibold text-ink">What this does not mean:</span> The score describes self-reported study habits. It is not an assessment of a student's intelligence, character, or integrity, and should be used solely as a starting point for supportive pastoral dialogue.
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-between pt-4 border-t border-border">
                <Button variant="secondary" onClick={() => setResult(null)}>
                  Score Another Profile
                </Button>
                <Button variant="primary" onClick={onClose}>
                  Done
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
