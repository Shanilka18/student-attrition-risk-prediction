import { useState } from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Download,
  ArrowRight,
  ArrowLeft,
  Wrench,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { getDownloadProblemsUrl } from '@/api/queries';
import type { ValidateUploadResponse, ColumnMatchDetail } from '@/api/types';

interface ColumnCheckStepProps {
  validation: ValidateUploadResponse;
  onConfirmScoring: (mappingOverrides: Record<string, string>) => void;
  onBack: () => void;
  isScoring?: boolean;
}

export function ColumnCheckStep({
  validation,
  onConfirmScoring,
  onBack,
  isScoring = false,
}: ColumnCheckStepProps) {
  // Track overrides by user: canonical_key -> raw_column
  const [overrides, setOverrides] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    if (validation.column_matches) {
      for (const m of validation.column_matches) {
        if (m.matched_column) {
          init[m.matched_column] = m.required_key;
        }
      }
    }
    return init;
  });

  // Track raw column selection per canonical key
  const [selectedMapping, setSelectedMapping] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    if (validation.column_matches) {
      for (const m of validation.column_matches) {
        if (m.matched_column) {
          init[m.required_key] = m.matched_column;
        }
      }
    }
    return init;
  });

  const handleColumnChange = (requiredKey: string, rawCol: string) => {
    setSelectedMapping((prev) => ({
      ...prev,
      [requiredKey]: rawCol,
    }));

    setOverrides((prev) => {
      const next = { ...prev };
      // Remove any prior mapping for this rawCol
      if (rawCol && rawCol !== '__unmapped__') {
        next[rawCol] = requiredKey;
      }
      return next;
    });
  };

  const matches: ColumnMatchDetail[] = validation.column_matches || [];
  const quality = validation.quality_summary;
  const problems = validation.fixable_problems || [];

  // Traffic light styling
  const isGreen = validation.traffic_light === 'GREEN';
  const isAmber = validation.traffic_light === 'AMBER';

  // Count how many required items are mapped
  const totalRequired = matches.length || 29;
  const mappedCount = Object.values(selectedMapping).filter((v) => v && v !== '__unmapped__').length;
  const canProceed = mappedCount >= 20 && (validation.scorable_rows_count > 0);

  return (
    <div className="space-y-6">
      {/* Step Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="font-serif text-page-title text-ink font-semibold">
            Check Columns & Quality Audit
          </h2>
          <p className="text-body text-muted mt-1">
            Review automatically detected survey questions, verify match confidences, and confirm data remediation actions.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Button variant="secondary" size="sm" onClick={onBack} icon={<ArrowLeft className="w-3.5 h-3.5" />}>
            Upload Different File
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={() => onConfirmScoring(overrides)}
            disabled={!canProceed || isScoring}
            icon={<ArrowRight className="w-3.5 h-3.5" />}
          >
            {isScoring ? 'Scoring Cohort...' : 'Score Validated Cohort'}
          </Button>
        </div>
      </div>

      {/* Traffic Light Status Card */}
      <Card
        className={`border ${
          isGreen
            ? 'border-forest/40 bg-forest-subtle/40'
            : isAmber
            ? 'border-amber/40 bg-amber-subtle/40'
            : 'border-crimson/40 bg-crimson-subtle/40'
        }`}
      >
        <CardContent className="p-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="flex items-start gap-4">
              <div
                className={`w-12 h-12 rounded-full flex items-center justify-center shrink-0 ${
                  isGreen
                    ? 'bg-forest text-surface'
                    : isAmber
                    ? 'bg-amber text-surface'
                    : 'bg-crimson text-surface'
                }`}
              >
                {isGreen ? (
                  <CheckCircle2 className="w-6 h-6" />
                ) : isAmber ? (
                  <AlertTriangle className="w-6 h-6" />
                ) : (
                  <XCircle className="w-6 h-6" />
                )}
              </div>

              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <h3 className="font-serif text-section font-semibold text-ink">
                    {isGreen
                      ? 'Dataset Audit Passed'
                      : isAmber
                      ? 'Dataset Audit Needs Review'
                      : 'Column Mapping Incomplete'}
                  </h3>
                  <Badge variant={isGreen ? 'forest' : isAmber ? 'amber' : 'crimson'}>
                    {validation.traffic_light}
                  </Badge>
                </div>
                <p className="text-body text-ink/80 max-w-2xl leading-relaxed">
                  {quality?.message ||
                    (isGreen
                      ? 'All required survey items matched cleanly. The file is ready for scoring.'
                      : 'Please verify the proposed column mappings below before proceeding.')}
                </p>
              </div>
            </div>

            {/* Quick Stats Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-surface p-3.5 rounded-md border border-border text-center">
              <div>
                <span className="text-[11px] text-muted block uppercase tracking-wider font-semibold">
                  Total Rows
                </span>
                <span className="text-section font-serif font-bold text-ink tabular-nums">
                  {validation.total_rows.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="text-[11px] text-muted block uppercase tracking-wider font-semibold">
                  Scorable
                </span>
                <span className="text-section font-serif font-bold text-forest tabular-nums">
                  {validation.scorable_rows_count.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="text-[11px] text-muted block uppercase tracking-wider font-semibold">
                  Missing (&lt;50%)
                </span>
                <span className="text-section font-serif font-bold text-muted tabular-nums">
                  {quality?.missing_answers_rows ?? 0}
                </span>
              </div>
              <div>
                <span className="text-[11px] text-muted block uppercase tracking-wider font-semibold">
                  Duplicates
                </span>
                <span
                  className={`text-section font-serif font-bold tabular-nums ${
                    (quality?.duplicate_rows ?? 0) > 0 ? 'text-amber-text' : 'text-muted'
                  }`}
                >
                  {quality?.duplicate_rows ?? 0}
                </span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Fixable Problems Remediation Strip */}
      {problems.length > 0 && (
        <Card className="border-border">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Wrench className="w-4 h-4 text-forest" />
              <CardTitle className="text-caption font-semibold uppercase tracking-wider text-ink">
                Automated Data Cleaning & Remediation ({problems.length})
              </CardTitle>
            </div>
            <CardDescription>
              The system detected the following formatting items and will handle them automatically during preprocessing.
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-0 space-y-2.5">
            {problems.map((prob, i) => (
              <div
                key={i}
                className="p-3 bg-paper rounded border border-border flex flex-wrap items-center justify-between gap-3 text-caption"
              >
                <div className="space-y-0.5">
                  <span className="font-semibold text-ink">{prob.issue}</span>
                  <p className="text-small text-muted">{prob.system_action}</p>
                </div>
                <Badge variant="neutral" className="tabular-nums shrink-0">
                  {prob.affected_rows} rows affected
                </Badge>
              </div>
            ))}

            {validation.problematic_rows_count > 0 && (
              <div className="pt-2 flex justify-end">
                <a
                  href={getDownloadProblemsUrl(validation.upload_id, 'csv')}
                  download={`problem_rows_${validation.upload_id}.csv`}
                  className="inline-flex items-center gap-1.5 text-caption font-medium text-forest hover:underline"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download list of problem rows ({validation.problematic_rows_count})
                </a>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Column Matching Table */}
      <Card className="border-border">
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <CardTitle className="text-section">Required Questions vs Detected Columns</CardTitle>
              <CardDescription>
                High-confidence matches are accepted automatically. You can change any column mapping using the dropdowns.
              </CardDescription>
            </div>
            <Badge variant="neutral">
              {mappedCount} of {totalRequired} mapped
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-caption">
              <thead>
                <tr className="border-b border-border bg-paper text-muted font-semibold text-[11px] uppercase tracking-wider">
                  <th className="py-3 px-4">Required Survey Item</th>
                  <th className="py-3 px-4">Matched Column in File</th>
                  <th className="py-3 px-4">Match Confidence</th>
                  <th className="py-3 px-4">Sample Values in Upload</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {matches.map((item) => {
                  const currentRaw = selectedMapping[item.required_key] ?? '';
                  const isHigh = item.confidence_tier === 'High';
                  const isCheck = item.confidence_tier === 'Check this';

                  return (
                    <tr key={item.required_key} className="hover:bg-paper/60 transition-colors">
                      {/* Required Item */}
                      <td className="py-3 px-4 font-medium text-ink">
                        <div>{item.plain_name}</div>
                        <div className="text-[11px] font-mono text-muted">{item.required_key}</div>
                      </td>

                      {/* Matched Column Dropdown */}
                      <td className="py-3 px-4">
                        <select
                          value={currentRaw}
                          onChange={(e) => handleColumnChange(item.required_key, e.target.value)}
                          className="w-full max-w-xs px-2.5 py-1.5 text-caption bg-surface border border-border rounded font-mono text-ink focus:outline-none focus:border-forest"
                        >
                          <option value="__unmapped__">-- Not Mapped --</option>
                          {validation.detected_columns.map((colName) => (
                            <option key={colName} value={colName}>
                              {colName}
                            </option>
                          ))}
                        </select>
                      </td>

                      {/* Confidence Tier Badge */}
                      <td className="py-3 px-4">
                        <Badge
                          variant={isHigh ? 'forest' : isCheck ? 'amber' : 'crimson'}
                          size="sm"
                        >
                          {isHigh ? 'High Confidence' : isCheck ? 'Check This' : 'Not Found'}
                        </Badge>
                        {item.confidence_score > 0 && (
                          <span className="text-[11px] font-mono text-muted ml-2">
                            {item.confidence_score.toFixed(0)}%
                          </span>
                        )}
                      </td>

                      {/* Sample Values */}
                      <td className="py-3 px-4">
                        {item.sample_values && item.sample_values.length > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            {item.sample_values.map((val, idx) => (
                              <span
                                key={idx}
                                className="px-1.5 py-0.5 bg-paper rounded border border-border text-[11px] font-mono text-ink/80"
                              >
                                {val}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span className="text-muted text-small italic">No sample data</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Footer Navigation Buttons */}
      <div className="flex flex-wrap items-center justify-between gap-4 pt-4 border-t border-border">
        <Button variant="secondary" onClick={onBack} icon={<ArrowLeft className="w-4 h-4" />}>
          Back to Upload
        </Button>

        <div className="flex items-center gap-3">
          {validation.problematic_rows_count > 0 && (
            <a
              href={getDownloadProblemsUrl(validation.upload_id, 'csv')}
              download={`problem_rows_${validation.upload_id}.csv`}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-caption font-medium border border-border rounded hover:bg-paper text-ink"
            >
              <Download className="w-3.5 h-3.5" />
              Download Problem Rows
            </a>
          )}

          <Button
            variant="primary"
            onClick={() => onConfirmScoring(overrides)}
            disabled={!canProceed || isScoring}
            icon={<ArrowRight className="w-4 h-4" />}
          >
            {isScoring
              ? 'Executing Scoring Pipeline...'
              : `Continue with ${validation.scorable_rows_count.toLocaleString()} Scorable Students`}
          </Button>
        </div>
      </div>
    </div>
  );
}
