import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Check } from 'lucide-react';
import { useValidateUpload, useRunScoring } from '@/api/queries';
import type { ValidateUploadResponse, RunScoringResponse } from '@/api/types';
import { ScoreUploader } from '@/components/score/ScoreUploader';
import { ColumnCheckStep } from '@/components/score/ColumnCheckStep';
import { ScoreResultsStep } from '@/components/score/ScoreResultsStep';
import { SingleScoreModal } from '@/components/score/SingleScoreModal';

export function ScorePage() {
  const [searchParams] = useSearchParams();
  const [currentStep, setCurrentStep] = useState<1 | 2 | 3>(1);
  const [validatedData, setValidatedData] = useState<ValidateUploadResponse | null>(null);
  const [scoringResponse, setScoringResponse] = useState<RunScoringResponse | null>(null);
  const [isSingleScoreOpen, setIsSingleScoreOpen] = useState(false);

  const validateMutation = useValidateUpload();
  const runScoringMutation = useRunScoring();

  useEffect(() => {
    const uploadId = searchParams.get('upload_id');
    const stepParam = searchParams.get('step');
    const singleParam = searchParams.get('single');

    if (singleParam === 'true') {
      setIsSingleScoreOpen(true);
    }

    if (uploadId) {
      fetch(`/api/score/${uploadId}/session`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (data && data.validation) {
            setValidatedData(data.validation);
            if (data.scoring && stepParam !== '2') {
              setScoringResponse(data.scoring);
              setCurrentStep(3);
            } else {
              setCurrentStep(2);
            }
          }
        })
        .catch(() => {});
    }
  }, [searchParams]);

  const handleFileSelect = async (file: File) => {
    try {
      const res = await validateMutation.mutateAsync(file);
      setValidatedData(res);
      setCurrentStep(2);
    } catch (err: any) {
      console.error('Validation failed:', err);
    }
  };

  const handleConfirmScoring = async (overrides: Record<string, string>) => {
    if (!validatedData) return;
    try {
      const res = await runScoringMutation.mutateAsync({
        upload_id: validatedData.upload_id,
        mapping_overrides: Object.keys(overrides).length > 0 ? overrides : null,
      });
      setScoringResponse(res);
      setCurrentStep(3);
    } catch (err: any) {
      alert(`Scoring execution failed: ${err.message}`);
    }
  };

  const handleReset = () => {
    setValidatedData(null);
    setScoringResponse(null);
    setCurrentStep(1);
  };

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-serif text-page-title text-ink font-semibold">
          Score students
        </h1>
        <p className="text-body text-muted mt-1">
          Upload a survey spreadsheet to review student risk scores and their
          explanations.
        </p>
      </header>

      {/* 3-Step Stepper Progress Bar */}
      <div className="p-4 bg-surface rounded-lg border border-border">
        <div className="flex items-center justify-between max-w-2xl mx-auto">
          {/* Step 1 */}
          <div className="flex items-center gap-3">
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center font-mono text-small font-semibold ${
                currentStep === 1
                  ? 'bg-forest text-surface'
                  : currentStep > 1
                  ? 'bg-forest-subtle text-forest border border-forest/30'
                  : 'bg-paper text-muted border border-border'
              }`}
            >
              {currentStep > 1 ? <Check className="w-4 h-4" /> : '1'}
            </div>
            <div className="hidden sm:block">
              <span
                className={`text-caption font-semibold block ${
                  currentStep === 1 ? 'text-ink' : 'text-muted'
                }`}
              >
                1. Upload Survey
              </span>
              <span className="text-[11px] text-muted">CSV or Excel file</span>
            </div>
          </div>

          <div className="flex-1 h-px bg-border mx-4" />

          {/* Step 2 */}
          <div className="flex items-center gap-3">
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center font-mono text-small font-semibold ${
                currentStep === 2
                  ? 'bg-forest text-surface'
                  : currentStep > 2
                  ? 'bg-forest-subtle text-forest border border-forest/30'
                  : 'bg-paper text-muted border border-border'
              }`}
            >
              {currentStep > 2 ? <Check className="w-4 h-4" /> : '2'}
            </div>
            <div className="hidden sm:block">
              <span
                className={`text-caption font-semibold block ${
                  currentStep === 2 ? 'text-ink' : 'text-muted'
                }`}
              >
                2. Check Columns
              </span>
              <span className="text-[11px] text-muted">Validate schema & audit</span>
            </div>
          </div>

          <div className="flex-1 h-px bg-border mx-4" />

          {/* Step 3 */}
          <div className="flex items-center gap-3">
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center font-mono text-small font-semibold ${
                currentStep === 3
                  ? 'bg-forest text-surface'
                  : 'bg-paper text-muted border border-border'
              }`}
            >
              3
            </div>
            <div className="hidden sm:block">
              <span
                className={`text-caption font-semibold block ${
                  currentStep === 3 ? 'text-ink' : 'text-muted'
                }`}
              >
                3. Scored Results
              </span>
              <span className="text-[11px] text-muted">Index & explanations</span>
            </div>
          </div>
        </div>
      </div>

      {/* Step View Switcher */}
      {currentStep === 1 && (
        <ScoreUploader
          onFileSelect={handleFileSelect}
          onOpenSingleScore={() => setIsSingleScoreOpen(true)}
          isLoading={validateMutation.isPending}
          error={validateMutation.error ? validateMutation.error.message : null}
        />
      )}

      {currentStep === 2 && validatedData && (
        <ColumnCheckStep
          validation={validatedData}
          onConfirmScoring={handleConfirmScoring}
          onBack={() => setCurrentStep(1)}
          isScoring={runScoringMutation.isPending}
        />
      )}

      {currentStep === 3 && validatedData && scoringResponse && (
        <ScoreResultsStep
          uploadId={validatedData.upload_id}
          scoringResponse={scoringResponse}
          onReset={handleReset}
        />
      )}

      {/* Single Student Scoring Modal */}
      <SingleScoreModal
        isOpen={isSingleScoreOpen}
        onClose={() => setIsSingleScoreOpen(false)}
      />
    </div>
  );
}
