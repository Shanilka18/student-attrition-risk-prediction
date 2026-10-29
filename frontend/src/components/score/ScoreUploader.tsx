import { useState } from 'react';
import { Upload, Download, FileSpreadsheet, ShieldCheck, UserCheck, AlertCircle, FileText } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { getTemplateDownloadUrl } from '@/api/queries';

interface ScoreUploaderProps {
  onFileSelect: (file: File) => void;
  onOpenSingleScore: () => void;
  isLoading?: boolean;
  error?: string | null;
}

export function ScoreUploader({
  onFileSelect,
  onOpenSingleScore,
  isLoading = false,
  error = null,
}: ScoreUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      validateAndPass(file);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      validateAndPass(file);
    }
  };

  const validateAndPass = (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!['csv', 'xlsx', 'xls'].includes(ext || '')) {
      setFileError('Choose a CSV or Excel survey spreadsheet (.csv, .xlsx, or .xls).');
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setFileError('This file is larger than the 20 MB limit.');
      return;
    }
    setFileError(null);
    onFileSelect(file);
  };

  const errorMessage = fileError ?? error;

  return (
    <div className="space-y-6">
      {/* Primary Upload Dropzone Card */}
      <Card className="border-border">
        <CardContent className="p-4 sm:p-8">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative flex flex-col items-center justify-center p-4 sm:p-10 border-2 border-dashed rounded-lg transition-colors cursor-pointer text-center ${
              isDragging
                ? 'border-forest bg-forest/5'
                : 'border-border-strong hover:border-forest hover:bg-forest/5'
            }`}
          >
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={handleFileInput}
              disabled={isLoading}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
              aria-label="Upload survey file"
            />

            <div className="w-14 h-14 rounded-full bg-forest-subtle flex items-center justify-center text-forest mb-4 shadow-none">
              <Upload className="w-6 h-6" />
            </div>

            <h3 className="font-serif text-section font-semibold text-ink">
              Drop survey spreadsheet here, or browse
            </h3>
            <p className="text-body text-muted mt-1 max-w-md">
              Supports CSV, XLSX, and XLS formats up to 20 MB. Any column ordering or header text will be automatically mapped.
            </p>

            <div className="flex flex-wrap items-center justify-center gap-2 mt-4">
              <Badge variant="neutral">Excel (.xlsx, .xls)</Badge>
              <Badge variant="neutral">CSV (.csv)</Badge>
              <Badge variant="neutral">Max 20 MB</Badge>
              <Badge variant="forest">Auto-detection</Badge>
            </div>

            {isLoading && (
              <div className="mt-4 text-caption text-forest font-medium flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-forest border-t-transparent rounded-full animate-spin" />
                Reading and validating survey structure...
              </div>
            )}
          </div>

          {errorMessage && (
            <div
              className="mt-4 p-4 bg-crimson-subtle border border-crimson/30 rounded-md flex items-start gap-3"
              role="alert"
              aria-live="assertive"
            >
              <AlertCircle className="w-5 h-5 text-crimson shrink-0 mt-0.5" />
              <div className="text-caption text-crimson">
                <span className="font-semibold">
                  {fileError ? 'File Error:' : 'Validation Error:'}
                </span>{' '}
                {errorMessage}
              </div>
            </div>
          )}

          {/* Action Strip: Templates, Dictionary & Single Student */}
          <div className="mt-6 pt-6 border-t border-border flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-caption text-muted font-medium">Resources:</span>
              <a
                href={getTemplateDownloadUrl('xlsx')}
                download="MEI_Survey_Template.xlsx"
                className="inline-flex items-center gap-1.5 text-caption font-medium text-forest hover:underline"
              >
                <Download className="w-3.5 h-3.5" />
                Download Excel Template
              </a>
              <span className="text-border-strong text-caption">|</span>
              <a
                href={getTemplateDownloadUrl('csv')}
                download="MEI_Survey_Template.csv"
                className="inline-flex items-center gap-1.5 text-caption font-medium text-forest hover:underline"
              >
                <Download className="w-3.5 h-3.5" />
                Download CSV Template
              </a>
              <span className="text-border-strong text-caption">|</span>
              <a
                href="/api/dictionary"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 text-caption font-medium text-forest hover:underline"
              >
                <FileText className="w-3.5 h-3.5" />
                Data Dictionary
              </a>
            </div>

            <Button
              variant="secondary"
              size="sm"
              onClick={onOpenSingleScore}
              icon={<UserCheck className="w-4 h-4 text-forest" />}
            >
              Score an Individual Student
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Trust & Privacy Notice Card */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="border-border bg-paper">
          <CardContent className="p-5 flex items-start gap-3.5">
            <ShieldCheck className="w-5 h-5 text-forest shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h4 className="text-caption font-semibold text-ink">
                Strict Privacy & Automated 24-Hour Removal
              </h4>
              <p className="text-small text-muted leading-relaxed">
                Uploaded survey files are processed strictly within your isolated institutional server environment. No student identifiers or response records are transmitted to third-party APIs. Upload sessions are automatically purged after 24 hours.
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border bg-paper">
          <CardContent className="p-5 flex items-start gap-3.5">
            <FileSpreadsheet className="w-5 h-5 text-forest shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h4 className="text-caption font-semibold text-ink">
                Longitudinal Survey Structure
              </h4>
              <p className="text-small text-muted leading-relaxed">
                The model expects 7 engagement indicators measured across 4 semester intervals (28 indicators total): LMS resource access, study hours, review time, consistency, class participation, assignment timing, and quiz exercises.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
