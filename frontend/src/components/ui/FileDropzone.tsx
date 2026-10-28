import React, { useCallback, useState } from 'react';
import { useDropzone, type FileRejection } from 'react-dropzone';
import { Upload, FileSpreadsheet, AlertCircle, X } from 'lucide-react';
import { Button } from './Button';
import { cn } from '@/utils/cn';

export interface FileDropzoneProps {
  onFileSelect: (file: File) => void;
  maxSizeMB?: number;
  disabled?: boolean;
  className?: string;
}

export function FileDropzone({
  onFileSelect,
  maxSizeMB = 20,
  disabled = false,
  className,
}: FileDropzoneProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[], fileRejections: FileRejection[]) => {
      setErrorMessage(null);

      if (fileRejections.length > 0) {
        const rejection = fileRejections[0];
        if (rejection.errors.some((e) => e.code === 'file-too-large')) {
          setErrorMessage(`File exceeds the ${maxSizeMB} MB limit.`);
        } else if (rejection.errors.some((e) => e.code === 'file-invalid-type')) {
          setErrorMessage('Please upload an Excel (.xlsx, .xls) or CSV spreadsheet.');
        } else {
          setErrorMessage(rejection.errors[0]?.message || 'Invalid file format.');
        }
        setSelectedFile(null);
        return;
      }

      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setSelectedFile(file);
        onFileSelect(file);
      }
    },
    [maxSizeMB, onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    disabled,
    maxFiles: 1,
    maxSize: maxSizeMB * 1024 * 1024,
    accept: {
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'text/csv': ['.csv'],
      'text/tab-separated-values': ['.tsv'],
    },
  });

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedFile(null);
    setErrorMessage(null);
  };

  return (
    <div className={cn('w-full space-y-3', className)}>
      <div
        {...getRootProps()}
        className={cn(
          'border-2 border-dashed rounded-md p-8 text-center transition-colors duration-150 cursor-pointer select-none',
          isDragActive
            ? 'border-forest bg-forest-subtle/50'
            : 'border-line bg-surface hover:border-forest/50 hover:bg-[#FAF8F5]',
          disabled && 'opacity-50 cursor-not-allowed bg-paper',
          errorMessage && 'border-risk-high/40 bg-risk-high-subtle/20'
        )}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center justify-center">
          <div className="w-12 h-12 rounded-md bg-paper border border-line flex items-center justify-center text-muted mb-3">
            {selectedFile ? (
              <FileSpreadsheet className="w-6 h-6 text-forest stroke-1.5" />
            ) : (
              <Upload className="w-6 h-6 stroke-1.5" />
            )}
          </div>

          {selectedFile ? (
            <div className="space-y-1">
              <p className="font-serif font-semibold text-ink text-table">
                {selectedFile.name}
              </p>
              <p className="text-caption text-muted tabular-nums">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for validation
              </p>
              <div className="pt-2">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={handleClear}
                  icon={<X className="w-3.5 h-3.5" />}
                >
                  Choose a different file
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-1">
              <p className="text-table font-medium text-ink">
                {isDragActive
                  ? 'Drop the spreadsheet here...'
                  : 'Drag and drop your student survey file here, or browse'}
              </p>
              <p className="text-caption text-muted">
                Accepts Excel (.xlsx, .xls) and CSV files up to {maxSizeMB} MB
              </p>
            </div>
          )}
        </div>
      </div>

      {errorMessage && (
        <div className="flex items-center gap-2 text-caption text-risk-high bg-risk-high-subtle border border-risk-high/30 px-3 py-2 rounded">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
}
