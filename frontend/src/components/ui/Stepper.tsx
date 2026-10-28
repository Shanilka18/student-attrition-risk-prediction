import { Check } from 'lucide-react';
import { cn } from '@/utils/cn';

export interface StepItem {
  title: string;
  description?: string;
}

export interface StepperProps {
  steps: StepItem[];
  activeStep: number;
  className?: string;
}

export function Stepper({ steps, activeStep, className }: StepperProps) {
  return (
    <nav aria-label="Progress" className={cn('w-full', className)}>
      <ol className="flex items-center w-full">
        {steps.map((step, idx) => {
          const isCompleted = idx < activeStep;
          const isCurrent = idx === activeStep;

          return (
            <li
              key={step.title}
              className={cn(
                'relative flex items-center',
                idx !== steps.length - 1 ? 'flex-1 pr-4 sm:pr-8' : ''
              )}
            >
              <div className="flex items-center gap-3">
                <span
                  className={cn(
                    'w-7 h-7 rounded-full flex items-center justify-center text-caption font-semibold transition-colors duration-150 select-none shrink-0',
                    isCompleted && 'bg-forest text-surface',
                    isCurrent && 'border-2 border-forest text-forest bg-forest-subtle',
                    !isCompleted && !isCurrent && 'border border-line text-muted bg-paper'
                  )}
                  aria-current={isCurrent ? 'step' : undefined}
                >
                  {isCompleted ? <Check className="w-4 h-4 stroke-2" /> : idx + 1}
                </span>

                <div className="hidden sm:block">
                  <div
                    className={cn(
                      'text-caption font-medium',
                      isCurrent ? 'text-forest font-semibold' : 'text-ink'
                    )}
                  >
                    {step.title}
                  </div>
                  {step.description && (
                    <div className="text-[11px] text-muted">
                      {step.description}
                    </div>
                  )}
                </div>
              </div>

              {idx !== steps.length - 1 && (
                <div
                  className={cn(
                    'flex-1 h-[1px] ml-4 transition-colors duration-150',
                    isCompleted ? 'bg-forest' : 'bg-line'
                  )}
                  aria-hidden="true"
                />
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
