import { Database, Play, History } from 'lucide-react';
import { EmptyState } from '@/components/ui/EmptyState';
import { Button } from '@/components/ui/Button';

export function TrainPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="font-serif text-page-title text-ink font-semibold">
          Train on New Data
        </h2>
        <p className="text-body text-muted mt-1">
          Trigger end-to-end model retraining and state discovery on your university's validated survey data.
        </p>
      </div>

      <EmptyState
        title="Institutional Retraining Pipeline"
        description="Configure training time budgets, cross-validation splits, and run background model updates when new survey cohorts are administered."
        icon={<Database className="w-6 h-6 stroke-1.5 text-forest" />}
        action={
          <Button
            variant="primary"
            onClick={() => window.alert('Retraining job can be initiated via CLI or authenticated job endpoint.')}
            icon={<Play className="w-4 h-4" />}
          >
            Start retraining run
          </Button>
        }
        secondaryAction={
          <Button
            variant="secondary"
            onClick={() => window.open('/api/runs', '_blank')}
            icon={<History className="w-4 h-4" />}
          >
            View run history
          </Button>
        }
      />
    </div>
  );
}
