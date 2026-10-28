import { useNavigate } from 'react-router-dom';
import { Compass, Home } from 'lucide-react';
import { EmptyState } from '@/components/ui/EmptyState';
import { Button } from '@/components/ui/Button';

export function NotFoundPage() {
  const navigate = useNavigate();

  return (
    <div className="py-12">
      <EmptyState
        title="Page not found (404)"
        description="The analytics view or report URL you requested does not exist or has been relocated."
        icon={<Compass className="w-6 h-6 stroke-1.5 text-forest" />}
        action={
          <Button
            variant="primary"
            onClick={() => navigate('/')}
            icon={<Home className="w-4 h-4" />}
          >
            Return to overview
          </Button>
        }
      />
    </div>
  );
}
