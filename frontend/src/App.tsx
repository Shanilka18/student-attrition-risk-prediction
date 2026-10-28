import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppShell } from '@/components/layout/AppShell';
import { OverviewPage } from '@/pages/OverviewPage';
import { LeaderboardPage } from '@/pages/LeaderboardPage';
import { ModelDetailPage } from '@/pages/ModelDetailPage';
import { ComparePage } from '@/pages/ComparePage';
import { PatternsPage } from '@/pages/PatternsPage';
import { DriversPage } from '@/pages/DriversPage';
import { ScorePage } from '@/pages/ScorePage';
import { TrainPage } from '@/pages/TrainPage';
import { NotFoundPage } from '@/pages/NotFoundPage';
import { ComponentGalleryPage } from '@/pages/ComponentGalleryPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/overview" element={<Navigate to="/" replace />} />
            <Route path="/leaderboard" element={<LeaderboardPage />} />
            <Route path="/leaderboard/:modelId" element={<ModelDetailPage />} />
            <Route path="/compare" element={<ComparePage />} />
            <Route path="/patterns" element={<PatternsPage />} />
            <Route path="/drivers" element={<DriversPage />} />
            <Route path="/score" element={<ScorePage />} />
            <Route path="/train" element={<TrainPage />} />
            <Route path="/_components" element={<ComponentGalleryPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
