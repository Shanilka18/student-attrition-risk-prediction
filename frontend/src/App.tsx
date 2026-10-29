import React, { useEffect, useState } from 'react';
import { Header, PageId } from './components/common/Header';
import { SyntheticBanner } from './components/common/SyntheticBanner';
import { OverviewPage } from './pages/OverviewPage';
import { DataCleaningPage } from './pages/DataCleaningPage';
import { ModelZooPage } from './pages/ModelZooPage';
import { LeaderboardPage } from './pages/LeaderboardPage';
import { FinalModelPage } from './pages/FinalModelPage';
import { ShapExplainabilityPage } from './pages/ShapExplainabilityPage';
import { CausalAnalysisPage } from './pages/CausalAnalysisPage';
import { UploadPredictPage } from './pages/UploadPredictPage';
import { apiService } from './services/api';
import { MetaResponse } from './types/api';

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<PageId>('overview');
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(true);
  const [meta, setMeta] = useState<MetaResponse | null>(null);

  useEffect(() => {
    async function checkBackend() {
      try {
        const [health, metaData] = await Promise.all([
          apiService.getHealth(),
          apiService.getMeta(),
        ]);
        setIsBackendOnline(health.status === 'healthy');
        setMeta(metaData);
      } catch (err) {
        setIsBackendOnline(false);
      }
    }
    checkBackend();
    const interval = setInterval(checkBackend, 15000);
    return () => clearInterval(interval);
  }, []);

  const renderActivePage = () => {
    switch (activePage) {
      case 'overview':
        return <OverviewPage />;
      case 'data-cleaning':
        return <DataCleaningPage />;
      case 'model-zoo':
        return <ModelZooPage />;
      case 'leaderboard':
        return <LeaderboardPage />;
      case 'final-model':
        return <FinalModelPage />;
      case 'shap':
        return <ShapExplainabilityPage />;
      case 'causal':
        return <CausalAnalysisPage />;
      case 'upload-predict':
        return <UploadPredictPage />;
      default:
        return <OverviewPage />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 flex flex-col font-sans transition-colors duration-200">
      {/* Persistent Synthetic Validation Banner */}
      <SyntheticBanner bannerText={meta?.data_banner || 'SYNTHETIC DATA - pipeline validation only'} />

      {/* Main Navigation Header */}
      <Header
        activePage={activePage}
        onPageChange={setActivePage}
        isBackendOnline={isBackendOnline}
      />

      {/* Page Content Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderActivePage()}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 py-6 text-center text-xs text-slate-500 dark:text-slate-400">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>
            <p>
              SLIIT CDAP 2026 | Project ID: <strong>KND-CDAP-IT-05</strong>
            </p>
            <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">
              Environmental & Financial Causal Risk Intelligence Framework
            </p>
          </div>
          <div className="flex items-center gap-3 text-[11px] font-mono bg-slate-100 dark:bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700">
            <span>Run: <strong className="text-blue-600 dark:text-blue-400">{meta?.run_id || 'run_prod_01'}</strong></span>
            <span>&bull;</span>
            <span>Artifacts: <span className="text-slate-600 dark:text-slate-300">{meta?.timestamp ? new Date(meta.timestamp).toLocaleDateString() : (meta?.training_date || '2026-10-25')}</span></span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
