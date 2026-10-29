import React, { useState, useEffect } from 'react';
import { ThemeProvider } from './context/ThemeContext';
import { api } from './api/client';
import type {
  HealthResponse,
  MetaResponse,
  LeaderboardResponse,
  ResearchQuestionAnswer,
  DataSummaryResponse,
} from './api/client';
import { Navbar } from './components/Navbar';
import type { PageId } from './components/Navbar';
import { OverviewPage } from './pages/OverviewPage';
import { DataPage } from './pages/DataPage';
import { LeaderboardPage } from './pages/LeaderboardPage';
import { EarlyWarningPage } from './pages/EarlyWarningPage';
import { ExplainabilityPage } from './pages/ExplainabilityPage';
import { PredictPage } from './pages/PredictPage';

export function AppContent() {
  const [currentPage, setCurrentPage] = useState<PageId>('overview');

  // Global Telemetry State
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardResponse | null>(null);
  const [researchAnswer, setResearchAnswer] = useState<ResearchQuestionAnswer | null>(null);
  const [dataSummary, setDataSummary] = useState<DataSummaryResponse | null>(null);

  const refreshMeta = () => {
    api.getMeta().then(setMeta).catch(console.warn);
  };

  useEffect(() => {
    // Initial fetch of foundational metadata
    api.getHealth().then(setHealth).catch(console.warn);
    refreshMeta();
    api.getLeaderboard().then(setLeaderboard).catch(console.warn);
    api.getResearchAnswer().then(setResearchAnswer).catch(console.warn);
    api.getDataSummary().then(setDataSummary).catch(console.warn);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans transition-colors">
      {/* 1. Header Navigation & Persistent Synthetic Banner */}
      <Navbar
        currentPage={currentPage}
        onPageChange={setCurrentPage}
        health={health}
        meta={meta}
      />

      {/* 2. Main Page Content Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-8">
        {currentPage === 'overview' && (
          <OverviewPage
            meta={meta}
            researchAnswer={researchAnswer}
            onNavigate={setCurrentPage}
          />
        )}

        {currentPage === 'data' && <DataPage dataSummary={dataSummary} />}

        {currentPage === 'leaderboard' && (
          <LeaderboardPage leaderboard={leaderboard} />
        )}

        {currentPage === 'early-warning' && <EarlyWarningPage />}

        {currentPage === 'explainability' && <ExplainabilityPage />}

        {currentPage === 'predict' && (
          <PredictPage meta={meta} onRefreshMeta={refreshMeta} />
        )}
      </main>

      {/* 3. Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 px-4 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div>
            <span className="font-semibold text-slate-300">
              Academic Intelligence Framework (KND_CDAP_IT_05)
            </span>
            <p className="mt-0.5">
              SLIIT Final Year Research Project | Author: V. S. H. Gunasekara (IT23164680)
            </p>
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span>FastAPI 0.115 + React 18 + Vite + Tailwind</span>
            <span>•</span>
            <span>NFR1 Compliant (&lt; 2.0s)</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}

export default App;
