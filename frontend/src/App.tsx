import React, { useState } from 'react';
import { Sidebar, PageId } from './components/layout/Sidebar';
import { EthicsBanner } from './components/common/EthicsBanner';
import { OverviewPage } from './pages/OverviewPage';
import { ReliabilityPage } from './pages/ReliabilityPage';
import { LeaderboardPage } from './pages/LeaderboardPage';
import { ChampionPage } from './pages/ChampionPage';
import { ExplainabilityPage } from './pages/ExplainabilityPage';
import { ScoreDataPage } from './pages/ScoreDataPage';
import { StudentViewPage } from './pages/StudentViewPage';
import { CohortProvider, useCohort } from './context/CohortContext';

function AppContent() {
  const { currentPage, setCurrentPage } = useCohort();
  const [presentationMode, setPresentationMode] = useState<boolean>(false);

  const renderPage = () => {
    switch (currentPage) {
      case 'overview':
        return <OverviewPage />;
      case 'reliability':
        return <ReliabilityPage />;
      case 'leaderboard':
        return <LeaderboardPage />;
      case 'champion':
        return <ChampionPage />;
      case 'explainability':
        return <ExplainabilityPage />;
      case 'score':
        return <ScoreDataPage />;
      case 'student':
        return <StudentViewPage />;
      default:
        return <OverviewPage />;
    }
  };

  return (
    <div className={`min-h-screen flex flex-col bg-navy-950 text-slate-100 ${presentationMode ? 'text-base font-normal' : 'text-sm'}`}>
      {/* Top Persistent Legal and Governance Banner */}
      <EthicsBanner />

      {/* Main Workspace Body */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Navigation Sidebar */}
        <Sidebar
          currentPage={currentPage}
          onSelectPage={setCurrentPage}
          presentationMode={presentationMode}
          onTogglePresentation={() => setPresentationMode(!presentationMode)}
        />

        {/* Content Area */}
        <main className={`flex-1 overflow-y-auto p-6 md:p-8 ${presentationMode ? 'max-w-7xl mx-auto space-y-8' : ''}`}>
          {renderPage()}
        </main>
      </div>
    </div>
  );
}

export function App() {
  return (
    <CohortProvider>
      <AppContent />
    </CohortProvider>
  );
}

export default App;
