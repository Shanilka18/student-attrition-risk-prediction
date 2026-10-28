import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { TooltipProvider } from '@/components/ui/Tooltip';
import { ToastProvider } from '@/components/ui/Toast';

export function AppShell() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <TooltipProvider>
      <ToastProvider>
        <div className="min-h-screen flex bg-paper text-ink selection:bg-forest-subtle selection:text-forest">
          {/* Sidebar */}
          <Sidebar
            isOpenMobile={mobileMenuOpen}
            onCloseMobile={() => setMobileMenuOpen(false)}
          />

          {/* Main Area */}
          <div className="flex-1 flex flex-col min-w-0">
            <TopBar onOpenMobileMenu={() => setMobileMenuOpen(true)} />

            <main className="flex-1 p-5 sm:p-8 max-w-content w-full mx-auto overflow-y-auto">
              <Outlet />
            </main>
          </div>
        </div>
      </ToastProvider>
    </TooltipProvider>
  );
}
