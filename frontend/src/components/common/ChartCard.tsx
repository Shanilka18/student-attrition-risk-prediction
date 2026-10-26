import React, { useRef } from 'react';
import { Download, Camera } from 'lucide-react';
import { toPng } from 'html-to-image';

interface ChartCardProps {
  title: string;
  subtitle?: string;
  badge?: React.ReactNode;
  children: React.ReactNode;
  exportable?: boolean;
  className?: string;
}

export const ChartCard: React.FC<ChartCardProps> = ({
  title,
  subtitle,
  badge,
  children,
  exportable = true,
  className = '',
}) => {
  const cardRef = useRef<HTMLDivElement>(null);

  const handleExportPng = async () => {
    if (!cardRef.current) return;
    try {
      const dataUrl = await toPng(cardRef.current, {
        backgroundColor: '#0b1120',
        pixelRatio: 2,
      });
      const link = document.createElement('a');
      link.download = `${title.toLowerCase().replace(/\s+/g, '_')}_export.png`;
      link.href = dataUrl;
      link.click();
    } catch (err) {
      console.error('Failed to export chart PNG', err);
    }
  };

  return (
    <div
      ref={cardRef}
      className={`bg-navy-900 border border-navy-700/80 rounded-xl p-5 shadow-lg relative flex flex-col ${className}`}
    >
      <div className="flex items-start justify-between gap-4 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-sm text-slate-100">{title}</h3>
            {badge}
          </div>
          {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
        </div>
        {exportable && (
          <button
            onClick={handleExportPng}
            className="no-print p-1.5 rounded-md text-slate-400 hover:text-slate-200 hover:bg-navy-800 transition-colors"
            title="Export chart as PNG"
            aria-label="Export chart as PNG"
          >
            <Camera className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
      <div className="flex-1 w-full min-h-[220px]">{children}</div>
    </div>
  );
};
