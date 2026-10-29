import React, { useRef } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts-for-react';
import { Download, FileSpreadsheet } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from './Card';
import { Button } from './Button';
import { cn } from '@/utils/cn';

export interface ChartCardProps {
  title: string;
  subtitle?: string;
  sampleSize?: number;
  caption?: string;
  option: EChartsOption;
  height?: number | string;
  onExportCsv?: () => void;
  className?: string;
  headerAction?: React.ReactNode;
  tableAlternative?: React.ReactNode;
}

// Research-grade ECharts theme tokens
export const RESEARCH_CHART_THEME = {
  color: [
    '#1F4E46', // Primary forest
    '#2B5C8A', // Info blue
    '#C98A00', // Warning amber
    '#8C4A3B', // Clay
    '#6E7F3F', // Olive
    '#7A6F63', // Stone
  ],
  textStyle: {
    fontFamily: '"Public Sans", system-ui, sans-serif',
    color: '#5B6168',
    fontSize: 12,
  },
  grid: {
    top: 36,
    right: 20,
    bottom: 36,
    left: 44,
    containLabel: true,
  },
  line: {
    itemStyle: {
      borderWidth: 2,
    },
    lineStyle: {
      width: 2,
    },
    symbol: 'circle',
    symbolSize: 4,
  },
  bar: {
    barMaxWidth: 28,
    itemStyle: {
      borderRadius: 0, // Flat bars, NO rounded corners
    },
  },
  tooltip: {
    backgroundColor: '#1A1D21',
    borderColor: '#1A1D21',
    textStyle: {
      color: '#FFFFFF',
      fontFamily: '"Public Sans", sans-serif',
      fontSize: 12,
    },
    padding: [6, 10],
    borderRadius: 4,
    extraCssText: 'box-shadow: 0 4px 12px rgba(26,29,33,0.15);',
  },
};

export function ChartCard({
  title,
  subtitle,
  sampleSize,
  caption,
  option,
  height = 320,
  onExportCsv,
  className,
  headerAction,
  tableAlternative,
}: ChartCardProps) {
  const chartRef = useRef<ReactECharts>(null);
  const [showTable, setShowTable] = React.useState(false);

  const handleDownloadPng = () => {
    const chartInstance = chartRef.current?.getEchartsInstance();
    if (!chartInstance) return;
    const dataUrl = chartInstance.getDataURL({
      type: 'png',
      pixelRatio: 2,
      backgroundColor: '#FFFFFF',
    });
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = `${title.toLowerCase().replace(/[^a-z0-9]/g, '_')}.png`;
    link.click();
  };

  // Merge default research theme with custom options
  const mergedOption: EChartsOption = {
    color: RESEARCH_CHART_THEME.color,
    textStyle: RESEARCH_CHART_THEME.textStyle,
    grid: RESEARCH_CHART_THEME.grid,
    tooltip: {
      ...RESEARCH_CHART_THEME.tooltip,
      ...(option.tooltip || {}),
    },
    ...option,
  };

  return (
    <Card className={cn('overflow-hidden', className)}>
      <CardHeader className="flex flex-wrap items-center justify-between gap-3 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <CardTitle>{title}</CardTitle>
            {sampleSize !== undefined && (
              <span className="text-caption text-muted bg-paper px-2 py-0.5 rounded border border-line tabular-nums">
                N = {sampleSize.toLocaleString()}
              </span>
            )}
          </div>
          {subtitle && <CardDescription>{subtitle}</CardDescription>}
        </div>

        <div className="flex items-center gap-2">
          {headerAction}
          {tableAlternative && (
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setShowTable((prev) => !prev)}
            >
              {showTable ? 'Chart view' : 'Table view'}
            </Button>
          )}
          {onExportCsv && (
            <Button
              variant="secondary"
              size="sm"
              onClick={onExportCsv}
              icon={<FileSpreadsheet className="w-3.5 h-3.5" />}
            >
              CSV
            </Button>
          )}
          <Button
            variant="secondary"
            size="sm"
            onClick={handleDownloadPng}
            icon={<Download className="w-3.5 h-3.5" />}
          >
            PNG
          </Button>
        </div>
      </CardHeader>

      <CardContent className="pt-2 pb-4">
        {showTable && tableAlternative ? (
          <div className="overflow-x-auto my-2 border border-line rounded">
            {tableAlternative}
          </div>
        ) : (
          <div
            role="img"
            aria-label={`${title}. ${subtitle ?? caption ?? 'Chart visualization.'}`}
          >
            <ReactECharts
              ref={chartRef}
              option={mergedOption}
              style={{ height, width: '100%' }}
              opts={{ renderer: 'canvas' }}
              notMerge={true}
            />
          </div>
        )}
        {/* Screen reader text alternative always present in DOM */}
        {tableAlternative && !showTable && (
          <div className="sr-only">
            <h4>Data table alternative for {title}</h4>
            {tableAlternative}
          </div>
        )}
        {caption && (
          <p className="text-caption text-muted mt-3 pt-2 border-t border-line/60 italic leading-relaxed">
            {caption}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
