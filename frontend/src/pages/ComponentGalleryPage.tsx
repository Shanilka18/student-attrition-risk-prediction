import { useState } from 'react';
import {
  Button,
  IconButton,
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Stat,
  Tag,
  StatusTag,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
  Tooltip,
  Term,
  DataTable,
  ChartCard,
  EmptyState,
  ErrorState,
  FileDropzone,
  Stepper,
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  Drawer,
  DrawerTrigger,
  DrawerContent,
  useToast,
  StatSkeleton,
  TableSkeleton,
} from '@/components/ui';
import {
  Download,
  Share2,
  TrendingUp,
  Inbox,
  Sliders,
} from 'lucide-react';
import type { ColumnDef } from '@tanstack/react-table';

interface GalleryRow {
  id: string;
  name: string;
  category: string;
  f1: number;
  auc: number;
  status: string;
}

const GALLERY_TABLE_DATA: GalleryRow[] = [
  { id: 'knn', name: 'k-Nearest Neighbours', category: 'Classical', f1: 0.852, auc: 0.902, status: 'Good fit' },
  { id: 'gru', name: 'Gated Recurrent Unit', category: 'Deep', f1: 0.841, auc: 0.895, status: 'Good fit' },
  { id: 'xgboost', name: 'XGBoost (Hybrid)', category: 'Boosting', f1: 0.838, auc: 0.889, status: 'Good fit' },
  { id: 'rf', name: 'Random Forest', category: 'Classical', f1: 0.831, auc: 0.884, status: 'Slight overfit' },
  { id: 'mlp', name: 'Wide MLP', category: 'Deep', f1: 0.825, auc: 0.879, status: 'Overfit' },
];

const GALLERY_COLUMNS: ColumnDef<GalleryRow, any>[] = [
  {
    accessorKey: 'name',
    header: 'Model Architecture',
    cell: ({ row }) => <span className="font-medium text-ink">{row.original.name}</span>,
  },
  {
    accessorKey: 'category',
    header: 'Category',
    cell: ({ row }) => <span className="text-muted">{row.original.category}</span>,
  },
  {
    accessorKey: 'f1',
    header: 'Validation F1',
    cell: ({ row }) => <span className="tabular-nums font-mono font-medium">{row.original.f1.toFixed(3)}</span>,
  },
  {
    accessorKey: 'auc',
    header: 'AUC-ROC',
    cell: ({ row }) => <span className="tabular-nums font-mono text-muted">{row.original.auc.toFixed(3)}</span>,
  },
  {
    accessorKey: 'status',
    header: 'Generalisation',
    cell: ({ row }) => <StatusTag status={row.original.status} />,
  },
];

export function ComponentGalleryPage() {
  const { toast } = useToast();
  const [activeStep, setActiveStep] = useState(1);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Sample chart option
  const sampleChartOption = {
    xAxis: {
      type: 'category',
      data: ['Period 1', 'Period 2', 'Period 3', 'Period 4'],
      axisLine: { lineStyle: { color: '#DDD8CE' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      name: 'Engagement (0–100)',
      nameTextStyle: { color: '#5B6168', fontSize: 11 },
      splitLine: { lineStyle: { color: '#E7E2D8', width: 1 } },
    },
    series: [
      {
        name: 'Consistent cohort',
        type: 'line',
        data: [72, 70, 71, 73],
        itemStyle: { color: '#1F4E46' },
      },
      {
        name: 'Declining cohort',
        type: 'line',
        data: [68, 55, 38, 22],
        itemStyle: { color: '#8C4A3B' },
      },
      {
        name: 'Recovering cohort',
        type: 'line',
        data: [35, 42, 60, 68],
        itemStyle: { color: '#2B5C8A' },
      },
    ],
  };

  return (
    <div className="space-y-10 pb-12">
      {/* Page Header */}
      <div className="border-b border-line pb-4">
        <div className="inline-block px-2.5 py-0.5 bg-forest-subtle text-forest text-caption font-semibold rounded mb-2">
          Internal Design System Showcase
        </div>
        <h2 className="font-serif text-page-title text-ink font-semibold">
          Shared Components Gallery
        </h2>
        <p className="text-body text-muted mt-1">
          Accessible research UI primitives constructed to adhere strictly to the project design tokens.
        </p>
      </div>

      {/* 1. Buttons & IconButtons */}
      <section className="space-y-4">
        <h3 className="font-serif text-section-title text-ink font-semibold">1. Buttons</h3>
        <Card>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <Button variant="primary">Primary Action</Button>
              <Button variant="secondary">Secondary Action</Button>
              <Button variant="ghost">Tertiary / Ghost</Button>
              <Button variant="danger">Destructive Action</Button>
              <Button variant="primary" loading>Loading Button</Button>
              <Button variant="secondary" disabled>Disabled State</Button>
            </div>
            <div className="flex flex-wrap items-center gap-3 pt-2">
              <Button size="sm" variant="secondary" icon={<Download className="w-3.5 h-3.5" />}>Small (sm)</Button>
              <Button size="md" variant="secondary" icon={<Sliders className="w-4 h-4" />}>Medium (md)</Button>
              <Button size="lg" variant="primary">Large (lg)</Button>
              <IconButton aria-label="Share" variant="secondary"><Share2 className="w-4 h-4" /></IconButton>
              <IconButton aria-label="Download" variant="ghost"><Download className="w-4 h-4" /></IconButton>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 2. Stats */}
      <section className="space-y-4">
        <h3 className="font-serif text-section-title text-ink font-semibold">2. Research Stats</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Stat
            label={<Term termKey="mei" customLabel="Average Risk Score" />}
            value="24.8"
            unit="/ 100"
            note={<><TrendingUp className="w-3.5 h-3.5 text-risk-low" /> -2.4 pts from previous term</>}
            sampleSize={2000}
            indicator="positive"
          />
          <Stat
            label={<Term termKey="f1" customLabel="At-Risk F1 Score" />}
            value="85.2%"
            note="Cross-validation optimal threshold 0.40"
            sampleSize={1600}
            indicator="neutral"
          />
          <Stat
            label="Declining Cohort Share"
            value="21.1%"
            note="421 students flagged for early outreach"
            sampleSize={2000}
            indicator="warning"
          />
          <Stat
            label={<Term termKey="generalisation_gap" customLabel="Over-Practice Gap" />}
            value="0.052"
            note="Status: Good fit (under 0.10 threshold)"
            indicator="positive"
          />
        </div>
      </section>

      {/* 3. Tags & Status Badges */}
      <section className="space-y-4">
        <h3 className="font-serif text-section-title text-ink font-semibold">3. Tags & Status Badges</h3>
        <Card>
          <CardContent className="space-y-4">
            <div>
              <div className="text-caption font-semibold text-muted uppercase tracking-wider mb-2">Overfit Status</div>
              <div className="flex flex-wrap items-center gap-2">
                <Tag variant="good-fit">Good fit</Tag>
                <Tag variant="slight-overfit">Slight overfit</Tag>
                <Tag variant="overfit">Overfit</Tag>
              </div>
            </div>

            <div>
              <div className="text-caption font-semibold text-muted uppercase tracking-wider mb-2">Risk Tiers</div>
              <div className="flex flex-wrap items-center gap-2">
                <Tag variant="risk-low">Low risk (0–34)</Tag>
                <Tag variant="risk-moderate">Moderate risk (35–64)</Tag>
                <Tag variant="risk-high">High risk (65–100)</Tag>
              </div>
            </div>

            <div>
              <div className="text-caption font-semibold text-muted uppercase tracking-wider mb-2">Longitudinal Patterns</div>
              <div className="flex flex-wrap items-center gap-2">
                <Tag variant="consistent">Consistent</Tag>
                <Tag variant="declining">Declining</Tag>
                <Tag variant="recovering">Recovering</Tag>
                <Tag variant="variable">Variable / Sporadic</Tag>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 4. Interactive Terms & Tooltips */}
      <section className="space-y-4">
        <h3 className="font-serif text-section-title text-ink font-semibold">4. Plain-English Terms with Tooltips</h3>
        <Card>
          <CardContent className="space-y-3">
            <p className="text-body text-ink">
              The university retention model prioritises <Term termKey="f1" /> and <Term termKey="auc" /> to ensure that advisors receive reliable alerts.
              Each student is assigned a calibrated <Term termKey="mei" /> within a dedicated <Term termKey="band" />.
              To protect staff from over-interpreting flukes, we compute the <Term termKey="generalisation_gap" /> and report the <Term termKey="calibration" />.
            </p>
            <div className="pt-2">
              <Tooltip content="Direct hover popover for contextual assistance">
                <span className="inline-block px-3 py-1 bg-paper border border-line rounded text-table text-muted cursor-help">
                  Hover over this generic tooltip test element
                </span>
              </Tooltip>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 5. Tabs */}
      <section className="space-y-4">
        <h3 className="font-serif text-section-title text-ink font-semibold">5. Tabs</h3>
        <Card>
          <CardContent>
            <Tabs defaultValue="tab1">
              <TabsList>
                <TabsTrigger value="tab1">Classification Track A</TabsTrigger>
                <TabsTrigger value="tab2">Continuous Track B</TabsTrigger>
                <TabsTrigger value="tab3">Unsupervised Clusters</TabsTrigger>
              </TabsList>
              <TabsContent value="tab1" className="p-3 bg-[#FAF8F5] rounded border border-line text-body text-ink">
                Track A benchmarks all 34 candidate models predicting student attrition status (at_risk = 1) under optimal decision cutoffs.
              </TabsContent>
              <TabsContent value="tab2" className="p-3 bg-[#FAF8F5] rounded border border-line text-body text-ink">
                Track B evaluates 32 regression models estimating continuous intent score (0–100) using R², RMSE, and Spearman correlation.
              </TabsContent>
              <TabsContent value="tab3" className="p-3 bg-[#FAF8F5] rounded border border-line text-body text-ink">
                Unsupervised clustering across KMeans, GMM, and Ward agglomerative hierarchical clustering with bootstrap stability scoring.
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </section>

      {/* 6. DataTable */}
      <section className="space-y-4">
        <h3 className="font-serif text-section-title text-ink font-semibold">6. Dense Research DataTable</h3>
        <DataTable
          data={GALLERY_TABLE_DATA}
          columns={GALLERY_COLUMNS}
          filename="benchmark_models_export.csv"
          defaultPageSize={10}
        />
      </section>

      {/* 7. ChartCard */}
      <section className="space-y-4">
        <h3 className="font-serif text-section-title text-ink font-semibold">7. Research ChartCard</h3>
        <ChartCard
          title="Longitudinal Engagement Trajectories by Pattern"
          subtitle="Mean composite engagement scores (0–100) across 4 academic periods."
          sampleSize={2000}
          option={sampleChartOption}
          height={300}
          onExportCsv={() => toast({ title: 'Export Triggered', description: 'Sample trajectory coordinates exported to CSV.', type: 'info' })}
        />
      </section>

      {/* 8. Stepper */}
      <section className="space-y-4">
        <h3 className="font-serif text-section-title text-ink font-semibold">8. Multi-Step Stepper</h3>
        <Card>
          <CardContent className="space-y-4">
            <Stepper
              steps={[
                { title: 'Upload file', description: 'Excel or CSV spreadsheet' },
                { title: 'Confirm mappings', description: '28 engagement questions' },
                { title: 'View results', description: 'Scores and risk bands' },
              ]}
              activeStep={activeStep}
            />
            <div className="flex gap-2 pt-2">
              <Button size="sm" variant="secondary" onClick={() => setActiveStep((prev) => Math.max(0, prev - 1))} disabled={activeStep === 0}>
                Previous step
              </Button>
              <Button size="sm" variant="primary" onClick={() => setActiveStep((prev) => Math.min(2, prev + 1))} disabled={activeStep === 2}>
                Next step
              </Button>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 9. Dropzone, Dialog, Drawer, Toast */}
      <section className="space-y-4">
        <h3 className="font-serif text-section-title text-ink font-semibold">9. Dropzone, Modals & Notifications</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader><CardTitle>File Dropzone</CardTitle></CardHeader>
            <CardContent>
              <FileDropzone onFileSelect={(file) => toast({ title: 'File Received', description: `${file.name} ready for validation.`, type: 'success' })} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Interactive Overlays & Toasts</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap gap-2">
                <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
                  <DialogTrigger asChild>
                    <Button variant="secondary">Open Accessible Dialog</Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Confirm Model Activation</DialogTitle>
                      <DialogDescription>
                        Switching active run to run_20261026_124514 will update the leaderboard, student scores, and explainability point clouds across all campus views.
                      </DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                      <Button variant="secondary" onClick={() => setDialogOpen(false)}>Cancel</Button>
                      <Button variant="primary" onClick={() => { setDialogOpen(false); toast({ title: 'Run Activated', description: 'Active run re-bound successfully.', type: 'success' }); }}>Confirm Activation</Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>

                <Drawer open={drawerOpen} onOpenChange={setDrawerOpen}>
                  <DrawerTrigger asChild>
                    <Button variant="secondary">Open Side Drawer</Button>
                  </DrawerTrigger>
                  <DrawerContent>
                    <h3 className="font-serif text-section-title font-semibold text-ink mb-2">Student Attribution Breakdown</h3>
                    <p className="text-body text-muted mb-4">Detailed waterfall factor contributions for student STU_2026_0042.</p>
                    <div className="p-4 bg-paper rounded border border-line text-caption space-y-2">
                      <div className="flex justify-between font-medium"><span>Baseline Population MEI</span><span className="tabular-nums font-mono">24.8 pts</span></div>
                      <div className="flex justify-between text-risk-high"><span>Steep Quiz Practice drop</span><span className="tabular-nums font-mono">+18 pts</span></div>
                      <div className="flex justify-between text-risk-high"><span>Assignment engagement drop</span><span className="tabular-nums font-mono">+14 pts</span></div>
                      <div className="flex justify-between text-risk-low"><span>High initial study hours (P1)</span><span className="tabular-nums font-mono">-6 pts</span></div>
                      <div className="pt-2 border-t border-line flex justify-between font-bold text-ink"><span>Final MEI</span><span className="tabular-nums font-mono">51 pts (Moderate)</span></div>
                    </div>
                  </DrawerContent>
                </Drawer>

                <Button
                  variant="primary"
                  onClick={() => toast({ title: 'Outreach Flag Saved', description: 'Student advisory record updated.', type: 'success' })}
                >
                  Trigger Toast
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* 10. Empty & Error States */}
      <section className="space-y-4">
        <h3 className="font-serif text-section-title text-ink font-semibold">10. Empty & Error States</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <EmptyState
            title="No Scored Cohorts Yet"
            description="Upload an Excel or CSV file containing the 28 student engagement survey answers to calculate MEI risk scores."
            icon={<Inbox className="w-6 h-6 stroke-1 text-forest" />}
            action={<Button variant="primary">Upload survey</Button>}
          />
          <ErrorState
            title="Could not load leaderboard"
            message="We were unable to reach the backend server to load cross-validation fold telemetry. Check server connection."
            error="HTTP 503 Service Unavailable: Backend is offline"
            onRetry={() => toast({ title: 'Retrying connection...', type: 'info' })}
          />
        </div>
      </section>

      {/* 11. Skeletons */}
      <section className="space-y-4">
        <h3 className="font-serif text-section-title text-ink font-semibold">11. Loading Skeletons</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <StatSkeleton />
          <StatSkeleton />
          <StatSkeleton />
        </div>
        <TableSkeleton rows={3} cols={4} />
      </section>
    </div>
  );
}
