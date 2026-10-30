import React, { useState, useMemo } from 'react';
import {
  UserCheck,
  Sparkles,
  AlertTriangle,
  ArrowRight,
  Shield,
  Users,
  ChevronLeft,
  ChevronRight,
  Filter,
  Layers,
  Search,
  X,
} from 'lucide-react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { SyntheticBadge } from '../components/common/Badges';
import { ChartCard } from '../components/common/ChartCard';
import { useCohort } from '../context/CohortContext';
import { StudentPrediction } from '../types/api';

export const StudentViewPage: React.FC = () => {
  const { predictionResult, fileName, selectedStudent, setSelectedStudent } = useCohort();

  const hasUploadedCohort = Boolean(predictionResult?.students && predictionResult.students.length > 0);

  // Active viewing mode: 'uploaded' vs 'preset'
  const [viewMode, setViewMode] = useState<'uploaded' | 'preset'>(() => {
    return hasUploadedCohort ? 'uploaded' : 'preset';
  });

  // Filter for uploaded cohort
  const [riskFilter, setRiskFilter] = useState<'All' | 'High' | 'Moderate' | 'Low'>('All');
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Preset synthetic sample students for demonstration
  const [selectedPreset, setSelectedPreset] = useState<'high' | 'moderate' | 'low'>('high');

  const presets = {
    high: {
      study_id: 'SLIIT_01986',
      pri_score: 84.5,
      risk_category: 'High' as const,
      factors: { stress: 4.67, motivation: 1.67, belongingness: 2.0, self_efficacy: 2.33, peer_support: 2.0 },
      shap_decomposition: [
        { feature: 'STRESS', val: 18.2, direction: 'risk_increasing' },
        { feature: 'BELONGING', val: 12.4, direction: 'risk_increasing' },
        { feature: 'MOTIVATION', val: 9.8, direction: 'risk_increasing' },
        { feature: 'PEER SUPPORT', val: -3.2, direction: 'risk_decreasing' },
      ],
      headline: 'Psychosocial risk is HIGH primarily driven by severe academic stress and alienation.',
      guidance: 'Actionable Outreach: Academic advising meeting recommended within 5 days; peer mentoring connection advised.',
    },
    moderate: {
      study_id: 'SLIIT_00514',
      pri_score: 51.2,
      risk_category: 'Moderate' as const,
      factors: { stress: 3.33, motivation: 3.0, belongingness: 3.0, self_efficacy: 3.67, peer_support: 3.33 },
      shap_decomposition: [
        { feature: 'STRESS', val: 5.4, direction: 'risk_increasing' },
        { feature: 'MOTIVATION', val: 2.8, direction: 'risk_increasing' },
        { feature: 'BELONGING', val: -3.1, direction: 'risk_decreasing' },
        { feature: 'SELF-EFFICACY', val: -6.8, direction: 'risk_decreasing' },
        { feature: 'PEER SUPPORT', val: -4.1, direction: 'risk_decreasing' },
      ],
      headline: 'Psychosocial risk is MODERATE with strong internal self-efficacy acting as a buffer.',
      guidance: 'Routine Follow-up: Mid-semester check-in via faculty advisor; maintain peer study group participation.',
    },
    low: {
      study_id: 'SLIIT_02946',
      pri_score: 18.7,
      risk_category: 'Low' as const,
      factors: { stress: 1.67, motivation: 4.67, belongingness: 4.33, self_efficacy: 4.67, peer_support: 4.67 },
      shap_decomposition: [
        { feature: 'STRESS', val: 2.1, direction: 'risk_increasing' },
        { feature: 'MOTIVATION', val: -14.2, direction: 'risk_decreasing' },
        { feature: 'BELONGING', val: -11.5, direction: 'risk_decreasing' },
        { feature: 'SELF-EFFICACY', val: -8.9, direction: 'risk_decreasing' },
        { feature: 'PEER SUPPORT', val: -10.8, direction: 'risk_decreasing' },
      ],
      headline: 'Psychosocial risk is LOW. Strong protective factors across all psychological domains.',
      guidance: 'Standard Academic Track: Student displays high resilience and strong institutional connection.',
    },
  };

  // Filtered list of uploaded students by risk tier
  const filteredUploadedStudents = useMemo(() => {
    if (!predictionResult?.students) return [];
    if (riskFilter === 'All') return predictionResult.students;
    return predictionResult.students.filter((s) => s.risk_category === riskFilter);
  }, [predictionResult?.students, riskFilter]);

  // Search filter across student ID
  const searchedStudents = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return filteredUploadedStudents;

    // Search within current filtered list first
    const withinFilter = filteredUploadedStudents.filter((s) =>
      s.study_id.toLowerCase().includes(q)
    );
    if (withinFilter.length > 0) return withinFilter;

    // Fallback: search across all students in cohort if not found in current risk tier
    if (predictionResult?.students) {
      return predictionResult.students.filter((s) =>
        s.study_id.toLowerCase().includes(q)
      );
    }
    return [];
  }, [filteredUploadedStudents, predictionResult?.students, searchQuery]);

  // Active student resolution
  const activeUploadedStudent: StudentPrediction | null = useMemo(() => {
    if (!searchedStudents.length) {
      return predictionResult?.students?.[0] || null;
    }
    if (selectedStudent && searchedStudents.some((s) => s.study_id === selectedStudent.study_id)) {
      return selectedStudent;
    }
    if (selectedStudent && predictionResult?.students?.some((s) => s.study_id === selectedStudent.study_id)) {
      return selectedStudent;
    }
    return searchedStudents[0];
  }, [searchedStudents, selectedStudent, predictionResult]);

  // Navigate next/prev in searched/filtered cohort
  const currentStudentIndex = useMemo(() => {
    if (!activeUploadedStudent || !searchedStudents.length) return 0;
    const idx = searchedStudents.findIndex((s) => s.study_id === activeUploadedStudent.study_id);
    return idx >= 0 ? idx : 0;
  }, [activeUploadedStudent, searchedStudents]);

  const handleStepStudent = (delta: number) => {
    if (!searchedStudents.length) return;
    const newIdx = (currentStudentIndex + delta + searchedStudents.length) % searchedStudents.length;
    setSelectedStudent(searchedStudents[newIdx]);
  };

  // Resolve active student model depending on viewMode
  const student = useMemo(() => {
    if (viewMode === 'uploaded' && activeUploadedStudent) {
      const stu = activeUploadedStudent;
      const factorScores = stu.factor_scores || {
        stress: 3.0,
        motivation: 3.0,
        belongingness: 3.0,
        self_efficacy: 3.0,
        peer_support: 3.0,
      };

      // Map raw contributor items
      const factorMap: Record<string, { val: number; direction: string }> = {};

      (stu.top_risk_contributors || []).forEach((c) => {
        const raw = (c.factor || c.feature || '').replace('factor_', '').toLowerCase();
        let key = raw;
        if (raw.includes('stress')) key = 'STRESS';
        else if (raw.includes('motiv')) key = 'MOTIVATION';
        else if (raw.includes('belong')) key = 'BELONGING';
        else if (raw.includes('effic') || raw.includes('self')) key = 'SELF-EFFICACY';
        else if (raw.includes('peer')) key = 'PEER SUPPORT';
        else key = raw.toUpperCase();

        factorMap[key] = {
          val: Number(Math.abs(c.shap_value).toFixed(2)),
          direction: 'risk_increasing',
        };
      });

      (stu.top_protective_contributors || []).forEach((c) => {
        const raw = (c.factor || c.feature || '').replace('factor_', '').toLowerCase();
        let key = raw;
        if (raw.includes('stress')) key = 'STRESS';
        else if (raw.includes('motiv')) key = 'MOTIVATION';
        else if (raw.includes('belong')) key = 'BELONGING';
        else if (raw.includes('effic') || raw.includes('self')) key = 'SELF-EFFICACY';
        else if (raw.includes('peer')) key = 'PEER SUPPORT';
        else key = raw.toUpperCase();

        factorMap[key] = {
          val: -Number(Math.abs(c.shap_value).toFixed(2)),
          direction: 'risk_decreasing',
        };
      });

      // Guarantee ALL 5 canonical psychometric factors are always present!
      const canonicalKeys = ['STRESS', 'MOTIVATION', 'BELONGING', 'SELF-EFFICACY', 'PEER SUPPORT'];
      const shapDecomp = canonicalKeys.map((fKey) => {
        if (factorMap[fKey]) {
          return {
            feature: fKey,
            val: factorMap[fKey].val,
            direction: factorMap[fKey].direction,
          };
        }
        // Fallback calculation from factor score if missing from SHAP list
        let estimatedVal = 0;
        let dir = 'risk_decreasing';
        if (fKey === 'STRESS') {
          estimatedVal = Number(((factorScores.stress - 3.0) * 3.5).toFixed(2));
          dir = estimatedVal >= 0 ? 'risk_increasing' : 'risk_decreasing';
        } else if (fKey === 'MOTIVATION') {
          estimatedVal = Number(((3.0 - factorScores.motivation) * 3.0).toFixed(2));
          dir = estimatedVal >= 0 ? 'risk_increasing' : 'risk_decreasing';
        } else if (fKey === 'BELONGING') {
          estimatedVal = Number(((3.0 - factorScores.belongingness) * 2.8).toFixed(2));
          dir = estimatedVal >= 0 ? 'risk_increasing' : 'risk_decreasing';
        } else if (fKey === 'SELF-EFFICACY') {
          estimatedVal = Number(((3.0 - factorScores.self_efficacy) * 2.5).toFixed(2));
          dir = estimatedVal >= 0 ? 'risk_increasing' : 'risk_decreasing';
        } else if (fKey === 'PEER SUPPORT') {
          estimatedVal = Number(((3.0 - factorScores.peer_support) * 2.0).toFixed(2));
          dir = estimatedVal >= 0 ? 'risk_increasing' : 'risk_decreasing';
        }
        return {
          feature: fKey,
          val: Math.abs(estimatedVal),
          direction: dir,
        };
      });

      return {
        study_id: stu.study_id,
        pri_score: stu.pri_score,
        risk_category: stu.risk_category,
        factors: factorScores,
        shap_decomposition: shapDecomp,
        headline: stu.narrative_headline || `Student ${stu.study_id} classified under ${stu.risk_category} risk tier.`,
        guidance: stu.narrative_actionable_advice || stu.narrative_drivers || 'Standard institutional follow-up recommended.',
      };
    }
    return presets[selectedPreset];
  }, [viewMode, activeUploadedStudent, selectedPreset]);

  const radarData = [
    { subject: 'Stress', val: student.factors.stress },
    { subject: 'Motivation', val: student.factors.motivation },
    { subject: 'Belonging', val: student.factors.belongingness },
    { subject: 'Self-Efficacy', val: student.factors.self_efficacy },
    { subject: 'Peer Support', val: student.factors.peer_support },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-2 border-b border-navy-800">
        <div>
          <div className="flex items-center gap-2.5">
            <h2 className="text-xl font-bold text-white tracking-tight">Individual Student Diagnostic View</h2>
            <SyntheticBadge />
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Matching research prototype interface: PRI circular gauge, 5-dimensional factor radar, and local SHAP waterfall breakdown.
          </p>
        </div>

        {/* View Mode Switcher (Uploaded Cohort vs Research Presets) */}
        <div className="flex items-center gap-2">
          {hasUploadedCohort ? (
            <div className="flex items-center p-1 bg-navy-900 border border-navy-700 rounded-lg text-xs font-semibold">
              <button
                onClick={() => setViewMode('uploaded')}
                className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                  viewMode === 'uploaded'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Users className="w-3.5 h-3.5" />
                Uploaded Cohort ({predictionResult?.students?.length})
              </button>
              <button
                onClick={() => setViewMode('preset')}
                className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                  viewMode === 'preset'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                Research Presets
              </button>
            </div>
          ) : (
            <div className="flex items-center p-1 bg-navy-900 border border-navy-700 rounded-lg text-xs font-semibold">
              <button
                onClick={() => setSelectedPreset('high')}
                className={`px-3 py-1.5 rounded-md transition-all ${
                  selectedPreset === 'high' ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' : 'text-slate-400'
                }`}
              >
                High Risk Case
              </button>
              <button
                onClick={() => setSelectedPreset('moderate')}
                className={`px-3 py-1.5 rounded-md transition-all ${
                  selectedPreset === 'moderate' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' : 'text-slate-400'
                }`}
              >
                Moderate Risk Case
              </button>
              <button
                onClick={() => setSelectedPreset('low')}
                className={`px-3 py-1.5 rounded-md transition-all ${
                  selectedPreset === 'low' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'text-slate-400'
                }`}
              >
                Low Risk Case
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Cohort Student Selector Bar (When viewing uploaded cohort) */}
      {viewMode === 'uploaded' && hasUploadedCohort && (
        <div className="p-3 bg-navy-900 border border-navy-800 rounded-xl flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-slate-400 font-medium flex items-center gap-1">
              <Filter className="w-3.5 h-3.5 text-indigo-400" /> Filter:
            </span>
            {(['All', 'High', 'Moderate', 'Low'] as const).map((tier) => (
              <button
                key={tier}
                onClick={() => setRiskFilter(tier)}
                className={`px-2.5 py-1 rounded text-xs transition-colors ${
                  riskFilter === tier
                    ? 'bg-indigo-600 text-white font-medium'
                    : 'bg-navy-800 text-slate-300 hover:bg-navy-750'
                }`}
              >
                {tier}
              </button>
            ))}

            {/* Search Input for typing Student ID */}
            <div className="relative flex items-center ml-2">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2 pointer-events-none" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && searchedStudents.length > 0) {
                    setSelectedStudent(searchedStudents[0]);
                  }
                }}
                placeholder="Type Student ID..."
                className="bg-navy-850 text-slate-100 placeholder-slate-500 border border-navy-700 rounded pl-7 pr-6 py-1 text-xs font-mono focus:border-indigo-500 focus:outline-none w-36 sm:w-44 transition-all"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-1.5 text-slate-400 hover:text-slate-200"
                  title="Clear search"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            {/* Student Dropdown Selector */}
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400 hidden sm:inline">Select:</span>
              <select
                value={activeUploadedStudent?.study_id || ''}
                onChange={(e) => {
                  const found = predictionResult?.students?.find((s) => s.study_id === e.target.value);
                  if (found) setSelectedStudent(found);
                }}
                className="bg-navy-850 text-slate-100 border border-navy-700 rounded px-2.5 py-1 text-xs font-mono focus:border-indigo-500 focus:outline-none max-w-[240px] sm:max-w-[280px] truncate"
              >
                {!searchedStudents.some((s) => s.study_id === activeUploadedStudent?.study_id) && activeUploadedStudent && (
                  <option key={activeUploadedStudent.study_id} value={activeUploadedStudent.study_id}>
                    {activeUploadedStudent.study_id} — PRI: {activeUploadedStudent.pri_score.toFixed(1)} ({activeUploadedStudent.risk_category}) [Active]
                  </option>
                )}
                {searchedStudents.map((s) => (
                  <option key={s.study_id} value={s.study_id}>
                    {s.study_id} — PRI: {s.pri_score.toFixed(1)} ({s.risk_category})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Stepper Controls */}
          <div className="flex items-center gap-2">
            <span className="text-slate-400 font-mono text-[11px]">
              {searchedStudents.length > 0
                ? `${currentStudentIndex + 1} of ${searchedStudents.length}${searchQuery ? ' (filtered)' : ''}`
                : '0 of 0'}
            </span>
            <button
              onClick={() => handleStepStudent(-1)}
              disabled={searchedStudents.length <= 1}
              className="p-1 rounded bg-navy-800 hover:bg-navy-750 text-slate-300 hover:text-white border border-navy-700 disabled:opacity-40"
              title="Previous Student"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => handleStepStudent(1)}
              disabled={searchedStudents.length <= 1}
              className="p-1 rounded bg-navy-800 hover:bg-navy-750 text-slate-300 hover:text-white border border-navy-700 disabled:opacity-40"
              title="Next Student"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Preset Sub-selector (When viewing preset mode while an uploaded cohort exists) */}
      {viewMode === 'preset' && hasUploadedCohort && (
        <div className="flex items-center justify-between p-3 bg-navy-900 border border-navy-800 rounded-xl text-xs">
          <span className="text-slate-400 font-medium">Demonstration Benchmark Presets:</span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setSelectedPreset('high')}
              className={`px-3 py-1 rounded transition-all ${
                selectedPreset === 'high' ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' : 'bg-navy-800 text-slate-400'
              }`}
            >
              High Risk Case
            </button>
            <button
              onClick={() => setSelectedPreset('moderate')}
              className={`px-3 py-1 rounded transition-all ${
                selectedPreset === 'moderate' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' : 'bg-navy-800 text-slate-400'
              }`}
            >
              Moderate Risk Case
            </button>
            <button
              onClick={() => setSelectedPreset('low')}
              className={`px-3 py-1 rounded transition-all ${
                selectedPreset === 'low' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-navy-800 text-slate-400'
              }`}
            >
              Low Risk Case
            </button>
          </div>
        </div>
      )}

      {/* Main Student Console */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: PRI Circular Gauge & Details */}
        <div className="bg-navy-900 border border-navy-800 rounded-xl p-6 shadow-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-[11px] font-mono text-slate-400 uppercase tracking-widest">Pseudonymized ID</span>
              <span className="font-mono text-sm font-bold text-indigo-300">{student.study_id}</span>
            </div>

            {/* Gauge */}
            <div className="py-6 flex flex-col items-center justify-center">
              <div className="relative w-44 h-44 flex items-center justify-center">
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="40" stroke="#1e293b" strokeWidth="8" fill="transparent" />
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    stroke={
                      student.risk_category === 'High'
                        ? '#ef4444'
                        : student.risk_category === 'Moderate'
                        ? '#f59e0b'
                        : '#10b981'
                    }
                    strokeWidth="8"
                    strokeDasharray={251.2}
                    strokeDashoffset={251.2 - (251.2 * student.pri_score) / 100}
                    strokeLinecap="round"
                    fill="transparent"
                    className="transition-all duration-1000 ease-out"
                  />
                </svg>
                <div className="absolute flex flex-col items-center">
                  <span className="text-4xl font-extrabold text-white font-mono">{student.pri_score.toFixed(1)}</span>
                  <span className="text-[11px] text-slate-400 font-semibold tracking-wider uppercase mt-0.5">
                    PRI Score
                  </span>
                </div>
              </div>
            </div>

            <div className="text-center mt-2">
              <span
                className={`inline-block px-4 py-1.5 rounded-full text-xs font-bold border ${
                  student.risk_category === 'High'
                    ? 'bg-rose-500/15 text-rose-300 border-rose-500/30'
                    : student.risk_category === 'Moderate'
                    ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                    : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                }`}
              >
                {student.risk_category} Risk Category
              </span>
            </div>
          </div>

          <div className="pt-6 border-t border-navy-800 text-xs text-slate-400 space-y-1.5 font-mono">
            <div className="flex justify-between">
              <span>Source:</span>
              <span className="text-slate-200">
                {viewMode === 'uploaded' ? (fileName || 'Uploaded Cohort') : 'Benchmark Preset'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Instrument:</span>
              <span className="text-slate-200">15-Item Psychometric</span>
            </div>
            <div className="flex justify-between">
              <span>Attribution Method:</span>
              <span className="text-slate-200">TreeSHAP / LinearSHAP</span>
            </div>
          </div>
        </div>

        {/* Middle Column: 5-Factor Radar Chart */}
        <ChartCard
          title="Psychometric Factor Radar"
          subtitle="Subscale scores across 5 latent domains (Scale: 1.0 to 5.0)"
          className="lg:col-span-1"
        >
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="subject" stroke="#94a3b8" fontSize={11} />
                <PolarRadiusAxis angle={30} domain={[1, 5]} stroke="#64748b" fontSize={10} />
                <Radar name="Student Score" dataKey="val" stroke="#818cf8" fill="#6366f1" fillOpacity={0.4} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        {/* Right Column: Local SHAP Waterfall Bar Chart */}
        <ChartCard
          title="Local SHAP Attribution Breakdown"
          subtitle="Additive contribution of factors to student's predicted risk score"
          className="lg:col-span-1"
        >
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={student.shap_decomposition}
                layout="vertical"
                margin={{ top: 10, right: 30, left: 90, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" stroke="#94a3b8" fontSize={11} />
                <YAxis dataKey="feature" type="category" stroke="#94a3b8" fontSize={10} width={90} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                  formatter={(val: number) => [val, 'SHAP Impact']}
                />
                <Bar
                  dataKey="val"
                  radius={[0, 4, 4, 0]}
                  fill="#6366f1"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>

      {/* Plain-Language Narrative Box */}
      <div className="bg-navy-900 border border-navy-800 rounded-xl p-5 shadow-lg space-y-3">
        <div className="flex items-center gap-2 text-indigo-300 font-semibold text-sm">
          <Sparkles className="w-4 h-4 text-indigo-400" />
          <h3>Deterministic Plain-Language Academic Interpretation</h3>
        </div>
        <p className="text-slate-200 text-xs leading-relaxed font-medium">{student.headline}</p>
        <p className="text-indigo-200/90 text-xs leading-relaxed p-3 rounded-lg bg-indigo-950/40 border border-indigo-500/20">
          <strong>Decision-Support Advisory:</strong> {student.guidance}
        </p>
      </div>
    </div>
  );
};
