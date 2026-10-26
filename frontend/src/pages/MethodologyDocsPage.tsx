import React, { useEffect, useState } from 'react';
import { BookOpen, FileText, ChevronRight, Layers, ArrowRight } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api } from '../services/api';
import { DocItem } from '../types/api';
import { LoadingState } from '../components/common/FeedbackStates';

export const MethodologyDocsPage: React.FC = () => {
  const [docList, setDocList] = useState<DocItem[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<string>('00_project_overview.md');
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchList = async () => {
      try {
        const list = await api.getDocsList();
        setDocList(list);
        if (list.length > 0) {
          setSelectedDoc(list[0].filename);
        }
      } catch (err) {
        console.error('Failed to load doc list', err);
      }
    };
    fetchList();
  }, []);

  useEffect(() => {
    if (!selectedDoc) return;
    const fetchContent = async () => {
      setLoading(true);
      try {
        const data = await api.getDocContent(selectedDoc);
        setContent(data.content);
      } catch (err) {
        setContent('# Error\nFailed to load document content.');
      } finally {
        setLoading(false);
      }
    };
    fetchContent();
  }, [selectedDoc]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="pb-2 border-b border-navy-800">
        <h2 className="text-xl font-bold text-white tracking-tight">Methodology & Viva Documentation</h2>
        <p className="text-xs text-slate-400 mt-1">
          In-app rendering of formal research phase documents (ELI5 + Technical Examiner Justifications) and Fusion Framework architecture.
        </p>
      </div>

      {/* Fusion Framework Architecture Blueprint */}
      <div className="bg-navy-900 border border-navy-800 rounded-xl p-5 shadow-lg">
        <div className="flex items-center gap-2 mb-3">
          <Layers className="w-4 h-4 text-indigo-400" />
          <h3 className="font-semibold text-sm text-slate-100">Multi-Factor Student Attrition Architecture (Fusion Context)</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-3 rounded-lg bg-navy-950 border border-indigo-500/40 shadow-sm relative overflow-hidden">
            <span className="text-[10px] text-indigo-400 uppercase tracking-widest block font-bold">This Component (Ours)</span>
            <div className="font-bold text-slate-100 text-sm mt-1">Psychosocial Risk (PRI)</div>
            <p className="text-[11px] text-slate-400 mt-1">15-Item Likert (Stress, Motivation, Belonging, Efficacy, Peer)</p>
            <div className="mt-2 text-indigo-300 font-bold text-[10px]">Outputs PRI (0-100) + SHAP</div>
          </div>

          <div className="p-3 rounded-lg bg-navy-950/60 border border-navy-800 opacity-80">
            <span className="text-[10px] text-slate-400 uppercase tracking-widest block">Teammate Component A</span>
            <div className="font-bold text-slate-200 text-sm mt-1">Academic Index</div>
            <p className="text-[11px] text-slate-400 mt-1">GPA trajectories, credit completion, repeating courses</p>
          </div>

          <div className="p-3 rounded-lg bg-navy-950/60 border border-navy-800 opacity-80">
            <span className="text-[10px] text-slate-400 uppercase tracking-widest block">Teammate Component B</span>
            <div className="font-bold text-slate-200 text-sm mt-1">Engagement Index</div>
            <p className="text-[11px] text-slate-400 mt-1">LMS activity, lecture attendance, forum interaction</p>
          </div>

          <div className="p-3 rounded-lg bg-navy-950/60 border border-navy-800 opacity-80">
            <span className="text-[10px] text-slate-400 uppercase tracking-widest block">Teammate Component C</span>
            <div className="font-bold text-slate-200 text-sm mt-1">Environmental Index</div>
            <p className="text-[11px] text-slate-400 mt-1">Commute distance, financial aid status, accommodation</p>
          </div>
        </div>

        <div className="mt-4 p-3 bg-indigo-950/20 border border-indigo-500/20 rounded-lg flex items-center justify-between text-xs text-indigo-200">
          <span>Downstream Fusion: Combined multi-modal attrition risk score for university administration</span>
          <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-indigo-500/20 border border-indigo-500/30">
            Strict Standard Export Contract
          </span>
        </div>
      </div>

      {/* Docs Browser Layout */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Document Selector Sidebar */}
        <div className="md:col-span-1 bg-navy-900 border border-navy-800 rounded-xl p-3 shadow-lg space-y-1">
          <div className="px-3 py-2 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
            Phase Reports
          </div>
          {docList.map((doc) => {
            const isSelected = selectedDoc === doc.filename;
            return (
              <button
                key={doc.filename}
                onClick={() => setSelectedDoc(doc.filename)}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center justify-between ${
                  isSelected
                    ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40'
                    : 'text-slate-300 hover:text-white hover:bg-navy-800'
                }`}
              >
                <div className="flex items-center gap-2 truncate">
                  <FileText className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                  <span className="truncate">{doc.title}</span>
                </div>
                {isSelected && <ChevronRight className="w-3.5 h-3.5 text-indigo-400 shrink-0" />}
              </button>
            );
          })}
        </div>

        {/* Document Markdown Viewer */}
        <div className="md:col-span-3 bg-navy-900 border border-navy-800 rounded-xl p-6 shadow-lg min-h-[500px]">
          {loading ? (
            <LoadingState message="Rendering viva documentation..." />
          ) : (
            <div className="prose prose-invert prose-indigo max-w-none text-xs leading-relaxed">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content}
              </ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
