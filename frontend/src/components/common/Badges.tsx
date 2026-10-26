import React from 'react';
import { Sparkles, Database } from 'lucide-react';

interface SyntheticBadgeProps {
  label?: string;
  className?: string;
}

export const SyntheticBadge: React.FC<SyntheticBadgeProps> = ({
  label = "SYNTHETIC DATA",
  className = "",
}) => {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-indigo-500/10 text-indigo-300 border border-indigo-500/30 tracking-wide ${className}`}
      title="This dataset was probabilistically synthesized using latent factor copulas to protect student privacy."
    >
      <Database className="w-3 h-3 text-indigo-400" />
      {label}
    </span>
  );
};

export const ResearchPrototypeBadge: React.FC = () => {
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[10px] font-semibold bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 tracking-wider uppercase"
      title="SLIIT Kandy Research Component — CDAP IT05"
    >
      <Sparkles className="w-3 h-3 text-cyan-400" />
      Research Prototype
    </span>
  );
};
