import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';

export interface TrajectoryPoint {
  semester: number | string;
  gpa?: number;
  attendance_pct?: number;
  exam_marks?: number;
  assignment_delay_days?: number;
  late_assignments_count?: number;
}

interface TrajectoryChartProps {
  data: TrajectoryPoint[];
  title?: string;
  height?: number;
  showProbationLine?: boolean;
  showAttendanceWarningLine?: boolean;
}

export const TrajectoryChart: React.FC<TrajectoryChartProps> = ({
  data,
  title,
  height = 320,
  showProbationLine = true,
  showAttendanceWarningLine = true,
}) => {
  const chartData = data.map(d => ({
    ...d,
    semLabel: typeof d.semester === 'number' ? `Sem ${d.semester}` : d.semester,
  }));

  return (
    <div className="w-full bg-slate-900/40 border border-slate-800 rounded-xl p-4">
      {title && (
        <div className="flex items-center justify-between mb-4 border-b border-slate-800/60 pb-2">
          <h4 className="text-sm font-semibold text-slate-200">{title}</h4>
          <span className="text-xs text-slate-400 font-mono">
            Longitudinal Sequence ({data.length} Semesters)
          </span>
        </div>
      )}

      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="semLabel"
              stroke="#64748b"
              fontSize={12}
              tickLine={false}
            />
            {/* Left Y Axis for GPA (0 - 4.0) */}
            <YAxis
              yAxisId="gpa"
              domain={[0.0, 4.0]}
              ticks={[0.0, 1.0, 2.0, 3.0, 4.0]}
              stroke="#818cf8"
              fontSize={11}
              tickLine={false}
              unit=" GPA"
            />
            {/* Right Y Axis for Percentages (0 - 100%) */}
            <YAxis
              yAxisId="pct"
              orientation="right"
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
              stroke="#38bdf8"
              fontSize={11}
              tickLine={false}
              unit="%"
            />

            <Tooltip
              contentStyle={{
                backgroundColor: '#0f172a',
                borderColor: '#334155',
                borderRadius: '8px',
                fontSize: '12px',
                color: '#f8fafc',
              }}
            />
            <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '8px' }} />

            {/* GPA Probation Threshold (2.0) */}
            {showProbationLine && (
              <ReferenceLine
                yAxisId="gpa"
                y={2.0}
                stroke="#f43f5e"
                strokeDasharray="4 4"
                label={{
                  value: 'Probation (GPA < 2.0)',
                  position: 'insideBottomLeft',
                  fill: '#f43f5e',
                  fontSize: 10,
                }}
              />
            )}

            {/* Attendance Exam Ineligible Threshold (65%) */}
            {showAttendanceWarningLine && (
              <ReferenceLine
                yAxisId="pct"
                y={65.0}
                stroke="#fbbf24"
                strokeDasharray="4 4"
                label={{
                  value: 'Exam Barred (< 65%)',
                  position: 'insideTopRight',
                  fill: '#fbbf24',
                  fontSize: 10,
                }}
              />
            )}

            <Line
              yAxisId="gpa"
              type="monotone"
              dataKey="gpa"
              name="Semester GPA"
              stroke="#818cf8"
              strokeWidth={3}
              dot={{ r: 4, fill: '#818cf8' }}
              activeDot={{ r: 6 }}
            />
            <Line
              yAxisId="pct"
              type="monotone"
              dataKey="attendance_pct"
              name="Attendance Rate"
              stroke="#38bdf8"
              strokeWidth={2}
              dot={{ r: 3, fill: '#38bdf8' }}
            />
            <Line
              yAxisId="pct"
              type="monotone"
              dataKey="exam_marks"
              name="Exam Marks Avg"
              stroke="#34d399"
              strokeWidth={2}
              dot={{ r: 3, fill: '#34d399' }}
            />
            <Line
              yAxisId="gpa"
              type="monotone"
              dataKey="assignment_delay_days"
              name="Delay (Days late)"
              stroke="#f87171"
              strokeWidth={1.5}
              strokeDasharray="2 2"
              dot={{ r: 3, fill: '#f87171' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
