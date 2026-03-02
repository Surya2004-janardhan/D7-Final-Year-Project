import { Loader2 } from 'lucide-react';

const STATUS_MAP = {
  'Uploading...': 'Uploading video...',
  'Initializing': 'Initializing pipeline...',
  'Extracting Audio': 'Extracting audio track...',
  'Processing Video': 'Analyzing video frames...',
  'Processing Audio': 'Processing audio features...',
  'Cognitive Analysis': 'Running cognitive analysis...',
  'Generating AI Response': 'Generating AI recommendations...',
  'Complete': 'Analysis complete!',
};

export default function ProcessingLoader({ progress, status }) {
  const displayStatus = STATUS_MAP[status] || status || 'Processing...';

  let sectionLabel = '';
  if (progress <= 10) sectionLabel = 'Audio Extraction';
  else if (progress <= 40) sectionLabel = 'Video Processing';
  else if (progress <= 80) sectionLabel = 'Audio Analysis';
  else if (progress <= 90) sectionLabel = 'Cognitive Layer';
  else if (progress < 100) sectionLabel = 'AI Engine';
  else sectionLabel = 'Complete';

  return (
    <div className="w-full max-w-2xl mx-auto animate-fade-up mt-6">
      <div className="glass glow-border rounded-2xl p-6 space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <Loader2 className="w-5 h-5 text-rajah animate-spin" />
            <span className="text-sm font-medium text-text-secondary">{displayStatus}</span>
          </div>
          <span className="text-2xl font-bold text-rajah tabular-nums">{Math.round(progress)}%</span>
        </div>

        {/* Progress bar */}
        <div className="relative">
          <div className="h-3 rounded-full bg-bluewood-dark overflow-hidden">
            <div
              className="progress-bar-fill h-full rounded-full bg-gradient-to-r from-rajah-dark via-rajah to-rajah-light relative"
              style={{ width: `${progress}%` }}
            >
              <div className="absolute inset-0 animate-shimmer rounded-full" />
            </div>
          </div>

          {/* Section markers */}
          <div className="flex justify-between mt-2.5 px-0.5">
            {[
              { pct: 10, label: 'Extract' },
              { pct: 40, label: 'Video' },
              { pct: 80, label: 'Audio' },
              { pct: 90, label: 'Cognitive' },
              { pct: 100, label: 'AI' },
            ].map((m) => (
              <div key={m.pct} className="flex flex-col items-center">
                <div
                  className="w-1.5 h-1.5 rounded-full mb-1 transition-colors duration-500"
                  style={{ backgroundColor: progress >= m.pct ? '#FBB06E' : 'rgba(100,116,139,0.3)' }}
                />
                <span
                  className="text-[10px] font-medium transition-colors duration-500"
                  style={{ color: progress >= m.pct ? '#FBB06E' : 'rgba(100,116,139,0.5)' }}
                >
                  {m.label}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Current section */}
        <p className="text-center text-xs text-text-muted">
          Stage: <span className="text-text-secondary font-medium">{sectionLabel}</span>
        </p>
      </div>
    </div>
  );
}
