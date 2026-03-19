import { useEffect, useState } from 'react';
import { Music, SmilePlus, X, Activity } from 'lucide-react';

function pct(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

export default function InterventionPopup({ results }) {
  const popupKey = results
    ? `${results.fused_emotion || 'neutral'}-${results.stress_label || 'moderate'}-${results.story || ''}`
    : '';
  const [dismissedKey, setDismissedKey] = useState('');

  useEffect(() => {
    if (!popupKey) return undefined;
    const timer = setTimeout(() => setDismissedKey(popupKey), 12000);
    return () => clearTimeout(timer);
  }, [popupKey]);

  if (!results || dismissedKey === popupKey) return null;

  const songs = Array.isArray(results.songs) ? results.songs.slice(0, 2) : [];
  const memes = Array.isArray(results.memes) ? results.memes.slice(0, 1) : [];

  return (
    <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 w-[min(92vw,860px)] animate-fade-up">
      <div className="glass shadow-2xl border border-primary/20 rounded-2xl overflow-hidden">
        <div className="flex items-start gap-4 p-4 sm:p-5">
          <div className="w-11 h-11 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
            <Activity className="w-5 h-5 text-primary" />
          </div>

          <div className="flex-1 min-w-0 space-y-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-primary">Stress Support Layer</p>
                <h3 className="text-sm sm:text-base font-black text-text-primary">
                  {results.stress_label ? `${results.stress_label[0].toUpperCase()}${results.stress_label.slice(1)} stress pattern detected` : 'New well-being suggestion ready'}
                </h3>
                <p className="text-xs text-text-secondary mt-1">
                  Current emotion trend: <span className="font-bold capitalize text-text-primary">{results.fused_emotion || 'neutral'}</span>
                  {' '}and estimated stress score <span className="font-bold text-text-primary">{pct(results.stress_score)}</span>.
                </p>
              </div>

              <button
                onClick={() => setDismissedKey(popupKey)}
                className="w-8 h-8 rounded-lg flex items-center justify-center text-text-muted hover:text-text-primary hover:bg-surface-raised transition-all cursor-pointer shrink-0"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="rounded-xl bg-surface-base border border-border-subtle p-3">
                <div className="flex items-center gap-2 mb-2">
                  <Music className="w-4 h-4 text-primary" />
                  <span className="text-[11px] font-bold uppercase tracking-widest text-text-primary">Top Songs</span>
                </div>
                {songs.length > 0 ? (
                  <div className="space-y-1.5">
                    {songs.map((song, index) => (
                      <p key={`${song.artist}-${song.title}-${index}`} className="text-xs text-text-secondary">
                        <span className="font-bold text-text-primary">{song.title}</span> by {song.artist}
                      </p>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-text-muted">Songs will appear here after support content is generated.</p>
                )}
              </div>

              <div className="rounded-xl bg-surface-base border border-border-subtle p-3">
                <div className="flex items-center gap-2 mb-2">
                  <SmilePlus className="w-4 h-4 text-primary" />
                  <span className="text-[11px] font-bold uppercase tracking-widest text-text-primary">Meme Relief</span>
                </div>
                {memes.length > 0 ? (
                  <div className="space-y-1.5">
                    <p className="text-[11px] font-bold text-primary uppercase tracking-widest">{memes[0].template}</p>
                    <p className="text-xs text-text-primary font-semibold">{memes[0].caption}</p>
                    <p className="text-[11px] text-text-secondary leading-relaxed">{memes[0].reason}</p>
                  </div>
                ) : (
                  <p className="text-xs text-text-muted">A light meme suggestion will appear here after analysis.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
