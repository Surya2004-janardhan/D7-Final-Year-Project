import { useState, useRef, useEffect } from 'react';
import { Sparkles, BookOpen, Quote, Play, Pause, Music, BookMarked, ExternalLink } from 'lucide-react';
import axios from 'axios';

/* ── YouTube embed helper ─────────────────────────────── */
function extractYTId(url) {
  if (!url) return null;
  const m = url.match(/(?:youtu\.be\/|youtube\.com\/(?:watch\?v=|embed\/))([a-zA-Z0-9_-]{11})/);
  return m ? m[1] : null;
}

function YouTubeEmbed({ url, title }) {
  const id = extractYTId(url);
  if (!id) return null;
  return (
    <div className="rounded-xl overflow-hidden" style={{ aspectRatio: '16/9' }}>
      <iframe
        src={`https://www.youtube.com/embed/${id}`}
        title={title || 'YouTube video'}
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowFullScreen
        className="w-full h-full border-0"
      />
    </div>
  );
}

/* ── Deezer song player ───────────────────────────────── */
function SongPlayer({ artist, title, link }) {
  const [track, setTrack] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const audioRef = useRef(null);

  const search = async () => {
    if (searched) { togglePlay(); return; }
    setLoading(true);
    try {
      const q = `${artist} ${title}`.trim();
      const res = await axios.get(`/music/search?q=${encodeURIComponent(q)}`);
      if (res.data.preview) {
        setTrack(res.data);
        setSearched(true);
        // Auto-play after search
        setTimeout(() => {
          if (audioRef.current) {
            audioRef.current.src = res.data.preview;
            audioRef.current.play();
            setPlaying(true);
          }
        }, 100);
      }
    } catch { /* silent */ }
    setLoading(false);
  };

  const togglePlay = () => {
    if (!audioRef.current || !track) return;
    if (playing) { audioRef.current.pause(); setPlaying(false); }
    else { audioRef.current.play(); setPlaying(true); }
  };

  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    const onTime = () => setProgress(a.duration ? (a.currentTime / a.duration) * 100 : 0);
    const onEnd = () => { setPlaying(false); setProgress(0); };
    a.addEventListener('timeupdate', onTime);
    a.addEventListener('ended', onEnd);
    return () => { a.removeEventListener('timeupdate', onTime); a.removeEventListener('ended', onEnd); };
  }, [track]);

  return (
    <div className="flex items-center gap-3 group">
      {/* Album art or play button */}
      <button
        onClick={search}
        disabled={loading}
        className="w-10 h-10 rounded-lg shrink-0 flex items-center justify-center overflow-hidden cursor-pointer transition-all hover:scale-105"
        style={{
          background: track?.album_art ? `url(${track.album_art}) center/cover` : 'rgba(213,207,47,0.08)',
          border: '1px solid rgba(213,207,47,0.12)',
        }}
      >
        {loading ? (
          <div className="w-3 h-3 border-2 border-wattle border-t-transparent rounded-full animate-spin" />
        ) : (
          <div className="w-full h-full flex items-center justify-center" style={track?.album_art ? { background: 'rgba(0,0,0,0.35)' } : {}}>
            {playing ? <Pause className="w-3.5 h-3.5 text-wattle" /> : <Play className="w-3.5 h-3.5 text-wattle" />}
          </div>
        )}
      </button>

      {/* Track info + progress */}
      <div className="flex-1 min-w-0">
        <p className="text-sm text-text-primary font-medium truncate">
          {title} <span className="text-text-muted font-normal">— {artist}</span>
        </p>
        {track ? (
          <div className="mt-1 h-1 rounded-full bg-cherry-dark overflow-hidden">
            <div className="h-full rounded-full bg-wattle transition-all" style={{ width: `${progress}%` }} />
          </div>
        ) : (
          <p className="text-[10px] text-text-muted mt-0.5">Click to play preview</p>
        )}
      </div>

      {/* External link */}
      {link && (
        <a href={link} target="_blank" rel="noopener noreferrer" className="text-wattle/40 hover:text-wattle transition-colors shrink-0">
          <ExternalLink className="w-3 h-3" />
        </a>
      )}

      <audio ref={audioRef} preload="none" />
    </div>
  );
}

/* ── Main component ───────────────────────────────────── */
export default function AIContent({ results }) {
  if (!results) return null;
  const { story, quote, video, books, songs } = results;

  const videoObj = typeof video === 'object' && video !== null
    ? video
    : typeof video === 'string'
      ? { title: video, link: video.match(/https?:\/\/[^\s"]+/)?.[0] || '#' }
      : null;

  return (
    <div className="max-w-4xl mx-auto space-y-5 animate-fade-up" style={{ animationDelay: '0.25s' }}>
      <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
        <Sparkles className="w-4 h-4 text-wattle" />
        AI Recommendations Engine
      </h3>

      {/* Row 1: Story + Books */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        {story && (
          <div className="lg:col-span-3 glass glow-border rounded-2xl p-5 space-y-3">
            <div className="flex items-center gap-2 text-wattle">
              <BookOpen className="w-4 h-4" />
              <span className="text-xs font-semibold uppercase tracking-wider">Read Story</span>
            </div>
            <p className="text-sm text-text-secondary leading-relaxed">{story}</p>
          </div>
        )}
        {books && books.length > 0 && (
          <div className="lg:col-span-2 glass glow-border rounded-2xl p-5 space-y-3">
            <div className="flex items-center gap-2 text-wattle">
              <BookMarked className="w-4 h-4" />
              <span className="text-xs font-semibold uppercase tracking-wider">Short Reads</span>
            </div>
            <div className="space-y-3">
              {books.map((book, i) => (
                <div key={i} className="flex items-start gap-2.5">
                  <span className="text-lg mt-0.5">📖</span>
                  <div className="min-w-0">
                    <p className="text-sm text-text-primary font-medium">{book.title || 'Unknown'}</p>
                    <p className="text-xs text-text-muted">by {book.author || 'Unknown'}</p>
                    {book.reason && <p className="text-xs text-text-secondary mt-0.5">{book.reason}</p>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Row 2: Quote + Video */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {quote && (
          <div className="glass glow-border rounded-2xl p-5 flex flex-col justify-center space-y-2">
            <Quote className="w-5 h-5 text-wattle/30" />
            <blockquote className="text-sm text-text-primary italic leading-relaxed pl-3" style={{ borderLeft: '2px solid rgba(213,207,47,0.25)' }}>
              {quote}
            </blockquote>
          </div>
        )}
        {videoObj && (
          <div className="glass glow-border rounded-2xl p-5 space-y-3">
            <div className="flex items-center gap-2 text-wattle">
              <Play className="w-4 h-4" />
              <span className="text-xs font-semibold uppercase tracking-wider">Watch This</span>
            </div>
            {extractYTId(videoObj.link) && <YouTubeEmbed url={videoObj.link} title={videoObj.title} />}
            <div className="space-y-1">
              {videoObj.title && <p className="text-sm text-text-primary font-medium">{videoObj.title}</p>}
              {videoObj.channel && <p className="text-[11px] text-text-muted">by {videoObj.channel}</p>}
              {videoObj.reason && <p className="text-xs text-text-secondary leading-relaxed">{videoObj.reason}</p>}
            </div>
          </div>
        )}
      </div>

      {/* Row 3: Songs with Deezer preview player */}
      {songs && songs.length > 0 && (
        <div className="glass glow-border rounded-2xl p-5 space-y-4">
          <div className="flex items-center gap-2 text-wattle">
            <Music className="w-4 h-4" />
            <span className="text-xs font-semibold uppercase tracking-wider">Playlist</span>
            <span className="text-[9px] text-text-muted ml-auto">Powered by Deezer</span>
          </div>
          <div className="space-y-3">
            {songs.map((song, i) => (
              <SongPlayer key={i} artist={song.artist} title={song.title} link={song.link} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
