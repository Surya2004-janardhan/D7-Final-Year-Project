import { Brain, MessageCircle } from 'lucide-react';

export default function Navbar({ chatOpen, onToggleChat }) {
  return (
    <nav className="fixed top-0 inset-x-0 z-50 glass" style={{ borderBottom: '1px solid rgba(213,207,47,0.1)', borderRadius: 0 }}>
      <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
        {/* Left */}
        <div className="flex items-center gap-3">
          <Brain className="w-6 h-6 text-wattle animate-float" strokeWidth={2} />
          <span className="text-lg font-semibold tracking-tight text-text-primary">
            Emotion<span className="text-wattle">AI</span>
          </span>
          <span className="text-xs font-medium text-text-muted px-2 py-0.5 rounded-full hidden sm:inline" style={{ background: 'rgba(122,42,61,0.5)' }}>
            Multimodal Analysis
          </span>
        </div>

        {/* Right */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-text-muted hidden sm:inline">Team Project — D7</span>
          <button
            onClick={onToggleChat}
            className={`p-2 rounded-lg transition-all cursor-pointer ${
              chatOpen ? 'bg-wattle/20 text-wattle' : 'text-text-secondary hover:text-wattle'
            }`}
            title="Toggle Chat"
          >
            <MessageCircle className="w-4.5 h-4.5" />
          </button>
        </div>
      </div>
    </nav>
  );
}
