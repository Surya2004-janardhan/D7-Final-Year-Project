import { useState, useRef, useEffect } from 'react';
import { X, Send, MessageCircle, Bot } from 'lucide-react';
import axios from 'axios';

export default function Chatbot({ results, isOpen, onClose }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hi! I\'m your EmotionAI assistant. Ask me anything about your analysis results.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async () => {
    if (!input.trim() || loading) return;

    const userMsg = { role: 'user', content: input.trim() };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput('');
    setLoading(true);

    try {
      // Send last 10 messages as history (excluding the system greeting)
      const history = updated
        .filter((m, i) => i > 0) // skip initial greeting
        .slice(-10);

      const res = await axios.post('/chat', {
        message: userMsg.content,
        context: results || {},
        history,
      });

      setMessages((prev) => [...prev, { role: 'assistant', content: res.data.reply || 'No response.' }]);
    } catch {
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Sorry, couldn\'t reach the server.' }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 w-80 sm:w-96 flex flex-col animate-fade-up" style={{ maxHeight: 'calc(100vh - 6rem)' }}>
      <div className="glass glow-border rounded-2xl flex flex-col overflow-hidden" style={{ maxHeight: 'calc(100vh - 6rem)', background: 'rgba(94,21,37,0.85)' }}>
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: '1px solid rgba(213,207,47,0.1)' }}>
          <div className="flex items-center gap-2">
            <Bot className="w-4 h-4 text-wattle" />
            <span className="text-sm font-semibold text-text-primary">EmotionAI Chat</span>
          </div>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary transition-colors cursor-pointer">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3" style={{ maxHeight: '400px' }}>
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[85%] rounded-xl px-3 py-2 text-xs leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-wattle/15 text-text-primary'
                    : 'text-text-secondary'
                }`}
                style={msg.role === 'assistant' ? { background: 'rgba(122,42,61,0.5)' } : {}}
              >
                {msg.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="rounded-xl px-3 py-2 text-xs text-text-muted" style={{ background: 'rgba(122,42,61,0.5)' }}>
                <span className="animate-pulse">Thinking...</span>
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        {/* Input */}
        <div className="px-3 pb-3 pt-1">
          <div className="flex items-center gap-2 rounded-xl px-3 py-2" style={{ background: 'rgba(61,14,24,0.6)', border: '1px solid rgba(213,207,47,0.1)' }}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about your results..."
              className="flex-1 bg-transparent text-xs text-text-primary placeholder-text-muted outline-none"
            />
            <button
              onClick={send}
              disabled={!input.trim() || loading}
              className="text-wattle disabled:opacity-30 hover:text-wattle-light transition-colors cursor-pointer"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
