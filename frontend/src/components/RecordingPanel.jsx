import { useState, useRef, useEffect } from 'react';
import { Upload, Video, Camera, StopCircle, RotateCcw } from 'lucide-react';

export default function RecordingPanel({
  recorder,
  onAnalyze,
  isProcessing,
}) {
  const [mode, setMode] = useState('upload'); // 'upload' | 'live'
  const [uploadedFile, setUploadedFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [recordingBlob, setRecordingBlob] = useState(null);
  const fileInputRef = useRef(null);

  const {
    isRecording,
    countdown,
    hasPermission,
    videoRef,
    requestPermission,
    startRecording,
    stopRecording,
    stopStream,
  } = recorder;

  useEffect(() => {
    if (mode === 'live') {
      requestPermission();
    } else {
      stopStream();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      setRecordingBlob(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setUploadedFile(file);
      setRecordingBlob(null);
    }
  };

  const handleRecord = async () => {
    if (isRecording) {
      stopRecording();
    } else {
      setRecordingBlob(null);
      const blob = await startRecording();
      if (blob) setRecordingBlob(blob);
    }
  };

  const handleAnalyze = () => {
    if (mode === 'upload' && uploadedFile) {
      onAnalyze(uploadedFile, 'file');
    } else if (mode === 'live' && recordingBlob) {
      onAnalyze(recordingBlob, 'blob');
    }
  };

  const canAnalyze = (mode === 'upload' && uploadedFile) || (mode === 'live' && recordingBlob && !isRecording);

  return (
    <div className="w-full max-w-2xl mx-auto animate-fade-up">
      {/* Mode Toggle */}
      <div className="flex justify-center mb-5">
        <div className="inline-flex glass rounded-xl p-1 gap-1">
          <button
            onClick={() => { setMode('upload'); setRecordingBlob(null); }}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-300 cursor-pointer ${
              mode === 'upload'
                ? 'bg-rajah/20 text-rajah'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <Upload className="w-4 h-4" />
            Upload
          </button>
          <button
            onClick={() => { setMode('live'); setUploadedFile(null); }}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-300 cursor-pointer ${
              mode === 'live'
                ? 'bg-rajah/20 text-rajah'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <Camera className="w-4 h-4" />
            Live Recording
          </button>
        </div>
      </div>

      {/* Upload Area */}
      {mode === 'upload' && (
        <div
          className={`glass glow-border rounded-2xl p-10 text-center transition-all duration-300 ${
            dragOver ? 'scale-[1.01]' : ''
          }`}
          style={dragOver ? { borderColor: 'rgba(251,176,110,0.5)', background: 'rgba(251,176,110,0.05)' } : {}}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          {uploadedFile ? (
            <div className="space-y-4">
              <div className="w-16 h-16 mx-auto rounded-2xl flex items-center justify-center" style={{ background: 'rgba(251,176,110,0.12)' }}>
                <Video className="w-8 h-8 text-rajah" />
              </div>
              <p className="text-text-primary font-medium text-base">{uploadedFile.name}</p>
              <p className="text-text-muted text-sm">
                {(uploadedFile.size / (1024 * 1024)).toFixed(1)} MB
              </p>
              <button
                onClick={() => { setUploadedFile(null); if (fileInputRef.current) fileInputRef.current.value = ''; }}
                className="inline-flex items-center gap-1.5 text-sm text-text-secondary hover:text-rajah transition-colors cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Change file
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="w-16 h-16 mx-auto rounded-2xl flex items-center justify-center" style={{ background: 'rgba(61,78,110,0.4)' }}>
                <Upload className="w-8 h-8 text-text-muted" />
              </div>
              <div>
                <p className="text-text-secondary text-base">
                  Drag & drop a video here, or{' '}
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="text-rajah underline underline-offset-2 hover:text-rajah-light transition-colors cursor-pointer"
                  >
                    browse
                  </button>
                </p>
                <p className="text-text-muted text-xs mt-1.5">MP4, WebM, AVI — up to 100 MB</p>
              </div>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={handleFileChange}
          />
        </div>
      )}

      {/* Live Recording Area */}
      {mode === 'live' && (
        <div className="glass glow-border rounded-2xl overflow-hidden">
          {/* Camera preview */}
          <div className="relative aspect-video bg-bluewood-dark flex items-center justify-center">
            {hasPermission ? (
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="text-center space-y-2 p-8">
                <Camera className="w-10 h-10 mx-auto text-text-muted" />
                <p className="text-text-secondary text-sm">Requesting camera access...</p>
              </div>
            )}

            {/* Countdown overlay */}
            {isRecording && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="relative">
                  <div className="absolute -inset-4 rounded-full animate-pulse-ring" style={{ border: '2px solid rgba(239,68,68,0.4)' }} />
                  <div className="w-20 h-20 rounded-full flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(8px)', border: '1px solid rgba(239,68,68,0.3)' }}>
                    <span className="text-3xl font-bold text-white tabular-nums">{countdown}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Recording indicator */}
            {isRecording && (
              <div className="absolute top-3 left-3 flex items-center gap-2 px-2.5 py-1 rounded-full" style={{ background: 'rgba(0,0,0,0.4)', backdropFilter: 'blur(8px)' }}>
                <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                <span className="text-xs font-medium text-white">REC</span>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="p-5 flex justify-center">
            {!isRecording ? (
              <button
                onClick={handleRecord}
                disabled={!hasPermission}
                className="flex items-center gap-2 px-6 py-3 rounded-xl text-rajah font-medium text-sm hover:bg-rajah/25 transition-all disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                style={{ background: 'rgba(251,176,110,0.1)', border: '1px solid rgba(251,176,110,0.25)' }}
              >
                <Camera className="w-4 h-4" />
                Start Recording
                <span className="text-text-muted text-xs ml-1">(11s)</span>
              </button>
            ) : (
              <button
                onClick={handleRecord}
                className="flex items-center gap-2 px-6 py-3 rounded-xl text-red-400 font-medium text-sm hover:bg-red-500/25 transition-all cursor-pointer"
                style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)' }}
              >
                <StopCircle className="w-4 h-4" />
                Stop & Analyze
              </button>
            )}
          </div>

          {/* Recorded confirmation */}
          {recordingBlob && !isRecording && (
            <div className="px-5 pb-4 text-center">
              <p className="text-sm text-em-happy flex items-center justify-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-em-happy inline-block" />
                Recording captured — ready to analyze
              </p>
            </div>
          )}
        </div>
      )}

      {/* Analyze Button */}
      <div className="mt-6 flex justify-center">
        <button
          onClick={handleAnalyze}
          disabled={!canAnalyze || isProcessing}
          className="px-10 py-3.5 rounded-xl font-semibold text-sm tracking-wide transition-all duration-300 cursor-pointer
            disabled:opacity-30 disabled:cursor-not-allowed
            bg-gradient-to-r from-rajah to-rajah-light text-bluewood-dark
            hover:shadow-lg hover:scale-[1.02]
            active:scale-[0.98]"
          style={{ boxShadow: canAnalyze ? '0 8px 24px rgba(251,176,110,0.2)' : 'none' }}
        >
          Analyze Emotions
        </button>
      </div>
    </div>
  );
}
