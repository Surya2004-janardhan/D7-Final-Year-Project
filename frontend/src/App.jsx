import React, { useState, useRef, useEffect } from "react";
// import React, { useState, useRef, useEffect } from "react";

// Stop recording when countdown hits 0 (in case setTimeout fails or is out of sync)
// This must come after React and useEffect are imported

// ErrorBoundary component to catch runtime errors
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    // You can log errorInfo here if needed
  }
  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            color: "white",
            background: "#1a1a1a",
            padding: 32,
            minHeight: "100vh",
          }}
        >
          <h1 style={{ color: "#ff6b00" }}>Something went wrong.</h1>
          <p>{this.state.error?.toString()}</p>
          <p>Please try refreshing the page or re-uploading your file.</p>
        </div>
      );
    }
    return this.props.children;
  }
}
import {
  Upload,
  Video,
  Play,
  Pause,
  Loader2,
  Heart,
  Brain,
  Music,
  BookOpen,
  Youtube,
  Sparkles,
  Zap,
  TrendingUp,
  Activity,
  X,
} from "lucide-react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
);

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [modalContent, setModalContent] = useState(null);
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  // Match backend emotion order (7 emotions)
  const emotions = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
  ];

  // Loader state for time-based progress
  const [loaderPercent, setLoaderPercent] = useState(0);
  const [showLoader, setShowLoader] = useState(false);
  const [showProcessing, setShowProcessing] = useState(false);

  // Time-based loader effect
  useEffect(() => {
    let interval;
    if (isProcessing) {
      setShowLoader(true);
      setShowProcessing(false);
      setLoaderPercent(0);
      const start = Date.now();
      interval = setInterval(() => {
        const elapsed = (Date.now() - start) / 1000;
        if (elapsed < 22) {
          // Go up to 95% in 22 seconds
          setLoaderPercent(Math.min(95, Math.floor((elapsed / 22) * 95)));
        } else {
          setLoaderPercent(95);
          setShowProcessing(true);
          clearInterval(interval);
        }
      }, 100);
    } else {
      setShowLoader(false);
      setShowProcessing(false);
      setLoaderPercent(0);
    }
    return () => clearInterval(interval);
  }, [isProcessing]);

  const [recordError, setRecordError] = useState("");
  const recordTimerRef = useRef(null);
  const [recordingStartTime, setRecordingStartTime] = useState(null);
  const [recordingStopped, setRecordingStopped] = useState(false);
  const [recordCountdown, setRecordCountdown] = useState(11);

  // Countdown effect for recording
  useEffect(() => {
    let countdownInterval;
    if (isRecording) {
      setRecordCountdown(11);
      countdownInterval = setInterval(() => {
        setRecordCountdown((prev) => {
          if (prev <= 1) {
            clearInterval(countdownInterval);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } else {
      setRecordCountdown(11);
    }
    return () => clearInterval(countdownInterval);
  }, [isRecording]);

  // Auto-stop recording when countdown hits 0
  useEffect(() => {
    if (isRecording && recordCountdown === 0) {
      stopRecording();
    }
    // eslint-disable-next-line
  }, [recordCountdown]);

  const startRecording = async () => {
    setRecordError("");
    setRecordingStopped(false);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setRecordError("Your browser does not support video/audio recording.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      videoRef.current.srcObject = stream;
      videoRef.current.muted = false; // allow playback feedback
      mediaRecorderRef.current = new MediaRecorder(stream);

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        if (!recordingStopped) {
          setRecordingStopped(true);
          const blob = new Blob(chunksRef.current, { type: "video/webm" });
          await processVideo(blob);
          chunksRef.current = [];
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingStartTime(Date.now());
      // Auto-stop after 11 seconds
      recordTimerRef.current = setTimeout(() => {
        if (mediaRecorderRef.current && isRecording) {
          mediaRecorderRef.current.stop();
          setIsRecording(false);
          if (videoRef.current.srcObject) {
            videoRef.current.srcObject
              .getTracks()
              .forEach((track) => track.stop());
          }
        }
      }, 11000);
    } catch (error) {
      setRecordError(
        "Could not access camera or microphone. Please check permissions and devices.",
      );
      console.error("Error starting recording:", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      clearTimeout(recordTimerRef.current);
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      }
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(file);
    }
  };

  const processUploadedVideo = async () => {
    if (uploadedFile) {
      await processVideo(uploadedFile);
    }
  };

  const processVideo = async (videoBlob) => {
    setIsProcessing(true);
    const formData = new FormData();
    formData.append("video", videoBlob, "video.webm");
    const audioBlob = new Blob([], { type: "audio/webm" });
    formData.append("audio", audioBlob, "audio.webm");

    try {
      const response = await fetch("/api/process", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Error processing video:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        position: "top",
        labels: {
          color: "#e0e7ff",
          font: {
            size: 11,
            weight: "500",
          },
          padding: 12,
          usePointStyle: true,
          pointStyle: "circle",
        },
      },
      title: {
        display: true,
        text: "Emotion Probabilities Over Time",
        color: "#fff",
        font: {
          size: 16,
          weight: "600",
        },
        padding: 20,
      },
      tooltip: {
        backgroundColor: "rgba(15, 23, 42, 0.95)",
        titleColor: "#fff",
        bodyColor: "#e0e7ff",
        borderColor: "rgba(255, 107, 0, 0.3)",
        borderWidth: 1,
        padding: 12,
        displayColors: true,
        callbacks: {
          label: function (context) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(3)}`;
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        grid: {
          color: "rgba(255, 255, 255, 0.1)",
          drawBorder: false,
        },
        ticks: {
          color: "#e0e7ff",
          font: {
            size: 11,
          },
        },
      },
      x: {
        grid: {
          color: "rgba(255, 255, 255, 0.05)",
          drawBorder: false,
        },
        ticks: {
          color: "#e0e7ff",
          font: {
            size: 11,
          },
        },
      },
    },
    elements: {
      point: {
        radius: 3,
        hoverRadius: 5,
        borderWidth: 2,
        hoverBorderWidth: 3,
      },
      line: {
        borderWidth: 3,
        fill: false,
        tension: 0.8, // Smooth curves instead of straight lines
        cubicInterpolationMode: "monotone",
      },
    },
    interaction: {
      intersect: false,
      mode: "index",
    },
    animation: {
      duration: 2000,
      easing: "easeInOutQuart",
    },
  };

  const createChartData = (probs) => ({
    labels: probs.map((_, i) => `T${i + 1}`),
    datasets: emotions.map((emotion, idx) => ({
      label: emotion,
      data: probs.map((p) => p[idx]),
      borderColor: `hsl(${(idx * 360) / emotions.length}, 85%, 55%)`,
      backgroundColor: `hsla(${(idx * 360) / emotions.length}, 85%, 55%, 0.25)`,
      borderWidth: 3,
      fill: true,
      tension: 0.5,
      pointRadius: 4,
      pointHoverRadius: 7,
      pointBackgroundColor: `hsl(${(idx * 360) / emotions.length}, 85%, 55%)`,
      pointBorderColor: "#0f172a",
      pointBorderWidth: 2,
    })),
  });

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gradient-mesh">
        <div className="floating-orbs">
          <div className="orb orb-1"></div>
          <div className="orb orb-2"></div>
          <div className="orb orb-3"></div>
        </div>

        <div className="max-w-7xl mx-auto px-8 py-12 relative z-10">
          {/* Header */}
          <header className="text-center mb-16">
            <div className="flex items-center justify-center mb-4">
              <Sparkles className="w-10 h-10 text-accent-orange icon-sparkle mr-3" />
              <h1 className="text-6xl font-black tracking-tight gradient-text">
                EmotionAI
              </h1>
              <Sparkles className="w-10 h-10 text-accent-cyan icon-sparkle ml-3" />
            </div>
            <p className="text-xl text-text-muted max-w-2xl mx-auto text-center">
              Advanced multimodal emotion recognition powered by deep learning
            </p>
            <div className="flex items-center justify-center gap-2 mt-4">
              <div className="status-badge">
                <Activity className="w-4 h-4 icon-pulse" />
                <span>Live Analysis</span>
              </div>
              <div className="status-badge">
                <Zap className="w-4 h-4 icon-zap" />
                <span>AI Powered</span>
              </div>
            </div>
          </header>

          {/* Main Content Grid */}
          <div className="grid lg:grid-cols-3 gap-6 mb-8">
            {/* Upload Section - 2 columns */}
            <div className="lg:col-span-2">
              <div className="glass-card">
                <div className="card-header">
                  <div className="flex items-center gap-3">
                    <div className="icon-wrapper">
                      <Upload className="w-6 h-6 icon-float" />
                    </div>
                    <h2 className="text-2xl font-bold text-white">
                      Upload & Analyze
                    </h2>
                  </div>
                  <TrendingUp className="w-5 h-5 text-accent-cyan icon-trend" />
                </div>

                <div className="upload-zone">
                  <div className="upload-zone-inner">
                    <Video className="w-16 h-16 text-accent-orange mb-4 mx-auto icon-float" />
                    <h3 className="text-lg font-semibold text-white mb-2">
                      Drop your video here
                    </h3>
                    <p className="text-sm text-text-muted mb-4">
                      or click to browse files
                    </p>
                    <input
                      type="file"
                      accept="video/*"
                      onChange={handleFileUpload}
                      className="file-input"
                      id="file-upload"
                    />
                    <label htmlFor="file-upload" className="file-label">
                      Choose File
                    </label>
                    {uploadedFile && (
                      <div className="mt-4 text-sm text-accent-cyan">
                        ✓ {uploadedFile.name}
                      </div>
                    )}
                  </div>
                </div>

                <button
                  onClick={processUploadedVideo}
                  disabled={!uploadedFile || isProcessing}
                  className="btn-primary w-full mt-6"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin icon-pulse" />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5 icon-bounce" />
                      <span>Start Analysis</span>
                    </>
                  )}
                </button>

                <div className="info-box mt-4">
                  <Sparkles className="w-4 h-4 text-accent-cyan" />
                  <p className="text-xs text-text-muted">
                    Supports MP4, WebM, AVI formats • Max 100MB • Best with
                    clear facial expressions
                  </p>
                </div>
              </div>
            </div>

            {/* Live Recording Section - 1 column */}
            <div className="lg:col-span-1">
              <div className="glass-card h-full">
                <div className="card-header">
                  <div className="flex items-center gap-3">
                    <div className="icon-wrapper">
                      <Video className="w-5 h-5 icon-wiggle" />
                    </div>
                    <h2 className="text-xl font-bold text-white">
                      Live Record
                    </h2>
                  </div>
                  {isRecording && (
                    <div className="recording-indicator">
                      <div className="recording-dot"></div>
                      <span>REC</span>
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  {isRecording && (
                    <div className="text-center text-lg font-bold text-accent-orange mb-2">
                      Recording: <span>{recordCountdown}</span>s
                    </div>
                  )}
                  {recordError && (
                    <div className="text-red-400 text-sm font-semibold mb-2">
                      {recordError}
                    </div>
                  )}
                  <div className="video-container">
                    <video
                      ref={videoRef}
                      className="video-preview"
                      autoPlay
                      muted
                    />
                    {!isRecording && (
                      <div className="video-overlay">
                        <Video className="w-12 h-12 text-white/50" />
                      </div>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={startRecording}
                      disabled={isRecording || isProcessing}
                      className="btn-secondary"
                    >
                      <Play className="w-4 h-4" />
                      <span>Start</span>
                    </button>
                    <button
                      onClick={stopRecording}
                      disabled={!isRecording}
                      className="btn-secondary-alt"
                    >
                      <Pause className="w-4 h-4" />
                      <span>Stop</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Time-based Loader & Processing Indicator */}
          {showLoader && (
            <div className="glass-card processing-card mb-8">
              <div className="flex flex-col items-center justify-center gap-4">
                <Loader2 className="w-10 h-10 animate-spin text-accent-orange icon-rotate" />
                <div className="text-center">
                  <h3 className="text-xl font-semibold text-white">
                    {showProcessing
                      ? "Processing..."
                      : `Loading... ${loaderPercent}%`}
                  </h3>
                  <p className="text-sm text-text-muted">
                    {showProcessing
                      ? "Analyzing facial expressions and voice patterns..."
                      : "Preparing your analysis. Please wait."}
                  </p>
                  {!showProcessing && (
                    <div className="w-64 h-3 bg-white/10 rounded-full mt-4 mx-auto">
                      <div
                        className="h-3 bg-accent-orange rounded-full transition-all duration-300"
                        style={{ width: `${loaderPercent}%` }}
                      ></div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Results Section */}
          {results && !results.error && (
            <div className="space-y-6 results-appear">
              {/* Emotion Results */}
              <div className="glass-card">
                <div className="card-header">
                  <div className="flex items-center gap-3">
                    <div className="icon-wrapper">
                      <Heart className="w-6 h-6 icon-pulse" />
                    </div>
                    <h2 className="text-2xl font-bold text-white">
                      Emotion Analysis
                    </h2>
                  </div>
                  <Sparkles className="w-5 h-5 text-accent-orange icon-sparkle" />
                </div>

                <div className="grid md:grid-cols-3 gap-6 mb-6">
                  <div className="emotion-card">
                    <div className="emotion-icon">
                      <Music className="w-8 h-8 icon-wiggle" />
                    </div>
                    <div className="emotion-label">Audio Emotion</div>
                    <div className="emotion-value emotion-value-1">
                      {results.audio_emotion}
                    </div>
                  </div>

                  <div className="emotion-card">
                    <div className="emotion-icon">
                      <Video className="w-8 h-8 icon-float" />
                    </div>
                    <div className="emotion-label">Video Emotion</div>
                    <div className="emotion-value emotion-value-2">
                      {results.video_emotion}
                    </div>
                  </div>

                  <div className="emotion-card emotion-card-primary">
                    <div className="emotion-icon">
                      <Brain className="w-8 h-8 icon-bounce" />
                    </div>
                    <div className="emotion-label">Fused Result</div>
                    <div className="emotion-value emotion-value-3">
                      {results.fused_emotion}
                    </div>
                  </div>
                </div>

                <div className="reasoning-box">
                  <div className="flex items-start gap-3">
                    <Brain className="w-5 h-5 text-accent-cyan mt-1 flex-shrink-0" />
                    <div>
                      <strong className="text-white">
                        Cognitive Analysis:
                      </strong>
                      <p className="text-text-muted mt-1">
                        {results.reasoning}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Temporal Charts */}
              <div className="glass-card">
                <div className="card-header">
                  <div className="flex items-center gap-3">
                    <div className="icon-wrapper">
                      <TrendingUp className="w-6 h-6 icon-trend" />
                    </div>
                    <h2 className="text-2xl font-bold text-white">
                      Temporal Analysis
                    </h2>
                  </div>
                  <Activity className="w-5 h-5 text-accent-cyan icon-pulse" />
                </div>

                <div className="grid md:grid-cols-2 gap-8">
                  <div className="chart-container">
                    <h4 className="chart-title">
                      <Music className="w-5 h-5 icon-wiggle" />
                      Audio Timeline
                    </h4>
                    <Line
                      data={createChartData(results.audio_probs_temporal)}
                      options={chartOptions}
                    />
                  </div>
                  <div className="chart-container">
                    <h4 className="chart-title">
                      <Video className="w-5 h-5 icon-float" />
                      Video Timeline
                    </h4>
                    <Line
                      data={createChartData(results.video_probs_temporal)}
                      options={chartOptions}
                    />
                  </div>
                </div>
              </div>

              {/* AI-Generated Content */}
              <div className="glass-card">
                <div className="card-header">
                  <div className="flex items-center gap-3">
                    <div className="icon-wrapper">
                      <Sparkles className="w-6 h-6 icon-sparkle" />
                    </div>
                    <h2 className="text-2xl font-bold text-white">
                      AI Recommendations
                    </h2>
                  </div>
                  <Zap className="w-5 h-5 text-accent-orange icon-zap" />
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                  <div className="content-card">
                    <div className="content-header">
                      <BookOpen className="w-5 h-5 text-accent-cyan" />
                      <h4 className="text-lg font-semibold text-white">
                        Personalized Story
                      </h4>
                    </div>
                    <div className="content-text">
                      {results.story.split("\n").slice(0, 3).join("\n")}
                      {results.story.split("\n").length > 3 && (
                        <button
                          onClick={() => {
                            setModalContent({
                              title: "Personalized Story",
                              content: results.story,
                              icon: BookOpen,
                            });
                            setShowModal(true);
                          }}
                          className="text-accent-cyan hover:text-accent-cyan/80 ml-2 underline text-sm"
                        >
                          read more
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="content-card">
                    <div className="content-header">
                      <Sparkles className="w-5 h-5 text-accent-orange" />
                      <h4 className="text-lg font-semibold text-white">
                        Inspirational Quote
                      </h4>
                    </div>
                    <p className="content-text italic">"{results.quote}"</p>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-6 mt-6">
                  <div className="content-card">
                    <div className="content-header">
                      <Youtube className="w-5 h-5 text-accent-cyan" />
                      <h4 className="text-lg font-semibold text-white">
                        Video Suggestion
                      </h4>
                    </div>
                    <div className="content-text">
                      {results.video && (
                        <div>
                          {/* Extract YouTube link if present */}
                          {results.video &&
                            results.video.match(
                              /https:\/\/www\.youtube\.com\/watch\?v=[\w-]+/,
                            ) && (
                              <div className="mb-2">
                                <a
                                  href={
                                    results.video.match(
                                      /https:\/\/www\.youtube\.com\/watch\?v=[\w-]+/,
                                    )[0]
                                  }
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-2 text-accent-cyan hover:text-accent-cyan/80 underline"
                                >
                                  <Youtube className="w-4 h-4" />
                                  Watch Video
                                </a>
                              </div>
                            )}
                          <p>
                            {results.video &&
                              results.video
                                .replace(
                                  /https:\/\/www\.youtube\.com\/watch\?v=[\w-]+/,
                                  "",
                                )
                                .trim()}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="content-card">
                    <div className="content-header">
                      <Music className="w-5 h-5 text-accent-orange" />
                      <h4 className="text-lg font-semibold text-white">
                        Music Recommendations
                      </h4>
                    </div>
                    <div className="space-y-3">
                      {Array.isArray(results.songs) &&
                      results.songs.length > 0 ? (
                        results.songs.slice(0, 3).map((song, index) => (
                          <div key={index} className="song-item">
                            <div className="flex items-center gap-2 mb-1">
                              <Music className="w-4 h-4 text-accent-orange" />
                              <span className="font-medium text-white">
                                {song.artist} - {song.title}
                              </span>
                            </div>
                            <p className="text-sm text-text-muted mb-2">
                              {song.explanation}
                            </p>
                            {song.link && (
                              <a
                                href={song.link}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 text-accent-cyan hover:text-accent-cyan/80 text-sm underline"
                              >
                                <Youtube className="w-3 h-3" />
                                Listen now
                              </a>
                            )}
                          </div>
                        ))
                      ) : (
                        <p className="content-text">
                          Uplifting music to enhance your current mood
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Error Display */}
          {results && results.error && (
            <div className="error-card">
              <div className="flex items-center gap-3">
                <Zap className="w-6 h-6 text-red-400" />
                <h3 className="text-xl font-semibold text-white">
                  Analysis Error
                </h3>
              </div>
              <p className="text-red-200 mt-2">{results.error}</p>
            </div>
          )}

          {/* Footer */}
          <footer className="text-center mt-16 pt-8 border-t border-white/10">
            <p className="text-text-muted text-sm">
              Powered by advanced deep learning • Real-time multimodal emotion
              recognition
            </p>
          </footer>
        </div>

        {/* Modal */}
        {showModal && modalContent && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="glass-card max-w-2xl w-full max-h-[80vh] overflow-y-auto">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="icon-wrapper">
                    <modalContent.icon className="w-6 h-6 text-accent-cyan" />
                  </div>
                  <h2 className="text-2xl font-bold text-white">
                    {modalContent.title}
                  </h2>
                </div>
                <button
                  onClick={() => setShowModal(false)}
                  className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                >
                  <X className="w-6 h-6 text-white" />
                </button>
              </div>
              <div className="content-text whitespace-pre-line">
                {modalContent.content}
              </div>
            </div>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
}

export default App;
