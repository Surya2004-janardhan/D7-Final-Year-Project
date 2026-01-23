import { useState, useRef } from "react";
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
  Smile,
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
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const emotions = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
  ];

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      videoRef.current.srcObject = stream;
      mediaRecorderRef.current = new MediaRecorder(stream);

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        await processVideo(blob);
        chunksRef.current = [];
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error starting recording:", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
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
  };

  const createChartData = (probs) => ({
    labels: probs.map((_, i) => `T${i + 1}`),
    datasets: emotions.map((emotion, idx) => ({
      label: emotion,
      data: probs.map((p) => p[idx]),
      borderColor: `hsl(${(idx * 360) / emotions.length}, 85%, 65%)`,
      backgroundColor: `hsl(${(idx * 360) / emotions.length}, 85%, 65%, 0.15)`,
      borderWidth: 2,
      tension: 0.4,
      pointRadius: 3,
      pointHoverRadius: 5,
      pointBackgroundColor: `hsl(${(idx * 360) / emotions.length}, 85%, 65%)`,
      pointBorderColor: "#0f172a",
      pointBorderWidth: 2,
    })),
  });

  return (
    <div className="min-h-screen bg-gradient-mesh">
      <div className="floating-orbs">
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
        <div className="orb orb-3"></div>
      </div>

      <div className="container mx-auto px-6 py-12 relative z-10 max-w-7xl">
        {" "}
        {/* Header */}
        <header className="text-center mb-16">
          <div className="flex items-center justify-center mb-4">
            <Smile className="w-10 h-10 text-accent-orange icon-smile mr-3" />
            <h1 className="text-6xl font-black tracking-tight gradient-text">
              EmotionAI
            </h1>
            <Sparkles className="w-10 h-10 text-accent-cyan icon-sparkle ml-3" />
          </div>
          <p className="text-xl text-text-muted max-w-2xl mx-auto text-center">
            Advanced multimodal emotion recognition powered by deep learning
          </p>
          <div className="flex items-centser justify-center gap-2 mt-4">
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
                  Supports MP4, WebM, AVI formats • Max 100MB • Best with clear
                  facial expressions
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
                  <h2 className="text-xl font-bold text-white">Live Record</h2>
                </div>
                {isRecording && (
                  <div className="recording-indicator">
                    <div className="recording-dot"></div>
                    <span>REC</span>
                  </div>
                )}
              </div>

              <div className="space-y-4">
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
        {/* Processing Indicator */}
        {isProcessing && (
          <div className="glass-card processing-card mb-8">
            <div className="flex items-center justify-center gap-4">
              <Loader2 className="w-10 h-10 animate-spin text-accent-orange icon-rotate" />
              <div>
                <h3 className="text-xl font-semibold text-white">
                  Processing your emotions
                </h3>
                <p className="text-sm text-text-muted">
                  Analyzing facial expressions and voice patterns...
                </p>
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
                    <strong className="text-white">Cognitive Analysis:</strong>
                    <p className="text-text-muted mt-1">{results.reasoning}</p>
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
                  <p className="content-text">{results.story}</p>
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
                  <p className="content-text">{results.video}</p>
                </div>

                <div className="content-card">
                  <div className="content-header">
                    <Music className="w-5 h-5 text-accent-orange" />
                    <h4 className="text-lg font-semibold text-white">
                      Music Recommendation
                    </h4>
                  </div>
                  <p className="content-text">
                    {results.song ||
                      "Uplifting music to enhance your current mood"}
                  </p>
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
    </div>
  );
}

export default App;
