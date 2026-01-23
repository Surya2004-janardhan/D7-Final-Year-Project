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

    // For uploaded files, we need to extract audio and video separately
    // For simplicity, we'll send the video and let backend handle extraction
    formData.append("video", videoBlob, "video.webm");

    // Create a dummy audio blob for now (backend will extract from video)
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
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "Emotion Probabilities Over Time",
      },
      tooltip: {
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
      },
    },
  };

  const createChartData = (probs) => ({
    labels: probs.map((_, i) => `T${i + 1}`),
    datasets: emotions.map((emotion, idx) => ({
      label: emotion,
      data: probs.map((p) => p[idx]),
      borderColor: `hsl(${(idx * 360) / emotions.length}, 70%, 50%)`,
      backgroundColor: `hsl(${(idx * 360) / emotions.length}, 70%, 50%, 0.1)`,
      tension: 0.1,
    })),
  });

  return (
    <div
      className="min-h-screen"
      style={{
        background: "linear-gradient(135deg, #51e2f5, #9df9ef, #edf756)",
      }}
    >
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <h1
            className="text-5xl font-bold mb-4"
            style={{
              background: "linear-gradient(135deg, #51e2f5, #9df9ef)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}
          >
            Emotion Recognition AI
          </h1>
          <p className="text-xl" style={{ color: "#a28089" }}>
            Discover your emotions through AI-powered analysis of facial
            expressions and voice patterns
          </p>
        </header>

        <div className="flex gap-8 mb-12">
          {/* Upload Section - Takes up 3/4 of the space */}
          <div className="flex-1">
            <div className="card">
              <div className="flex items-center mb-6">
                <Upload
                  className="w-8 h-8 icon-float mr-3"
                  style={{ color: "#51e2f5" }}
                />
                <h2
                  className="text-2xl font-semibold"
                  style={{ color: "#a28089" }}
                >
                  Upload Video
                </h2>
              </div>
              <div className="upload-zone">
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleFileUpload}
                  className="input-file w-full mb-4"
                />
                <button
                  onClick={processUploadedVideo}
                  disabled={!uploadedFile || isProcessing}
                  className="btn-primary w-full flex items-center justify-center"
                >
                  {isProcessing ? (
                    <Loader2 className="w-5 h-5 animate-spin mr-2 icon-pulse" />
                  ) : (
                    <Play className="w-5 h-5 mr-2 icon-bounce" />
                  )}
                  Analyze Video
                </button>
              </div>
              <p className="text-sm mt-4" style={{ color: "#a28089" }}>
                Upload a video file for emotion analysis. Results may vary based
                on camera clarity, lighting, facial expressions, and audio
                surroundings.
              </p>
            </div>
          </div>

          {/* Live Recording Section - Small sidebar */}
          <div className="w-80">
            <div className="card recording-small">
              <div className="flex items-center mb-6">
                <Video
                  className="w-6 h-6 icon-wiggle mr-2"
                  style={{ color: "#ffa8B6" }}
                />
                <h2
                  className="text-lg font-semibold"
                  style={{ color: "#a28089" }}
                >
                  Quick Record
                </h2>
              </div>
              <div className="space-y-3">
                <video
                  ref={videoRef}
                  className="w-full h-32 bg-black rounded-lg"
                  autoPlay
                  muted
                />
                <div className="flex flex-col space-y-2">
                  <button
                    onClick={startRecording}
                    disabled={isRecording || isProcessing}
                    className="btn-secondary flex items-center justify-center py-2"
                  >
                    <Play className="w-4 h-4 mr-1 icon-bounce" />
                    Start
                  </button>
                  <button
                    onClick={stopRecording}
                    disabled={!isRecording}
                    className="btn-secondary flex items-center justify-center py-2"
                  >
                    <Pause className="w-4 h-4 mr-1 icon-pulse" />
                    Stop & Analyze
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Processing Indicator */}
        {isProcessing && (
          <div className="card mb-8">
            <div className="flex items-center justify-center">
              <Loader2
                className="w-8 h-8 animate-spin icon-rotate mr-4"
                style={{ color: "#51e2f5" }}
              />
              <span className="text-xl" style={{ color: "#a28089" }}>
                Analyzing your emotions...
              </span>
            </div>
          </div>
        )}

        {/* Results Section */}
        {results && !results.error && (
          <div className="space-y-8">
            {/* Emotion Results */}
            <div className="card">
              <h3
                className="text-2xl font-semibold mb-6 flex items-center"
                style={{ color: "#a28089" }}
              >
                <Heart
                  className="w-6 h-6 icon-pulse mr-2"
                  style={{ color: "#51e2f5" }}
                />
                Emotion Analysis Results
              </h3>
              <div className="grid md:grid-cols-3 gap-4 mb-6">
                <div className="text-center">
                  <div
                    className="text-3xl font-bold icon-float"
                    style={{ color: "#51e2f5" }}
                  >
                    {results.audio_emotion}
                  </div>
                  <div className="text-sm" style={{ color: "#a28089" }}>
                    Audio Emotion
                  </div>
                </div>
                <div className="text-center">
                  <div
                    className="text-3xl font-bold icon-wiggle"
                    style={{ color: "#9df9ef" }}
                  >
                    {results.video_emotion}
                  </div>
                  <div className="text-sm" style={{ color: "#a28089" }}>
                    Video Emotion
                  </div>
                </div>
                <div className="text-center">
                  <div
                    className="text-4xl font-bold icon-bounce"
                    style={{ color: "#edf756" }}
                  >
                    {results.fused_emotion}
                  </div>
                  <div className="text-sm" style={{ color: "#a28089" }}>
                    Fused Emotion
                  </div>
                </div>
              </div>
              <div
                style={{
                  backgroundColor: "rgba(255, 168, 182, 0.2)",
                  padding: "1rem",
                  borderRadius: "0.5rem",
                }}
              >
                <strong style={{ color: "#a28089" }}>
                  Cognitive Analysis:
                </strong>{" "}
                {results.reasoning}
              </div>
            </div>

            {/* Temporal Charts */}
            <div className="card">
              <h3
                className="text-2xl font-semibold mb-6 flex items-center"
                style={{ color: "#a28089" }}
              >
                <Brain
                  className="w-6 h-6 icon-float mr-2"
                  style={{ color: "#9df9ef" }}
                />
                Temporal Emotion Analysis
              </h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4
                    className="text-lg font-semibold mb-4"
                    style={{ color: "#a28089" }}
                  >
                    Audio Emotions Over Time
                  </h4>
                  <Line
                    data={createChartData(results.audio_probs_temporal)}
                    options={chartOptions}
                  />
                </div>
                <div>
                  <h4
                    className="text-lg font-semibold mb-4"
                    style={{ color: "#a28089" }}
                  >
                    Video Emotions Over Time
                  </h4>
                  <Line
                    data={createChartData(results.video_probs_temporal)}
                    options={chartOptions}
                  />
                </div>
              </div>
            </div>

            {/* AI-Generated Content */}
            <div className="card">
              <h3
                className="text-2xl font-semibold mb-6 flex items-center"
                style={{ color: "#a28089" }}
              >
                <BookOpen
                  className="w-6 h-6 icon-rotate mr-2"
                  style={{ color: "#edf756" }}
                />
                AI-Generated Content
              </h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4
                    className="text-lg font-semibold mb-2 flex items-center"
                    style={{ color: "#a28089" }}
                  >
                    <BookOpen
                      className="w-5 h-5 icon-bounce mr-2"
                      style={{ color: "#51e2f5" }}
                    />
                    Short Story
                  </h4>
                  <p style={{ color: "#a28089" }} className="italic">
                    {results.story}
                  </p>
                </div>
                <div>
                  <h4
                    className="text-lg font-semibold mb-2"
                    style={{ color: "#a28089" }}
                  >
                    Inspirational Quote
                  </h4>
                  <p style={{ color: "#a28089" }} className="italic">
                    "{results.quote}"
                  </p>
                </div>
              </div>
              <div className="mt-6">
                <h4
                  className="text-lg font-semibold mb-2 flex items-center"
                  style={{ color: "#a28089" }}
                >
                  <Youtube
                    className="w-5 h-5 icon-pulse mr-2"
                    style={{ color: "#ffa8B6" }}
                  />
                  Recommended YouTube Video
                </h4>
                <p style={{ color: "#a28089" }}>{results.video}</p>
              </div>
              <div className="mt-6">
                <h4
                  className="text-lg font-semibold mb-2 flex items-center"
                  style={{ color: "#a28089" }}
                >
                  <Music
                    className="w-5 h-5 icon-wiggle mr-2"
                    style={{ color: "#9df9ef" }}
                  />
                  Recommended Song
                </h4>
                <p style={{ color: "#a28089" }}>
                  {results.song ||
                    "Based on your emotions, we recommend listening to uplifting music to enhance your current mood."}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {results && results.error && (
          <div className="card bg-red-500/20 border-red-400/30">
            <h3 className="text-xl font-semibold text-red-400 mb-2">Error</h3>
            <p className="text-red-200">{results.error}</p>
          </div>
        )}

        {/* Footer */}
        <footer
          className="text-center mt-16 py-8 border-t"
          style={{ borderColor: "rgba(162, 128, 137, 0.3)" }}
        >
          <p style={{ color: "#a28089" }}>
            Powered by advanced machine learning to understand human emotions
            through multimodal analysis
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
