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
    <div className="min-h-screen bg-gradient-to-br from-violet-900 via-purple-900 to-indigo-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-violet-400 to-purple-400 bg-clip-text text-transparent">
            Emotion Recognition AI
          </h1>
          <p className="text-xl text-violet-200 max-w-2xl mx-auto">
            Discover your emotions through AI-powered analysis of facial
            expressions and voice patterns
          </p>
        </header>

        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* Upload Section */}
          <div className="card">
            <div className="flex items-center mb-6">
              <Upload className="w-8 h-8 text-violet-400 mr-3" />
              <h2 className="text-2xl font-semibold">Upload Video</h2>
            </div>
            <div className="space-y-4">
              <input
                type="file"
                accept="video/*"
                onChange={handleFileUpload}
                className="input-file w-full"
              />
              <button
                onClick={processUploadedVideo}
                disabled={!uploadedFile || isProcessing}
                className="btn-primary w-full flex items-center justify-center"
              >
                {isProcessing ? (
                  <Loader2 className="w-5 h-5 animate-spin mr-2" />
                ) : (
                  <Play className="w-5 h-5 mr-2" />
                )}
                Analyze Video
              </button>
            </div>
            <p className="text-sm text-violet-300 mt-4">
              Upload a video file for emotion analysis. Results may vary based
              on camera clarity, lighting, facial expressions, and audio
              surroundings.
            </p>
          </div>

          {/* Live Recording Section */}
          <div className="card">
            <div className="flex items-center mb-6">
              <Video className="w-8 h-8 text-violet-400 mr-3" />
              <h2 className="text-2xl font-semibold">Live Recording</h2>
            </div>
            <div className="space-y-4">
              <video
                ref={videoRef}
                className="w-full h-48 bg-black rounded-lg"
                autoPlay
                muted
              />
              <div className="flex space-x-4">
                <button
                  onClick={startRecording}
                  disabled={isRecording || isProcessing}
                  className="btn-secondary flex-1 flex items-center justify-center"
                >
                  <Play className="w-5 h-5 mr-2" />
                  Start Recording
                </button>
                <button
                  onClick={stopRecording}
                  disabled={!isRecording}
                  className="btn-secondary flex-1 flex items-center justify-center"
                >
                  <Pause className="w-5 h-5 mr-2" />
                  Stop & Analyze
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Processing Indicator */}
        {isProcessing && (
          <div className="card mb-8">
            <div className="flex items-center justify-center">
              <Loader2 className="w-8 h-8 animate-spin text-violet-400 mr-4" />
              <span className="text-xl">Analyzing your emotions...</span>
            </div>
          </div>
        )}

        {/* Results Section */}
        {results && !results.error && (
          <div className="space-y-8">
            {/* Emotion Results */}
            <div className="card">
              <h3 className="text-2xl font-semibold mb-6 flex items-center">
                <Heart className="w-6 h-6 text-violet-400 mr-2" />
                Emotion Analysis Results
              </h3>
              <div className="grid md:grid-cols-3 gap-4 mb-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-violet-400">
                    {results.audio_emotion}
                  </div>
                  <div className="text-sm text-violet-300">Audio Emotion</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-violet-400">
                    {results.video_emotion}
                  </div>
                  <div className="text-sm text-violet-300">Video Emotion</div>
                </div>
                <div className="text-center">
                  <div className="text-4xl font-bold text-purple-400">
                    {results.fused_emotion}
                  </div>
                  <div className="text-sm text-violet-300">Fused Emotion</div>
                </div>
              </div>
              <div className="bg-violet-500/20 p-4 rounded-lg">
                <strong>Cognitive Analysis:</strong> {results.reasoning}
              </div>
            </div>

            {/* Temporal Charts */}
            <div className="card">
              <h3 className="text-2xl font-semibold mb-6 flex items-center">
                <Brain className="w-6 h-6 text-violet-400 mr-2" />
                Temporal Emotion Analysis
              </h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-lg font-semibold mb-4">
                    Audio Emotions Over Time
                  </h4>
                  <Line
                    data={createChartData(results.audio_probs_temporal)}
                    options={chartOptions}
                  />
                </div>
                <div>
                  <h4 className="text-lg font-semibold mb-4">
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
              <h3 className="text-2xl font-semibold mb-6 flex items-center">
                <BookOpen className="w-6 h-6 text-violet-400 mr-2" />
                AI-Generated Content
              </h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-lg font-semibold mb-2 flex items-center">
                    <BookOpen className="w-5 h-5 text-violet-400 mr-2" />
                    Short Story
                  </h4>
                  <p className="text-violet-200 italic">{results.story}</p>
                </div>
                <div>
                  <h4 className="text-lg font-semibold mb-2">
                    Inspirational Quote
                  </h4>
                  <p className="text-violet-200 italic">"{results.quote}"</p>
                </div>
              </div>
              <div className="mt-6">
                <h4 className="text-lg font-semibold mb-2 flex items-center">
                  <Youtube className="w-5 h-5 text-violet-400 mr-2" />
                  Recommended YouTube Video
                </h4>
                <p className="text-violet-200">{results.video}</p>
              </div>
              <div className="mt-6">
                <h4 className="text-lg font-semibold mb-2 flex items-center">
                  <Music className="w-5 h-5 text-violet-400 mr-2" />
                  Song Recommendations
                </h4>
                <ul className="text-violet-200 space-y-1">
                  {results.songs.map((song, index) => (
                    <li key={index} className="flex items-center">
                      <Music className="w-4 h-4 mr-2" />
                      {song}
                    </li>
                  ))}
                </ul>
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
        <footer className="text-center mt-16 py-8 border-t border-violet-500/30">
          <p className="text-violet-300">
            Powered by advanced machine learning to understand human emotions
            through multimodal analysis
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
