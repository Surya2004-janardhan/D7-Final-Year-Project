import { Activity, Camera, Mic, ShieldCheck } from "lucide-react";
import { useEffect, useRef } from "react";

export default function RecordingPanel({
  autoMode,
  daemonStatus,
  nextFireLabel,
  permissionReady,
  permissionError,
  requestPermission,
  intervalLabel,
  recordDurationLabel,
  settings,
  save,
  pausedForMusic,
}) {
  const isRecording = daemonStatus === "recording";
  const isWaiting = daemonStatus === "waiting";
  const isPermissionRequired = daemonStatus === "permission_required";
  const isDeviceBusy = daemonStatus === "paused_device_busy";
  const isProcessing = daemonStatus === "processing";

  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    let mounted = true;
    const startPreview = async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
        if (!mounted) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = s;
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          try {
            await videoRef.current.play();
          } catch (e) {
            /* ignore */
          }
        }
      } catch (err) {
        // ignore preview errors
      }
    };

    if (isRecording && settings?.mirrorPreview) {
      startPreview();
    } else {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (videoRef.current) videoRef.current.srcObject = null;
    }

    return () => {
      mounted = false;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (videoRef.current) videoRef.current.srcObject = null;
    };
  }, [isRecording, settings?.mirrorPreview]);

  const headline = isRecording
    ? "Background capture is running"
    : isProcessing
      ? "Previous capture is being analyzed"
      : isPermissionRequired
        ? "Camera and microphone access is needed"
        : isDeviceBusy
          ? "Devices are busy in another app"
          : autoMode
            ? "Auto monitoring is armed"
            : "Auto monitoring is currently off";

  const subline = isRecording
    ? "EmotionAI is collecting camera and microphone input for the active auto session."
    : isProcessing
      ? "The backend is processing the last clip while the next monitoring cycle waits in the background."
      : isPermissionRequired
        ? "Enable both permissions once and the app will start its timed monitoring flow automatically."
        : isDeviceBusy
          ? "A meeting or another capture app is using the devices right now. Monitoring will resume after they are free."
          : autoMode
            ? `The app waits ${intervalLabel}, records ${recordDurationLabel}, then repeats.`
            : "Turn on Auto Mode from the sidebar to let the app monitor in the background.";

  return (
    <div className="w-full max-w-3xl mx-auto animate-fade-up">
      <div className="panel overflow-hidden rounded-[28px] border-primary/20 shadow-[0_24px_80px_rgba(59,130,246,0.12)]">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(147,197,253,0.28),transparent_38%),radial-gradient(circle_at_bottom_left,rgba(219,234,254,0.8),transparent_32%)] pointer-events-none" />

        <div className="relative flex items-center justify-between px-6 py-5 border-b border-primary/10 bg-white/80 backdrop-blur-xl">
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.22em] text-primary">
              Auto Monitoring
            </p>
            <h2 className="text-lg font-black text-text-primary">
              Emotion based Stress Tracking
            </h2>
          </div>
          <div
            className={`px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider border ${
              isRecording
                ? "bg-red-50 text-red-600 border-red-200"
                : isProcessing
                  ? "bg-primary/10 text-primary border-primary/20"
                  : permissionReady
                    ? "bg-emerald-50 text-emerald-600 border-emerald-200"
                    : "bg-surface-raised text-text-secondary border-border-subtle"
            }`}
          >
            {isRecording
              ? "Recording"
              : isProcessing
                ? "Processing"
                : permissionReady
                  ? "Ready"
                  : "Permission Needed"}
          </div>
        </div>

        <div className="relative px-8 py-10">
          <div className="max-w-xl mx-auto text-center space-y-6">
            <div className="relative w-28 h-28 mx-auto">
              <div
                className={`absolute inset-0 rounded-full ${
                  isRecording
                    ? "bg-red-100 animate-pulse-ring"
                    : "bg-primary/12 animate-pulse-ring"
                }`}
              />
              <div className="relative w-28 h-28 rounded-full border border-primary/15 bg-white shadow-lg flex items-center justify-center">
                {isRecording ? (
                  <div className="flex items-center justify-center">
                    <div className="w-14 h-14 rounded-full border-4 border-red-200 border-t-red-500 animate-spin" />
                  </div>
                ) : isPermissionRequired ? (
                  <ShieldCheck className="w-12 h-12 text-primary" />
                ) : (
                  <Activity
                    className={`w-12 h-12 ${autoMode ? "text-primary" : "text-text-muted"}`}
                  />
                )}
              </div>
            </div>

            <div className="space-y-2">
              <p className="text-text-primary text-base font-bold">
                {headline}
              </p>
              <p className="text-text-secondary text-sm leading-relaxed">
                {subline}
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-left">
              <div className="rounded-2xl bg-surface-raised border border-border-subtle p-4">
                <p className="text-[11px] font-bold uppercase tracking-wider text-text-muted">
                  Permissions
                </p>
                <p className="text-sm font-semibold text-text-primary mt-2">
                  {permissionReady
                    ? "Camera + mic enabled"
                    : "Waiting for access"}
                </p>
              </div>
              <div className="rounded-2xl bg-surface-raised border border-border-subtle p-4">
                <p className="text-[11px] font-bold uppercase tracking-wider text-text-muted">
                  Cycle
                </p>
                <p className="text-sm font-semibold text-text-primary mt-2">
                  {nextFireLabel && isWaiting
                    ? `Next run in ${nextFireLabel}`
                    : isRecording
                      ? "Recording now"
                      : isProcessing
                        ? "Analyzing latest clip"
                        : "Standby"}
                </p>
              </div>
              <div className="rounded-2xl bg-surface-raised border border-border-subtle p-4">
                <p className="text-[11px] font-bold uppercase tracking-wider text-text-muted">
                  Plan
                </p>
                <p className="text-sm font-semibold text-text-primary mt-2">
                  {intervalLabel} wait / {recordDurationLabel} record
                </p>
              </div>
            </div>

            {!permissionReady && (
              <div className="space-y-3">
                <button
                  onClick={requestPermission}
                  className="inline-flex items-center gap-2 px-5 py-3 rounded-2xl bg-primary text-white text-sm font-bold cursor-pointer hover:opacity-90 transition-all"
                >
                  <Camera className="w-4 h-4" />
                  <Mic className="w-4 h-4" />
                  Enable Camera and Mic
                </button>

                {permissionError && (
                  <div className="max-w-md mx-auto rounded-xl bg-red-50 border border-red-200 px-3 py-2">
                    <p className="text-[11px] text-red-600 font-medium">
                      {permissionError}
                    </p>
                  </div>
                )}
              </div>
            )}
            <div className="mt-4 flex items-center justify-center gap-3">
              <label className="flex items-center gap-2 text-sm text-text-secondary">
                <input
                  type="checkbox"
                  checked={!!settings?.mirrorPreview}
                  onChange={() =>
                    save({ mirrorPreview: !settings?.mirrorPreview })
                  }
                />
                Mirror Preview (while recording)
              </label>
            </div>
          </div>
        </div>
      </div>
      {/* Mirror preview window — visible only when toggle is on and recording */}
      {settings?.mirrorPreview && isRecording && (
        <div
          style={{
            position: "fixed",
            right: 16,
            bottom: 16,
            width: 180,
            height: 140,
            zIndex: 60,
          }}
        >
          <div className="rounded-md overflow-hidden border border-border-subtle shadow-lg bg-black">
            <video
              id="mirror-preview"
              ref={videoRef}
              className="w-full h-full object-cover"
              autoPlay
              muted
            />
          </div>
        </div>
      )}
    </div>
  );
}
