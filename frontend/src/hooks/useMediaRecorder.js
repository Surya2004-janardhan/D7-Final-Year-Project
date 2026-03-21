import { useState, useRef, useCallback, useEffect } from 'react';
import { logError, logInfo, logWarn } from '../utils/logger';

const MEDIA_CONSTRAINTS = {
  video: {
    width: { ideal: 640, max: 1280 },
    height: { ideal: 480, max: 720 },
    frameRate: { ideal: 15, max: 24 },
  },
  audio: true,
};

export default function useMediaRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [stream, setStream] = useState(null);
  const [hasPermission, setHasPermission] = useState(false);
  const [permissionError, setPermissionError] = useState(null);
  const [lastCaptureMeta, setLastCaptureMeta] = useState(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const videoRef = useRef(null);

  const setVideoElement = useCallback((node) => {
    videoRef.current = node;
    if (node && stream) {
      node.srcObject = stream;
    }
  }, [stream]);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  const requestPermission = useCallback(async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia(MEDIA_CONSTRAINTS);
      setStream(s);
      setHasPermission(true);
      setPermissionError(null);
      logInfo('recorder', 'camera+mic permission granted');
      if (videoRef.current) {
        videoRef.current.srcObject = s;
      }
      return s;
    } catch (err) {
      logWarn('recorder', 'camera+mic permission denied, trying camera-only fallback', { error: err.message });
      try {
        const s = await navigator.mediaDevices.getUserMedia({ ...MEDIA_CONSTRAINTS, audio: false });
        setStream(s);
        setHasPermission(true);
        setPermissionError('Microphone unavailable. Using camera-only mode.');
        logWarn('recorder', 'camera-only fallback granted');
        if (videoRef.current) {
          videoRef.current.srcObject = s;
        }
        return s;
      } catch (fallbackErr) {
        logError('recorder', 'camera fallback denied', { error: fallbackErr?.message || 'Camera/microphone access blocked.' });
        setHasPermission(false);
        setPermissionError(fallbackErr?.message || 'Camera/microphone access blocked.');
        return null;
      }
    }
  }, []);

  const stopStream = useCallback(() => {
    if (stream) {
      logInfo('recorder', 'stopping media stream');
      stream.getTracks().forEach(t => t.stop());
      setStream(null);
      setHasPermission(false);
      setPermissionError(null);
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }
  }, [stream]);

  const startRecording = useCallback(async () => {
    logInfo('recorder', 'manual recording requested');
    let s = stream;
    if (!s) {
      s = await requestPermission();
      if (!s) return null;
    }

    chunksRef.current = [];

    const hasAudio = s.getAudioTracks().length > 0;
    const mimeType = hasAudio ? 'video/webm;codecs=vp8,opus' : 'video/webm;codecs=vp8';
    const recorder = new MediaRecorder(s, { mimeType });

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    return new Promise((resolve) => {
      const startedAt = new Date().toISOString();
      recorder.onstop = () => {
        const totalSize = chunksRef.current.reduce((sum, chunk) => sum + chunk.size, 0);
        const blob = new Blob(chunksRef.current, { type: 'video/webm' });
        chunksRef.current = [];
        setIsRecording(false);
        setLastCaptureMeta({
          startedAt,
          endedAt: new Date().toISOString(),
        });
        logInfo('recorder', 'manual recording completed', { startedAt, size: totalSize });
        resolve(blob);
      };

      mediaRecorderRef.current = recorder;
      recorder.start(100);
      setIsRecording(true);
      logInfo('recorder', 'manual recording started', { startedAt });
    });
  }, [stream, requestPermission]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      logInfo('recorder', 'manual recording stop requested');
      mediaRecorderRef.current.stop();
    }
  }, []);

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
    };
  }, [stream]);

  return {
    isRecording,
    stream,
    hasPermission,
    permissionError,
    lastCaptureMeta,
    videoRef: setVideoElement,
    requestPermission,
    stopStream,
    startRecording,
    stopRecording,
  };
}
