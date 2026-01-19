"""
Face Detection and Extraction using Haar Cascade
"""

import cv2
import numpy as np
from pathlib import Path
import signal
import threading

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Video processing timeout")

class FaceDetector:
    """Extract faces from video frames using Haar Cascade"""
    
    def __init__(self):
        # Load Haar Cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError("Failed to load Haar Cascade classifier")
    
    def detect_face(self, frame):
        """
        Detect and extract largest face from frame
        Args:
            frame: numpy array (height, width, 3)
        Returns:
            face_crop: extracted face region, or None if no face detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Balanced approach: detect at 0.5 scale first (speed), then try full scale if needed
        # This catches most faces quickly while still finding smaller ones
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,  # Moderate - catches most faces without being too slow
            minNeighbors=2,     # Permissive - catch weak detections
            minSize=(40, 40)    # Lower threshold for small faces
        )
        
        if len(faces) == 0:
            # Try one more time with slightly different parameters if first attempt fails
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Add padding
        padding = int(w * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Extract face
        face_crop = frame[y:y+h, x:x+w]
        
        # Resize to 128x128
        face_crop = cv2.resize(face_crop, (128, 128))
        
        return face_crop
    
    def extract_faces_from_video(self, video_path, n_frames=8):
        """
        Extract ACTUAL FACES from video frames using Haar Cascade
        Reference approach from Emotion-Recognition_SER-FER_RAVDESS
        Args:
            video_path: path to video file
            n_frames: number of frames to extract (default 8)
        Returns:
            numpy array of shape (n_frames, 128, 128, 3) with extracted faces
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count == 0:
                cap.release()
                return None
            
            # Calculate frame step to get n_frames evenly distributed
            frame_step = max(1, frame_count // n_frames)
            
            frames = []
            frame_num = 0
            frames_read = 0
            max_frames_to_read = min(300, frame_count + 50)
            
            # Read sequentially - MUCH faster than seeking
            while len(frames) < n_frames and frames_read < max_frames_to_read:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames_read += 1
                
                # Collect every frame_step'th frame - DO ACTUAL FACE DETECTION
                if frame_num % frame_step == 0:
                    face = self.detect_face(frame)
                    if face is not None:
                        frames.append(face.astype(np.float32) / 255.0)
                    else:
                        # Fallback to full frame if no face detected
                        resized = cv2.resize(frame, (128, 128))
                        frames.append(resized.astype(np.float32) / 255.0)
                
                frame_num += 1
            
            cap.release()
            
            if len(frames) == n_frames:
                return np.array(frames)
            elif len(frames) > 0:
                # Pad with last frame if needed
                while len(frames) < n_frames:
                    frames.append(frames[-1])
                return np.array(frames[:n_frames])
            
            return None
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None
            
            cap.release()
            
            if len(frames) == n_frames:
                return np.array(frames)
            elif len(frames) > 0:
                # Pad with last frame if needed
                while len(frames) < n_frames:
                    frames.append(frames[-1])
                return np.array(frames[:n_frames])
            
            return None
        except Exception as e:
            try:
                cap.release()
            except:
                pass
            print(f"Error processing video {video_path}: {str(e)[:50]}")
            return None
