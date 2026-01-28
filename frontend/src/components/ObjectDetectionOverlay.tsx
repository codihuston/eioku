import { useEffect, useState, RefObject } from 'react';

interface BoundingBox {
  timestamp: number;
  bbox: [number, number, number, number];
  confidence: number;
  label: string;
}

interface ArtifactPayload {
  label?: string;
  confidence?: number;
  bbox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

interface Artifact {
  payload: ArtifactPayload;
  span_start_ms?: number;
}

interface Props {
  videoId: string;
  videoRef: RefObject<HTMLVideoElement>;
  canvasRef: RefObject<HTMLCanvasElement>;
  enabled: boolean;
  apiUrl?: string;
}

export default function ObjectDetectionOverlay({
  videoId,
  videoRef,
  canvasRef,
  enabled,
  apiUrl = 'http://localhost:8080',
}: Props) {
  const [objects, setObjects] = useState<BoundingBox[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts?type=object.detection`)
      .then(res => res.json())
      .then((data: Artifact[]) => {
        const boxes: BoundingBox[] = [];

        data.forEach(artifact => {
          const payload = artifact.payload;

          if (!payload.bbox || !payload.label) return;

          const bb = payload.bbox;
          if (
            typeof bb.x !== 'number' ||
            typeof bb.y !== 'number' ||
            typeof bb.width !== 'number' ||
            typeof bb.height !== 'number'
          ) {
            return;
          }

          const bbox: [number, number, number, number] = [
            bb.x,
            bb.y,
            bb.x + bb.width,
            bb.y + bb.height,
          ];

          boxes.push({
            timestamp: (artifact.span_start_ms || 0) / 1000,
            bbox,
            confidence: payload.confidence || 0,
            label: payload.label,
          });
        });

        setObjects(boxes);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [videoId, apiUrl]);

  // Track video time and draw objects
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || objects.length === 0 || !enabled) return;

    const drawObjects = (timeSeconds: number) => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Find boxes for current time (within 1 second)
      const boxes = objects.filter(box => Math.abs(box.timestamp - timeSeconds) < 1.0);

      // Draw boxes
      boxes.forEach(box => {
        const [x1, y1, x2, y2] = box.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        // Scale coordinates to canvas size
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;

        const scaledX = x1 * scaleX;
        const scaledY = y1 * scaleY;
        const scaledWidth = width * scaleX;
        const scaledHeight = height * scaleY;

        // Color based on confidence
        let color = 'rgba(0, 150, 255, 0.8)'; // blue
        if (box.confidence < 0.5) {
          color = 'rgba(255, 100, 0, 0.8)'; // orange
        } else if (box.confidence < 0.7) {
          color = 'rgba(255, 200, 0, 0.8)'; // yellow
        }

        // Draw box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

        // Draw label and confidence
        ctx.fillStyle = color;
        ctx.font = 'bold 12px Arial';
        const label = `${box.label} ${(box.confidence * 100).toFixed(0)}%`;
        ctx.fillText(label, scaledX, scaledY - 5);
      });
    };

    const handleTimeUpdate = () => {
      drawObjects(video.currentTime);
    };

    const handleLoadedMetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      drawObjects(video.currentTime);
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('loadedmetadata', handleLoadedMetadata);

    // Initial draw if video is already loaded
    if (video.readyState >= 1) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      drawObjects(video.currentTime);
    }

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
    };
  }, [videoRef, canvasRef, objects, enabled]);

  if (loading) {
    return null;
  }

  if (error) {
    console.error('Object detection error:', error);
    return null;
  }

  return null;
}
