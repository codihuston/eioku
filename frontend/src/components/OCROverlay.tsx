import { useEffect, useState, RefObject } from 'react';

interface PolygonPoint {
  x: number;
  y: number;
}

interface OCRBox {
  timestamp: number;
  polygon: PolygonPoint[];
  confidence: number;
  text: string;
}

interface ArtifactPayload {
  text?: string;
  confidence?: number;
  polygon?: PolygonPoint[];
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

export default function OCROverlay({
  videoId,
  videoRef,
  canvasRef,
  enabled,
  apiUrl = 'http://localhost:8080',
}: Props) {
  const [ocrBoxes, setOcrBoxes] = useState<OCRBox[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts?type=ocr.text`)
      .then(res => res.json())
      .then((data: Artifact[]) => {
        const boxes: OCRBox[] = [];

        data.forEach(artifact => {
          const payload = artifact.payload;

          if (!payload.polygon || !payload.text) return;

          // Validate polygon has at least 3 points
          if (payload.polygon.length < 3) return;

          // Validate all points have x and y
          const validPolygon = payload.polygon.every(
            p => typeof p.x === 'number' && typeof p.y === 'number'
          );
          if (!validPolygon) return;

          boxes.push({
            timestamp: (artifact.span_start_ms || 0) / 1000,
            polygon: payload.polygon,
            confidence: payload.confidence || 0,
            text: payload.text,
          });
        });

        setOcrBoxes(boxes);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [videoId, apiUrl]);

  // Track video time and draw OCR boxes
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || ocrBoxes.length === 0 || !enabled) return;

    const drawOCR = (timeSeconds: number) => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Find boxes for current time (within 1 second)
      const boxes = ocrBoxes.filter(box => Math.abs(box.timestamp - timeSeconds) < 1.0);

      // Scale coordinates
      const scaleX = canvas.width / video.videoWidth;
      const scaleY = canvas.height / video.videoHeight;

      // Draw boxes
      boxes.forEach(box => {
        // Color based on confidence
        let color = 'rgba(0, 200, 100, 0.8)'; // green for OCR
        if (box.confidence < 0.5) {
          color = 'rgba(255, 100, 0, 0.8)'; // orange
        } else if (box.confidence < 0.7) {
          color = 'rgba(255, 200, 0, 0.8)'; // yellow
        }

        // Draw polygon
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;

        const scaledPolygon = box.polygon.map(p => ({
          x: p.x * scaleX,
          y: p.y * scaleY,
        }));

        ctx.moveTo(scaledPolygon[0].x, scaledPolygon[0].y);
        for (let i = 1; i < scaledPolygon.length; i++) {
          ctx.lineTo(scaledPolygon[i].x, scaledPolygon[i].y);
        }
        ctx.closePath();
        ctx.stroke();

        // Draw semi-transparent fill
        ctx.fillStyle = color.replace('0.8', '0.1');
        ctx.fill();

        // Draw text label above the polygon
        const minY = Math.min(...scaledPolygon.map(p => p.y));
        const minX = Math.min(...scaledPolygon.map(p => p.x));

        ctx.fillStyle = color;
        ctx.font = 'bold 11px Arial';

        // Truncate text if too long
        const displayText = box.text.length > 30 ? box.text.substring(0, 27) + '...' : box.text;
        const label = `${displayText} (${(box.confidence * 100).toFixed(0)}%)`;

        // Draw background for text
        const textMetrics = ctx.measureText(label);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(minX - 2, minY - 16, textMetrics.width + 4, 14);

        ctx.fillStyle = color;
        ctx.fillText(label, minX, minY - 5);
      });
    };

    const handleTimeUpdate = () => {
      drawOCR(video.currentTime);
    };

    const handleLoadedMetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      drawOCR(video.currentTime);
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('loadedmetadata', handleLoadedMetadata);

    // Initial draw if video is already loaded
    if (video.readyState >= 1) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      drawOCR(video.currentTime);
    }

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
    };
  }, [videoRef, canvasRef, ocrBoxes, enabled]);

  if (loading) {
    return null;
  }

  if (error) {
    console.error('OCR overlay error:', error);
    return null;
  }

  return null;
}
