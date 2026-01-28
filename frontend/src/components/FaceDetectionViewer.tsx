import { useEffect, useRef, useState } from 'react';

interface BoundingBox {
  frame: number;
  timestamp: number;
  bbox: [number, number, number, number];
  confidence: number;
}

interface FaceGroup {
  face_id: string;
  person_id: string | null;
  occurrences: number;
  confidence: number;
  bounding_boxes: BoundingBox[];
}

interface FaceData {
  video_id: string;
  faces: FaceGroup[];
  face_groups: number;
  total_occurrences: number;
}

interface Props {
  videoId: string;
  apiUrl?: string;
}

export default function FaceDetectionViewer({ videoId, apiUrl = 'http://localhost:8080' }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [faceData, setFaceData] = useState<FaceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentBoxes, setCurrentBoxes] = useState<BoundingBox[]>([]);

  useEffect(() => {
    // Fetch face detection data
    fetch(`${apiUrl}/api/v1/videos/${videoId}/faces`)
      .then(res => res.json())
      .then(data => {
        setFaceData(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [videoId, apiUrl]);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !faceData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawBoxes = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Find boxes for current time
      const currentTime = video.currentTime;
      const boxes: BoundingBox[] = [];

      faceData.faces.forEach(face => {
        face.bounding_boxes.forEach(box => {
          // Show box if within 0.1 seconds of timestamp
          if (Math.abs(box.timestamp - currentTime) < 0.1) {
            boxes.push(box);
          }
        });
      });

      setCurrentBoxes(boxes);

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
        let color = 'rgba(0, 255, 0, 0.8)'; // green
        if (box.confidence < 0.5) {
          color = 'rgba(255, 0, 0, 0.8)'; // red
        } else if (box.confidence < 0.7) {
          color = 'rgba(255, 255, 0, 0.8)'; // yellow
        }

        // Draw box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

        // Draw confidence label
        ctx.fillStyle = color;
        ctx.font = '14px Arial';
        ctx.fillText(
          `${(box.confidence * 100).toFixed(1)}%`,
          scaledX,
          scaledY - 5
        );
      });

      requestAnimationFrame(drawBoxes);
    };

    // Start drawing loop
    const animationId = requestAnimationFrame(drawBoxes);

    // Resize canvas when video metadata loads
    const handleLoadedMetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    };

    video.addEventListener('loadedmetadata', handleLoadedMetadata);

    return () => {
      cancelAnimationFrame(animationId);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
    };
  }, [faceData]);

  if (loading) {
    return <div style={{ padding: '20px' }}>Loading face detection data...</div>;
  }

  if (error) {
    return <div style={{ padding: '20px', color: 'red' }}>Error: {error}</div>;
  }

  if (!faceData) {
    return <div style={{ padding: '20px' }}>No face data available</div>;
  }

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h2>Face Detection Viewer</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <p><strong>Video ID:</strong> {videoId}</p>
        <p><strong>Face Groups:</strong> {faceData.face_groups}</p>
        <p><strong>Total Detections:</strong> {faceData.total_occurrences}</p>
        <p><strong>Current Boxes:</strong> {currentBoxes.length}</p>
      </div>

      <div style={{ position: 'relative', display: 'inline-block' }}>
        <video
          ref={videoRef}
          controls
          style={{ 
            width: '100%', 
            maxWidth: '800px',
            display: 'block'
          }}
          src={`${apiUrl}/api/v1/videos/${videoId}/stream`}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none'
          }}
        />
      </div>

      <div style={{ marginTop: '20px' }}>
        <h3>Legend</h3>
        <div style={{ display: 'flex', gap: '20px' }}>
          <div>
            <span style={{ 
              display: 'inline-block', 
              width: '20px', 
              height: '20px', 
              backgroundColor: 'rgba(0, 255, 0, 0.8)',
              border: '1px solid black'
            }} />
            {' '}High confidence (&gt; 70%)
          </div>
          <div>
            <span style={{ 
              display: 'inline-block', 
              width: '20px', 
              height: '20px', 
              backgroundColor: 'rgba(255, 255, 0, 0.8)',
              border: '1px solid black'
            }} />
            {' '}Medium confidence (50-70%)
          </div>
          <div>
            <span style={{ 
              display: 'inline-block', 
              width: '20px', 
              height: '20px', 
              backgroundColor: 'rgba(255, 0, 0, 0.8)',
              border: '1px solid black'
            }} />
            {' '}Low confidence (&lt; 50%)
          </div>
        </div>
      </div>
    </div>
  );
}
