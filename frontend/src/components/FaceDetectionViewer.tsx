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
    // Fetch face detection data from artifacts endpoint
    fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts?type=face.detection`)
      .then(res => res.json())
      .then((data: Array<{
        payload: Record<string, unknown>;
        span_start_ms?: number;
      }>) => {
        console.log('Fetched artifacts:', data);
        console.log(`Total artifacts: ${data.length}`);
        
        // Transform artifacts into face data format
        const faceMap = new Map<string, FaceGroup>();
        let processedCount = 0;
        let skippedCount = 0;

        data.forEach((artifact, idx) => {
          const payload = artifact.payload;
          console.log(`Artifact ${idx}:`, payload);
          
          // Check if bbox exists (not bounding_box)
          if (!payload.bbox) {
            console.warn(`Artifact ${idx}: No bbox in payload`);
            skippedCount++;
            return;
          }
          
          const clusterId = payload.cluster_id || `face_${Math.random()}`;
          
          if (!faceMap.has(clusterId)) {
            faceMap.set(clusterId, {
              face_id: clusterId,
              person_id: null,
              occurrences: 0,
              confidence: payload.confidence || 0,
              bounding_boxes: [],
            });
          }
          const face = faceMap.get(clusterId)!;
          face.occurrences += 1;
          
          // Check if bbox has required properties
          const bb = payload.bbox;
          if (!bb || typeof bb.x !== 'number' || typeof bb.y !== 'number' || typeof bb.width !== 'number' || typeof bb.height !== 'number') {
            console.warn(`Artifact ${idx}: Invalid bbox structure:`, bb);
            skippedCount++;
            return;
          }
          
          // Convert bounding box from (x, y, width, height) to (x1, y1, x2, y2)
          const bbox: [number, number, number, number] = [
            bb.x,
            bb.y,
            bb.x + bb.width,
            bb.y + bb.height,
          ];
          
          face.bounding_boxes.push({
            frame: payload.frame_index || 0,
            timestamp: (artifact.span_start_ms || 0) / 1000,
            bbox,
            confidence: payload.confidence || 0,
          });
          processedCount++;
        });

        const faceData: FaceData = {
          video_id: videoId,
          faces: Array.from(faceMap.values()),
          face_groups: faceMap.size,
          total_occurrences: data.length,
        };

        console.log(`Processed ${processedCount} artifacts, skipped ${skippedCount}, created ${faceMap.size} face groups`);
        console.log('Processed face data:', faceData);
        // Log sample timestamps
        if (faceData.faces.length > 0 && faceData.faces[0].bounding_boxes.length > 0) {
          const sampleBoxes = faceData.faces[0].bounding_boxes.slice(0, 5);
          console.log('Sample box timestamps:', sampleBoxes.map(b => b.timestamp));
        }
        setFaceData(faceData);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error fetching artifacts:', err);
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
          // Show box if within 1 second of timestamp (more lenient)
          if (Math.abs(box.timestamp - currentTime) < 1.0) {
            boxes.push(box);
          }
        });
      });

      if (boxes.length === 0 && currentTime > 0) {
        // Debug: log all available timestamps
        const allTimestamps: number[] = [];
        faceData.faces.forEach(face => {
          face.bounding_boxes.forEach(box => {
            allTimestamps.push(box.timestamp);
          });
        });
        console.log(`Current time: ${currentTime.toFixed(2)}s, Found 0 boxes. Available timestamps: min=${Math.min(...allTimestamps).toFixed(2)}, max=${Math.max(...allTimestamps).toFixed(2)}`);
      } else {
        console.log(`Current time: ${currentTime.toFixed(2)}s, Found ${boxes.length} boxes`);
      }
      setCurrentBoxes(boxes);

      // Draw boxes
      boxes.forEach((box, idx) => {
        const [x1, y1, x2, y2] = box.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        console.log(`Box ${idx}: bbox=[${x1}, ${y1}, ${x2}, ${y2}], width=${width}, height=${height}`);

        // Scale coordinates to canvas size
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;

        const scaledX = x1 * scaleX;
        const scaledY = y1 * scaleY;
        const scaledWidth = width * scaleX;
        const scaledHeight = height * scaleY;

        console.log(`Scaled: x=${scaledX}, y=${scaledY}, w=${scaledWidth}, h=${scaledHeight}`);

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
