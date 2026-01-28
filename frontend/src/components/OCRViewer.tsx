import { useEffect, useState, RefObject } from 'react';

interface Artifact {
  artifact_id: string;
  span_start_ms: number;
  span_end_ms: number;
  payload: {
    text?: string;
    confidence?: number;
  };
}

interface Props {
  videoId: string;
  videoRef?: RefObject<HTMLVideoElement>;
  apiUrl?: string;
}

export default function OCRViewer({ videoId, videoRef, apiUrl = 'http://localhost:8080' }: Props) {
  const [ocr, setOcr] = useState<Artifact[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts?type=ocr.text`)
      .then(res => res.json())
      .then(data => {
        setOcr(data as Artifact[]);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [videoId, apiUrl]);

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const secs = seconds % 60;
    const mins = minutes % 60;

    if (hours > 0) {
      return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const jumpToTime = (ms: number) => {
    if (videoRef?.current) {
      videoRef.current.currentTime = ms / 1000;
      videoRef.current.play();
    }
  };

  if (loading) {
    return <div style={{ padding: '20px' }}>Loading OCR data...</div>;
  }

  if (error) {
    return <div style={{ padding: '20px', color: '#ff6b6b' }}>Error: {error}</div>;
  }

  if (ocr.length === 0) {
    return <div style={{ padding: '20px', color: '#999' }}>No text detected</div>;
  }

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ marginBottom: '20px' }}>
        <p style={{ color: '#999', margin: '0 0 10px 0' }}>
          Total text detections: {ocr.length}
        </p>
      </div>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
      }}>
        {ocr.map(item => (
          <div
            key={item.artifact_id}
            onClick={() => jumpToTime(item.span_start_ms)}
            style={{
              padding: '12px',
              backgroundColor: '#2a2a2a',
              borderRadius: '4px',
              border: '1px solid #444',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#333';
              e.currentTarget.style.borderColor = '#1976d2';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#2a2a2a';
              e.currentTarget.style.borderColor = '#444';
            }}
          >
            <div style={{ fontSize: '12px', color: '#1976d2', marginBottom: '5px' }}>
              {formatTime(item.span_start_ms)}
            </div>
            <div style={{ fontSize: '14px', color: '#fff', marginBottom: '5px' }}>
              {item.payload.text || 'No text'}
            </div>
            {item.payload.confidence && (
              <div style={{ fontSize: '12px', color: '#999' }}>
                Confidence: {(item.payload.confidence * 100).toFixed(1)}%
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
