import { useEffect, useState, RefObject } from 'react';

interface Artifact {
  artifact_id: string;
  span_start_ms: number;
  span_end_ms: number;
  payload: {
    places?: Array<{
      name: string;
      confidence: number;
    }>;
    top_place?: string;
    confidence?: number;
  };
}

interface Props {
  videoId: string;
  videoRef?: RefObject<HTMLVideoElement>;
  apiUrl?: string;
}

export default function PlaceDetectionViewer({ videoId, videoRef, apiUrl = 'http://localhost:8080' }: Props) {
  const [places, setPlaces] = useState<Artifact[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts?type=place.classification`)
      .then(res => res.json())
      .then(data => {
        setPlaces(data as Artifact[]);
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
    return <div style={{ padding: '20px' }}>Loading place detection data...</div>;
  }

  if (error) {
    return <div style={{ padding: '20px', color: '#ff6b6b' }}>Error: {error}</div>;
  }

  if (places.length === 0) {
    return <div style={{ padding: '20px', color: '#999' }}>No places detected</div>;
  }

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ marginBottom: '20px' }}>
        <p style={{ color: '#999', margin: '0 0 10px 0' }}>
          Total place detections: {places.length}
        </p>
      </div>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
      }}>
        {places.map(item => (
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
            <div style={{ fontSize: '14px', color: '#fff', marginBottom: '8px', fontWeight: '600' }}>
              {item.payload.top_place || 'Unknown'}
            </div>
            {item.payload.confidence && (
              <div style={{ fontSize: '12px', color: '#999', marginBottom: '8px' }}>
                Confidence: {(item.payload.confidence * 100).toFixed(1)}%
              </div>
            )}
            {item.payload.places && item.payload.places.length > 0 && (
              <div style={{ fontSize: '11px', color: '#666' }}>
                <div style={{ marginBottom: '4px', color: '#999' }}>Top matches:</div>
                {item.payload.places.slice(0, 3).map((place, idx) => (
                  <div key={idx} style={{ marginLeft: '8px', marginBottom: '2px' }}>
                    • {place.name}: {(place.confidence * 100).toFixed(1)}%
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
