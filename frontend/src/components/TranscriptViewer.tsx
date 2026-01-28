import React, { useEffect, useState } from 'react';

interface TranscriptSegment {
  start_ms: number;
  end_ms: number;
  text: string;
}

interface Artifact {
  span_start_ms: number;
  span_end_ms: number;
  payload?: {
    text?: string;
  };
}

interface Props {
  videoId: string;
  videoRef: React.RefObject<HTMLVideoElement>;
  apiUrl?: string;
}

export default function TranscriptViewer({ videoId, videoRef, apiUrl = 'http://localhost:8080' }: Props) {
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts?type=transcript.segment`)
      .then(res => res.json())
      .then((data: Artifact[]) => {
        const transcriptSegments = data.map(artifact => ({
          start_ms: artifact.span_start_ms,
          end_ms: artifact.span_end_ms,
          text: artifact.payload?.text || '',
        }));
        setSegments(transcriptSegments);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [videoId, apiUrl]);

  // Track video time
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime * 1000);
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    return () => video.removeEventListener('timeupdate', handleTimeUpdate);
  }, [videoRef]);

  const handleSegmentClick = (startMs: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = startMs / 1000;
      videoRef.current.play();
    }
  };

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

  if (loading) {
    return <div style={{ padding: '10px', fontSize: '12px', color: '#666' }}>Loading transcript...</div>;
  }

  if (error) {
    return <div style={{ padding: '10px', fontSize: '12px', color: '#d32f2f' }}>Error: {error}</div>;
  }

  if (segments.length === 0) {
    return <div style={{ padding: '10px', fontSize: '12px', color: '#666' }}>No transcript available</div>;
  }

  return (
    <div
      style={{
        height: '100%',
        overflowY: 'auto',
        padding: '10px',
        backgroundColor: '#fafafa',
      }}
    >
      <h3 style={{ margin: '0 0 10px 0', fontSize: '14px', fontWeight: '600' }}>Transcript</h3>
      <div>
        {segments.map((segment, idx) => {
          const isActive = currentTime >= segment.start_ms && currentTime < segment.end_ms;
          return (
            <div
              key={idx}
              onClick={() => handleSegmentClick(segment.start_ms)}
              style={{
                padding: '8px',
                marginBottom: '4px',
                backgroundColor: isActive ? '#e3f2fd' : 'transparent',
                borderLeft: isActive ? '3px solid #1976d2' : '3px solid transparent',
                cursor: 'pointer',
                transition: 'background-color 0.2s',
                borderRadius: '2px',
              }}
              onMouseEnter={e => {
                const el = e.currentTarget as HTMLDivElement;
                if (!isActive) {
                  el.style.backgroundColor = '#f5f5f5';
                }
              }}
              onMouseLeave={e => {
                const el = e.currentTarget as HTMLDivElement;
                if (!isActive) {
                  el.style.backgroundColor = 'transparent';
                }
              }}
            >
              <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>
                {formatTime(segment.start_ms)}
              </div>
              <div style={{ fontSize: '13px', lineHeight: '1.4', color: '#333' }}>{segment.text}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
