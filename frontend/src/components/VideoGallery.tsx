import { useEffect, useState } from 'react';

interface Video {
  video_id: string;
  filename: string;
  file_size: number;
  status: string;
}

interface Props {
  apiUrl?: string;
  onSelectVideo: (videoId: string) => void;
}

export default function VideoGallery({ apiUrl = 'http://localhost:8080', onSelectVideo }: Props) {
  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${apiUrl}/api/v1/videos/`)
      .then(res => res.json())
      .then(data => {
        setVideos(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [apiUrl]);

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  if (loading) {
    return <div style={{ padding: '20px' }}>Loading videos...</div>;
  }

  if (error) {
    return <div style={{ padding: '20px', color: 'red' }}>Error: {error}</div>;
  }

  if (videos.length === 0) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>
        <p>No videos found. Add videos to get started.</p>
      </div>
    );
  }

  return (
    <div style={{ padding: '20px' }}>
      <h2>Videos</h2>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
          gap: '20px',
          marginTop: '20px',
        }}
      >
        {videos.map(video => (
          <div
            key={video.video_id}
            onClick={() => onSelectVideo(video.video_id)}
            style={{
              cursor: 'pointer',
              border: '1px solid #ddd',
              borderRadius: '8px',
              overflow: 'hidden',
              transition: 'transform 0.2s, box-shadow 0.2s',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            }}
            onMouseEnter={e => {
              const el = e.currentTarget as HTMLDivElement;
              el.style.transform = 'translateY(-4px)';
              el.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
            }}
            onMouseLeave={e => {
              const el = e.currentTarget as HTMLDivElement;
              el.style.transform = 'translateY(0)';
              el.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
            }}
          >
            {/* Placeholder thumbnail */}
            <div
              style={{
                width: '100%',
                paddingBottom: '56.25%', // 16:9 aspect ratio
                backgroundColor: '#f0f0f0',
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '48px',
                color: '#ccc',
              }}
            >
              ▶
            </div>

            {/* Video info */}
            <div style={{ padding: '12px' }}>
              <h3
                style={{
                  margin: '0 0 8px 0',
                  fontSize: '14px',
                  fontWeight: '600',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {video.filename}
              </h3>
              <div style={{ fontSize: '12px', color: '#666' }}>
                <div>{formatFileSize(video.file_size)}</div>
                <div style={{ marginTop: '4px' }}>
                  <span
                    style={{
                      display: 'inline-block',
                      padding: '2px 8px',
                      backgroundColor:
                        video.status === 'completed'
                          ? '#e8f5e9'
                          : video.status === 'processing'
                            ? '#fff3e0'
                            : '#f5f5f5',
                      color:
                        video.status === 'completed'
                          ? '#2e7d32'
                          : video.status === 'processing'
                            ? '#e65100'
                            : '#666',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: '500',
                    }}
                  >
                    {video.status}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
