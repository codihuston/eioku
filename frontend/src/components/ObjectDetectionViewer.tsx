import { useEffect, useState, RefObject } from 'react';

interface Artifact {
  artifact_id: string;
  span_start_ms: number;
  span_end_ms: number;
  payload: {
    label?: string;
    confidence?: number;
    bbox?: [number, number, number, number];
  };
}

interface RunInfo {
  run_id: string;
  created_at: string;
  artifact_count: number;
  model_profile: string | null;
}

interface Props {
  videoId: string;
  videoRef?: RefObject<HTMLVideoElement>;
  apiUrl?: string;
}

export default function ObjectDetectionViewer({ videoId, videoRef, apiUrl = 'http://localhost:8080' }: Props) {
  const [objects, setObjects] = useState<Artifact[]>([]);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [filterValue, setFilterValue] = useState<string>('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch available runs
    fetch(`${apiUrl}/api/v1/videos/${videoId}/runs?artifact_type=object.detection`)
      .then(res => res.json())
      .then(data => {
        setRuns(data.runs || []);
      })
      .catch(err => {
        console.error('Failed to fetch runs:', err);
      });

    let selectionParam = '';
    const selectionModes = ['all', 'latest'];
    if (selectionModes.includes(filterValue)) {
      if (filterValue !== 'all') {
        selectionParam = `&selection=${filterValue}`;
      }
    } else {
      // It's a run_id
      selectionParam = `&run_id=${filterValue}`;
    }

    fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts?type=object.detection${selectionParam}`)
      .then(res => res.json())
      .then(data => {
        setObjects(data as Artifact[]);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [videoId, apiUrl, filterValue]);

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

  const formatRunDate = (dateString: string) => {
    return new Date(dateString).toLocaleString(undefined, {
      dateStyle: 'short',
      timeStyle: 'short',
    });
  };

  const jumpToTime = (ms: number) => {
    if (videoRef?.current) {
      videoRef.current.currentTime = ms / 1000;
      videoRef.current.play();
    }
  };

  if (loading) {
    return <div style={{ padding: '20px' }}>Loading object detection data...</div>;
  }

  if (error) {
    return <div style={{ padding: '20px', color: '#ff6b6b' }}>Error: {error}</div>;
  }

  if (objects.length === 0) {
    return <div style={{ padding: '20px', color: '#999' }}>No objects detected</div>;
  }

  // Group by label
  const groupedByLabel = objects.reduce((acc, obj) => {
    const label = obj.payload.label || 'Unknown';
    if (!acc[label]) acc[label] = [];
    acc[label].push(obj);
    return acc;
  }, {} as Record<string, Artifact[]>);

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '16px' }}>
        <p style={{ color: '#999', margin: '0' }}>
          Total detections: {objects.length}
        </p>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <label style={{ color: '#999', fontSize: '14px' }}>Run:</label>
          <select
            value={filterValue}
            onChange={(e) => setFilterValue(e.target.value)}
            style={{
              padding: '6px 10px',
              backgroundColor: '#2a2a2a',
              color: '#fff',
              border: '1px solid #444',
              borderRadius: '4px',
              fontSize: '14px',
              cursor: 'pointer',
            }}
          >
            <option value="all">All runs</option>
            <option value="latest">Latest only</option>
            {runs.length > 0 && <option disabled>--- Runs ---</option>}
            {runs.map(run => (
              <option key={run.run_id} value={run.run_id}>
                Run: {formatRunDate(run.created_at)}{run.model_profile ? ` - ${run.model_profile}` : ''} ({run.artifact_count} detections)
              </option>
            ))}
          </select>
        </div>
      </div>

      {Object.entries(groupedByLabel).map(([label, items]) => (
        <div key={label} style={{ marginBottom: '20px' }}>
          <h4 style={{ margin: '0 0 10px 0', color: '#fff' }}>
            {label} ({items.length})
          </h4>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
            gap: '10px',
          }}>
            {items.map(obj => (
              <div
                key={obj.artifact_id}
                onClick={() => jumpToTime(obj.span_start_ms)}
                style={{
                  padding: '10px',
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
                  {formatTime(obj.span_start_ms)}
                </div>
                {obj.payload.confidence && (
                  <div style={{ fontSize: '12px', color: '#999' }}>
                    Confidence: {(obj.payload.confidence * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
