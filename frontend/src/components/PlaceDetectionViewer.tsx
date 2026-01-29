import { useEffect, useState, RefObject } from 'react';

interface Artifact {
  artifact_id: string;
  span_start_ms: number;
  span_end_ms: number;
  payload: {
    predictions?: Array<{
      label: string;
      confidence: number;
    }>;
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

export default function PlaceDetectionViewer({ videoId, videoRef, apiUrl = 'http://localhost:8080' }: Props) {
  const [places, setPlaces] = useState<Artifact[]>([]);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [filterValue, setFilterValue] = useState<string>('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch available runs
    fetch(`${apiUrl}/api/v1/videos/${videoId}/runs?artifact_type=place.classification`)
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

    fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts?type=place.classification${selectionParam}`)
      .then(res => res.json())
      .then(data => {
        setPlaces(data as Artifact[]);
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
      <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '16px' }}>
        <p style={{ color: '#999', margin: '0' }}>
          Total place detections: {places.length}
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
                Run: {formatRunDate(run.created_at)}{run.model_profile ? ` - ${run.model_profile}` : ''} ({run.artifact_count} places)
              </option>
            ))}
          </select>
        </div>
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
              {item.payload.predictions?.[0]?.label || 'Unknown'}
            </div>
            {item.payload.predictions?.[0]?.confidence && (
              <div style={{ fontSize: '12px', color: '#999', marginBottom: '8px' }}>
                Confidence: {(item.payload.predictions[0].confidence * 100).toFixed(1)}%
              </div>
            )}
            {item.payload.predictions && item.payload.predictions.length > 0 && (
              <div style={{ fontSize: '11px', color: '#666' }}>
                <div style={{ marginBottom: '4px', color: '#999' }}>Top matches:</div>
                {item.payload.predictions.slice(1, 3).map((prediction, idx) => (
                  <div key={idx} style={{ marginLeft: '8px', marginBottom: '2px' }}>
                    • {prediction.label}: {(prediction.confidence * 100).toFixed(1)}%
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

