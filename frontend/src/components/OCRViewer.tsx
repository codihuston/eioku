import { useEffect, useState, useMemo, RefObject } from 'react';

interface Artifact {
  artifact_id: string;
  span_start_ms: number;
  span_end_ms: number;
  payload: {
    text?: string;
    confidence?: number;
    language?: string;
    languages?: string[];
  };
}

interface Task {
  task_id: string;
  task_type: string;
  language?: string | null;
  status: string;
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

export default function OCRViewer({ videoId, videoRef, apiUrl = 'http://localhost:8080' }: Props) {
  const [ocr, setOcr] = useState<Artifact[]>([]);
  const [taskLanguages, setTaskLanguages] = useState<string[]>([]);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<string>('all');
  const [filterValue, setFilterValue] = useState<string>('all');

  // Fetch OCR tasks to get available languages
  useEffect(() => {
    fetch(`${apiUrl}/api/v1/videos/${videoId}/tasks`)
      .then(res => res.json())
      .then((tasks: Task[]) => {
        const ocrTasks = tasks.filter(t => t.task_type === 'ocr');
        const langs = ocrTasks
          .map(t => t.language)
          .filter((lang): lang is string => lang != null)
          .sort();
        setTaskLanguages(langs);
      })
      .catch(err => {
        console.error('Failed to fetch tasks:', err);
      });
  }, [videoId, apiUrl]);

  // Fetch available runs
  useEffect(() => {
    fetch(`${apiUrl}/api/v1/videos/${videoId}/runs?artifact_type=ocr.text`)
      .then(res => res.json())
      .then(data => {
        setRuns(data.runs || []);
      })
      .catch(err => {
        console.error('Failed to fetch runs:', err);
      });
  }, [videoId, apiUrl]);

  // Fetch OCR artifacts based on the selected filter
  useEffect(() => {
    setLoading(true);

    let selectionParam = '';
    const selectionModes = ['all', 'latest', 'latest_per_language'];
    if (selectionModes.includes(filterValue)) {
      if (filterValue !== 'all') {
        selectionParam = `&selection=${filterValue}`;
      }
    } else {
      // It's a run_id
      selectionParam = `&run_id=${filterValue}`;
    }

    fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts?type=ocr.text${selectionParam}`)
      .then(res => res.json())
      .then(data => {
        setOcr(data as Artifact[]);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [videoId, apiUrl, filterValue]);

  // Filter artifacts by selected language
  const filteredOcr = useMemo(() => {
    if (selectedLanguage === 'all') return ocr;
    return ocr.filter(item => {
      const lang = item.payload.language || (item.payload.languages?.[0]);
      return lang === selectedLanguage;
    });
  }, [ocr, selectedLanguage]);

  // Get language for an artifact (from payload)
  const getLanguage = (item: Artifact): string => {
    return item.payload.language || (item.payload.languages?.[0]) || 'unknown';
  };

  const formatRunDate = (dateString: string) => {
    return new Date(dateString).toLocaleString(undefined, {
      dateStyle: 'short',
      timeStyle: 'short',
    });
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
      <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
        <p style={{ color: '#999', margin: '0' }}>
          Total: {filteredOcr.length} {selectedLanguage !== 'all' && `of ${ocr.length}`} detections
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
            <option value="latest_per_language">Latest per language</option>
            <option value="latest">Latest only</option>
            {runs.length > 0 && <option disabled>--- Runs ---</option>}
            {runs.map(run => (
              <option key={run.run_id} value={run.run_id}>
                Run: {formatRunDate(run.created_at)}{run.model_profile ? ` - ${run.model_profile}` : ''}{run.language ? ` (lang: ${run.language})` : ''} ({run.artifact_count} artifacts)
              </option>
            ))}
          </select>
        </div>

        {taskLanguages.length > 1 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <label style={{ color: '#999', fontSize: '14px' }}>Language:</label>
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
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
              <option value="all">All ({ocr.length})</option>
              {taskLanguages.map(lang => (
                <option key={lang} value={lang}>
                  {lang.toUpperCase()} ({ocr.filter(item => getLanguage(item) === lang).length})
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
      }}>
        {filteredOcr.map(item => (
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
            <div style={{ fontSize: '12px', color: '#1976d2', marginBottom: '5px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>{formatTime(item.span_start_ms)}</span>
              <span
                style={{
                  backgroundColor: '#444',
                  color: '#fff',
                  padding: '2px 6px',
                  borderRadius: '3px',
                  fontSize: '10px',
                  fontWeight: 'bold',
                  textTransform: 'uppercase',
                }}
              >
                {getLanguage(item)}
              </span>
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
