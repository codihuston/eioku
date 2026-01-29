import React, { useEffect, useState, useMemo } from 'react';

interface TranscriptSegment {
  start_ms: number;
  end_ms: number;
  text: string;
  language?: string;
}

interface Artifact {
  span_start_ms: number;
  span_end_ms: number;
  payload?: {
    text?: string;
    language?: string;
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
  videoRef: React.RefObject<HTMLVideoElement>;
  apiUrl?: string;
}

export default function TranscriptViewer({ videoId, videoRef, apiUrl = 'http://localhost:8080' }: Props) {
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [taskLanguages, setTaskLanguages] = useState<string[]>([]);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [selectedLanguage, setSelectedLanguage] = useState<string>('all');
  const [filterValue, setFilterValue] = useState<string>('all');

  // Fetch transcription tasks to get available languages
  useEffect(() => {
    fetch(`${apiUrl}/api/v1/videos/${videoId}/tasks`)
      .then(res => res.json())
      .then((tasks: Task[]) => {
        const transcriptionTasks = tasks.filter(t => t.task_type === 'transcription');
        const langs = transcriptionTasks
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
    fetch(`${apiUrl}/api/v1/videos/${videoId}/runs?artifact_type=transcript.segment`)
      .then(res => res.json())
      .then(data => {
        setRuns(data.runs || []);
      })
      .catch(err => {
        console.error('Failed to fetch runs:', err);
      });
  }, [videoId, apiUrl]);

  // Fetch transcript artifacts based on the selected filter
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

    fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts?type=transcript.segment${selectionParam}`)
      .then(res => res.json())
      .then((data: Artifact[]) => {
        const transcriptSegments = data.map(artifact => ({
          start_ms: artifact.span_start_ms,
          end_ms: artifact.span_end_ms,
          text: artifact.payload?.text || '',
          language: artifact.payload?.language,
        }));
        setSegments(transcriptSegments);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [videoId, apiUrl, filterValue]);

  // Filter segments by selected language
  const filteredSegments = useMemo(() => {
    if (selectedLanguage === 'all') return segments;
    return segments.filter(seg => seg.language === selectedLanguage);
  }, [segments, selectedLanguage]);

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

  const formatRunDate = (dateString: string) => {
    return new Date(dateString).toLocaleString(undefined, {
      dateStyle: 'short',
      timeStyle: 'short',
    });
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
        backgroundColor: '#1a1a1a',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '10px', flexWrap: 'wrap' }}>
        <h3 style={{ margin: '0', fontSize: '14px', fontWeight: '600', color: '#fff' }}>Transcript</h3>
        <span style={{ color: '#999', fontSize: '12px' }}>
          {filteredSegments.length} {selectedLanguage !== 'all' && `of ${segments.length}`} segments
        </span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <label style={{ color: '#999', fontSize: '12px' }}>Run:</label>
          <select
            value={filterValue}
            onChange={(e) => setFilterValue(e.target.value)}
            style={{
              padding: '4px 8px',
              backgroundColor: '#2a2a2a',
              color: '#fff',
              border: '1px solid #444',
              borderRadius: '4px',
              fontSize: '12px',
              cursor: 'pointer',
            }}
          >
            <option value="all">All runs</option>
            <option value="latest_per_language">Latest per language</option>
            <option value="latest">Latest only</option>
            {runs.length > 0 && <option disabled>--- Runs ---</option>}
            {runs.map(run => (
              <option key={run.run_id} value={run.run_id}>
                Run: {formatRunDate(run.created_at)}{run.model_profile ? ` - ${run.model_profile}` : ''}{run.language ? ` (lang: ${run.language})` : ''} ({run.artifact_count} segments)
              </option>
            ))}
          </select>
        </div>

        {taskLanguages.length > 1 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <label style={{ color: '#999', fontSize: '12px' }}>Language:</label>
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              style={{
                padding: '4px 8px',
                backgroundColor: '#2a2a2a',
                color: '#fff',
                border: '1px solid #444',
                borderRadius: '4px',
                fontSize: '12px',
                cursor: 'pointer',
              }}
            >
              <option value="all">All ({segments.length})</option>
              {taskLanguages.map(lang => (
                <option key={lang} value={lang}>
                  {lang.toUpperCase()} ({segments.filter(s => s.language === lang).length})
                </option>
              ))}
            </select>
          </div>
        )}
      </div>
      <div>
        {filteredSegments.map((segment, idx) => {
          const isActive = currentTime >= segment.start_ms && currentTime < segment.end_ms;
          return (
            <div
              key={idx}
              onClick={() => handleSegmentClick(segment.start_ms)}
              style={{
                padding: '8px',
                marginBottom: '4px',
                backgroundColor: isActive ? '#2a3a4a' : 'transparent',
                borderLeft: isActive ? '3px solid #1976d2' : '3px solid transparent',
                cursor: 'pointer',
                transition: 'background-color 0.2s',
                borderRadius: '2px',
              }}
              onMouseEnter={e => {
                const el = e.currentTarget as HTMLDivElement;
                if (!isActive) {
                  el.style.backgroundColor = '#252525';
                }
              }}
              onMouseLeave={e => {
                const el = e.currentTarget as HTMLDivElement;
                if (!isActive) {
                  el.style.backgroundColor = 'transparent';
                }
              }}
            >
              <div style={{ fontSize: '11px', color: '#1976d2', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span>{formatTime(segment.start_ms)}</span>
                {segment.language && (
                  <span
                    style={{
                      backgroundColor: '#444',
                      color: '#fff',
                      padding: '1px 5px',
                      borderRadius: '3px',
                      fontSize: '9px',
                      fontWeight: 'bold',
                      textTransform: 'uppercase',
                    }}
                  >
                    {segment.language}
                  </span>
                )}
              </div>
              <div style={{ fontSize: '13px', lineHeight: '1.4', color: '#ddd' }}>{segment.text}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
