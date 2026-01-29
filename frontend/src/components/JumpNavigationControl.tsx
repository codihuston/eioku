import React, { useEffect, useState, useCallback } from 'react';

interface Artifact {
  artifact_id: string;
  artifact_type: string;
  payload: Record<string, unknown>;
}

interface ArtifactOption {
  type: string;
  label?: string;
  clusterId?: string;
  confidence?: number;
  count: number;
}

interface Props {
  videoId: string;
  videoRef?: React.RefObject<HTMLVideoElement>;
  apiUrl?: string;
}

export default function JumpNavigationControl({ videoId, videoRef, apiUrl = 'http://localhost:8080' }: Props) {
  const [artifactTypes, setArtifactTypes] = useState<string[]>([]);
  const [selectedType, setSelectedType] = useState<string>('');
  const [options, setOptions] = useState<ArtifactOption[]>([]);
  const [allArtifacts, setAllArtifacts] = useState<Artifact[]>([]);
  const [selectedOptions, setSelectedOptions] = useState<Set<string>>(new Set());
  const [confidenceThreshold, setConfidenceThreshold] = useState(0);
  const [jumpTime, setJumpTime] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentMatch, setCurrentMatch] = useState<string>('');
  const [lastArtifactStartMs, setLastArtifactStartMs] = useState<number | null>(null);

  // Fetch available artifact types
  useEffect(() => {
    const fetchArtifactTypes = async () => {
      try {
        const response = await fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts`);
        const artifacts: Artifact[] = await response.json();

        // Extract unique artifact types, excluding video.metadata
        const types = [...new Set(artifacts.map((a: Artifact) => a.artifact_type))].filter(
          type => type !== 'video.metadata'
        );
        setArtifactTypes(types);
        if (types.length > 0) {
          setSelectedType(types[0]);
        }
      } catch (err) {
        console.error('Failed to fetch artifact types:', err);
      }
    };

    fetchArtifactTypes();
  }, [videoId, apiUrl]);

  // Define calculateOptions with useCallback to memoize it
  const calculateOptions = useCallback((artifacts: Artifact[], threshold: number) => {
    // Aggregate options based on artifact type
    const optionMap = new Map<string, ArtifactOption>();

    artifacts.forEach((artifact: Artifact) => {
      const payload = artifact.payload;
      const confidence = payload.confidence as number;

      // Skip artifacts below confidence threshold
      if (threshold > 0 && confidence !== undefined && confidence < threshold) {
        return;
      }

      let key = '';
      let label = '';

      if (selectedType === 'object.detection') {
        key = `obj_${payload.label}`;
        label = payload.label as string;
      } else if (selectedType === 'face.detection') {
        key = `face_${payload.cluster_id || 'unknown'}`;
        label = payload.cluster_id ? `Face Cluster ${(payload.cluster_id as string).slice(0, 8)}` : 'Unknown Face';
      } else if (selectedType === 'place.classification') {
        key = `place_${payload.label}`;
        label = payload.label as string;
      } else if (selectedType === 'transcript.segment') {
        key = 'transcript';
        label = 'Transcript';
      } else if (selectedType === 'ocr.text') {
        key = 'ocr';
        label = 'OCR Text';
      } else if (selectedType === 'scene') {
        key = `scene_${payload.scene_index}`;
        label = `Scene ${payload.scene_index}`;
      }

      if (key) {
        const existing = optionMap.get(key) || {
          type: selectedType,
          label,
          confidence: confidence,
          count: 0,
        };
        existing.count += 1;
        optionMap.set(key, existing);
      }
    });

    setOptions(Array.from(optionMap.values()).sort((a, b) => b.count - a.count));
  }, [selectedType]);

  // Fetch options for selected artifact type
  useEffect(() => {
    if (!selectedType) return;

    const fetchOptions = async () => {
      try {
        const response = await fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts?type=${selectedType}`);
        const artifacts: Artifact[] = await response.json();
        setAllArtifacts(artifacts);

        // Calculate options with current confidence threshold
        calculateOptions(artifacts, confidenceThreshold);
        setSelectedOptions(new Set());
      } catch (err) {
        console.error('Failed to fetch options:', err);
      }
    };

    fetchOptions();
  }, [selectedType, videoId, apiUrl, confidenceThreshold, calculateOptions]);

  // Recalculate options when confidence threshold changes
  useEffect(() => {
    if (allArtifacts.length > 0) {
      calculateOptions(allArtifacts, confidenceThreshold);
    }
  }, [confidenceThreshold, allArtifacts, calculateOptions]);

  const toggleOption = (option: ArtifactOption) => {
    const key = `${option.label}`;
    const newSelected = new Set(selectedOptions);
    if (newSelected.has(key)) {
      newSelected.delete(key);
    } else {
      newSelected.add(key);
    }
    setSelectedOptions(newSelected);
  };

  const jump = async (direction: 'next' | 'prev') => {
    if (!videoRef?.current) return;

    setLoading(true);
    try {
      let currentMs = Math.floor(videoRef.current.currentTime * 1000);
      
      // For prev, if we just jumped to an artifact, search from its start time
      // to avoid returning the same artifact again
      if (direction === 'prev' && lastArtifactStartMs !== null) {
        currentMs = lastArtifactStartMs;
        console.log(`Using last artifact start time ${currentMs}ms for prev search`);
      }
      
      const labels = Array.from(selectedOptions).join(',');

      const params = new URLSearchParams({
        kind: selectedType.split('.')[0],
        direction,
        from_ms: currentMs.toString(),
        ...(labels && { label: labels }),
        ...(confidenceThreshold > 0 && { min_confidence: confidenceThreshold.toString() }),
      });

      console.log(`Jump ${direction} from ${currentMs}ms with kind=${selectedType.split('.')[0]}`);
      const response = await fetch(`${apiUrl}/api/v1/videos/${videoId}/jump?${params}`);
      const data = await response.json();

      console.log(`Jump response:`, data);

      if (data.jump_to) {
        const targetTime = data.jump_to.start_ms / 1000;
        console.log(`Setting video currentTime from ${videoRef.current.currentTime} to ${targetTime}`);
        
        // Create a promise that resolves when the seek completes
        const seekPromise = new Promise<void>((resolve) => {
          const handleSeeked = () => {
            console.log(`Seek completed, currentTime is now ${videoRef.current?.currentTime}`);
            videoRef.current?.removeEventListener('seeked', handleSeeked);
            resolve();
          };
          videoRef.current?.addEventListener('seeked', handleSeeked);
        });
        
        // Set the time and wait for seek to complete
        videoRef.current.currentTime = targetTime;
        
        // Wait for seek to complete, then play
        seekPromise.then(() => {
          if (videoRef.current) {
            videoRef.current.play().catch(err => console.error('Play failed:', err));
          }
        });
        
        // Store the start time of this artifact for the next prev jump
        setLastArtifactStartMs(data.jump_to.start_ms);
        setCurrentMatch(`${selectedType} @ ${formatTime(data.jump_to.start_ms)}`);
      } else {
        console.warn(`No jump_to in response for ${direction}`);
        setCurrentMatch(`No ${selectedType} found ${direction}`);
        setLastArtifactStartMs(null);
      }
    } catch (err) {
      console.error('Jump failed:', err);
      setCurrentMatch(`Error: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setLastArtifactStartMs(null);
    } finally {
      setLoading(false);
    }
  };

  const jumpToTime = () => {
    if (!videoRef?.current || !jumpTime) return;

    const [minutes, seconds] = jumpTime.split(':').map(Number);
    const totalSeconds = (minutes || 0) * 60 + (seconds || 0);
    videoRef.current.currentTime = totalSeconds;
    videoRef.current.play();
    setCurrentMatch(`Jumped to ${jumpTime}`);
  };

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  const getTypeLabel = (type: string) => {
    const labels: Record<string, string> = {
      'object.detection': 'Objects',
      'face.detection': 'Faces',
      'place.classification': 'Places',
      'transcript.segment': 'Transcript',
      'ocr.text': 'OCR',
      'scene': 'Scenes',
    };
    return labels[type] || type;
  };

  return (
    <div style={{
      backgroundColor: '#1a1a1a',
      borderTop: '1px solid #333',
      padding: '16px',
      display: 'flex',
      flexDirection: 'column',
      gap: '12px',
    }}>
      {/* Type selector */}
      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
        <span style={{ color: '#999', fontSize: '12px', minWidth: '60px' }}>Jump to:</span>
        <select
          value={selectedType}
          onChange={(e) => setSelectedType(e.target.value)}
          style={{
            padding: '6px 10px',
            backgroundColor: '#2a2a2a',
            color: '#fff',
            border: '1px solid #444',
            borderRadius: '4px',
            fontSize: '12px',
            cursor: 'pointer',
          }}
        >
          {artifactTypes.map(type => (
            <option key={type} value={type}>{getTypeLabel(type)}</option>
          ))}
        </select>
      </div>

      {/* Options chips */}
      {options.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {options.map((option, idx) => (
            <button
              key={idx}
              onClick={() => toggleOption(option)}
              style={{
                padding: '6px 12px',
                backgroundColor: selectedOptions.has(option.label || '') ? '#1976d2' : '#2a2a2a',
                color: '#fff',
                border: `1px solid ${selectedOptions.has(option.label || '') ? '#1976d2' : '#444'}`,
                borderRadius: '16px',
                fontSize: '12px',
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
            >
              {option.label} ({option.count})
            </button>
          ))}
        </div>
      )}

      {/* Confidence threshold */}
      {selectedType !== 'transcript.segment' && selectedType !== 'scene' && (
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span style={{ color: '#999', fontSize: '12px', minWidth: '60px' }}>Confidence:</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={confidenceThreshold}
            onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
            style={{ flex: 1, maxWidth: '200px' }}
          />
          <span style={{ color: '#999', fontSize: '12px', minWidth: '40px' }}>
            {(confidenceThreshold * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* Jump to timestamp */}
      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
        <span style={{ color: '#999', fontSize: '12px', minWidth: '60px' }}>Jump to:</span>
        <input
          type="text"
          placeholder="MM:SS"
          value={jumpTime}
          onChange={(e) => setJumpTime(e.target.value)}
          style={{
            padding: '6px 10px',
            backgroundColor: '#2a2a2a',
            color: '#fff',
            border: '1px solid #444',
            borderRadius: '4px',
            fontSize: '12px',
            width: '80px',
          }}
        />
        <button
          onClick={jumpToTime}
          style={{
            padding: '6px 12px',
            backgroundColor: '#2a2a2a',
            color: '#fff',
            border: '1px solid #444',
            borderRadius: '4px',
            fontSize: '12px',
            cursor: 'pointer',
          }}
        >
          Go
        </button>
      </div>

      {/* Navigation buttons */}
      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
        <button
          onClick={() => jump('prev')}
          disabled={loading}
          style={{
            padding: '8px 16px',
            backgroundColor: '#2a2a2a',
            color: '#fff',
            border: '1px solid #444',
            borderRadius: '4px',
            fontSize: '12px',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.5 : 1,
          }}
        >
          ← Previous
        </button>

        <button
          onClick={() => jump('next')}
          disabled={loading}
          style={{
            padding: '8px 16px',
            backgroundColor: '#2a2a2a',
            color: '#fff',
            border: '1px solid #444',
            borderRadius: '4px',
            fontSize: '12px',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.5 : 1,
          }}
        >
          Next →
        </button>

        {currentMatch && (
          <span style={{ color: '#999', fontSize: '12px', marginLeft: 'auto' }}>
            {currentMatch}
          </span>
        )}
      </div>
    </div>
  );
}
