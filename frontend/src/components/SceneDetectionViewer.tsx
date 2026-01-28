import { useEffect, useRef, useState } from 'react';

interface Artifact {
  artifact_id: string;
  span_start_ms: number;
  span_end_ms: number;
}

interface Scene {
  artifact_id: string;
  span_start_ms: number;
  span_end_ms: number;
  duration_ms: number;
}

interface Props {
  videoId: string;
  apiUrl?: string;
}

export default function SceneDetectionViewer({ videoId, apiUrl = 'http://localhost:8080' }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentSceneIndex, setCurrentSceneIndex] = useState(0);

  // Fetch scenes for the video
  useEffect(() => {
    fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts?type=scene`)
      .then(res => res.json())
      .then(data => {
        // Extract scene data from artifacts
        const sceneList = (data as Artifact[]).map((artifact: Artifact) => ({
          artifact_id: artifact.artifact_id,
          span_start_ms: artifact.span_start_ms,
          span_end_ms: artifact.span_end_ms,
          duration_ms: artifact.span_end_ms - artifact.span_start_ms,
        }));
        setScenes(sceneList);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [videoId, apiUrl]);

  // Update current scene index when video time changes
  useEffect(() => {
    const video = videoRef.current;
    if (!video || scenes.length === 0) return;

    const handleTimeUpdate = () => {
      const currentTimeMs = video.currentTime * 1000;
      const index = scenes.findIndex(
        scene => currentTimeMs >= scene.span_start_ms && currentTimeMs < scene.span_end_ms
      );
      if (index >= 0) {
        setCurrentSceneIndex(index);
      }
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    return () => video.removeEventListener('timeupdate', handleTimeUpdate);
  }, [scenes]);

  const jumpToScene = (index: number) => {
    if (videoRef.current && scenes[index]) {
      const targetTime = scenes[index].span_start_ms / 1000;
      videoRef.current.currentTime = targetTime;
      videoRef.current.play();
      setCurrentSceneIndex(index);
    }
  };

  const jumpNext = () => {
    if (currentSceneIndex < scenes.length - 1) {
      jumpToScene(currentSceneIndex + 1);
    }
  };

  const jumpPrev = () => {
    if (currentSceneIndex > 0) {
      jumpToScene(currentSceneIndex - 1);
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
    return <div style={{ padding: '20px' }}>Loading scene detection data...</div>;
  }

  if (error) {
    return <div style={{ padding: '20px', color: 'red' }}>Error: {error}</div>;
  }

  if (scenes.length === 0) {
    return <div style={{ padding: '20px' }}>No scenes detected for this video</div>;
  }

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h2>Scene Detection Viewer</h2>

      <div style={{ marginBottom: '20px' }}>
        <p><strong>Video ID:</strong> {videoId}</p>
        <p><strong>Total Scenes:</strong> {scenes.length}</p>
        <p><strong>Current Scene:</strong> {currentSceneIndex + 1} / {scenes.length}</p>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <video
          ref={videoRef}
          controls
          style={{
            width: '100%',
            maxWidth: '800px',
            display: 'block',
            backgroundColor: '#000'
          }}
          src={`${apiUrl}/api/v1/videos/${videoId}/stream`}
        />
      </div>

      <div style={{ marginBottom: '20px', display: 'flex', gap: '10px' }}>
        <button
          onClick={jumpPrev}
          disabled={currentSceneIndex === 0}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            cursor: currentSceneIndex === 0 ? 'not-allowed' : 'pointer',
            opacity: currentSceneIndex === 0 ? 0.5 : 1
          }}
        >
          ← Previous Scene
        </button>

        <button
          onClick={jumpNext}
          disabled={currentSceneIndex === scenes.length - 1}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            cursor: currentSceneIndex === scenes.length - 1 ? 'not-allowed' : 'pointer',
            opacity: currentSceneIndex === scenes.length - 1 ? 0.5 : 1
          }}
        >
          Next Scene →
        </button>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h3>Scene List</h3>
        <div style={{
          maxHeight: '400px',
          overflowY: 'auto',
          border: '1px solid #ccc',
          borderRadius: '4px'
        }}>
          {scenes.map((scene, index) => (
            <div
              key={scene.artifact_id}
              onClick={() => jumpToScene(index)}
              style={{
                padding: '10px',
                borderBottom: '1px solid #eee',
                cursor: 'pointer',
                backgroundColor: index === currentSceneIndex ? '#e3f2fd' : 'transparent',
                fontWeight: index === currentSceneIndex ? 'bold' : 'normal'
              }}
            >
              <div>
                Scene {index + 1}: {formatTime(scene.span_start_ms)} - {formatTime(scene.span_end_ms)}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>
                Duration: {formatTime(scene.duration_ms)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
