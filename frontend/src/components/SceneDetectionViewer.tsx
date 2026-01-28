import React, { useEffect, useState } from 'react';

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
  videoRef?: React.RefObject<HTMLVideoElement>;
  apiUrl?: string;
}

export default function SceneDetectionViewer({ videoId, videoRef, apiUrl = 'http://localhost:8080' }: Props) {
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
    const video = videoRef?.current;
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
  }, [scenes, videoRef]);

  const jumpToScene = (index: number) => {
    if (videoRef?.current && scenes[index]) {
      const targetTime = scenes[index].span_start_ms / 1000;
      videoRef.current.currentTime = targetTime;
      videoRef.current.play();
      setCurrentSceneIndex(index);
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
    return <div style={{ padding: '20px', fontSize: '14px', color: '#999' }}>Loading scenes...</div>;
  }

  if (error) {
    return <div style={{ padding: '20px', color: '#ff6b6b' }}>Error: {error}</div>;
  }

  if (scenes.length === 0) {
    return <div style={{ padding: '20px', fontSize: '14px', color: '#999' }}>No scenes detected</div>;
  }

  return (
    <div style={{ padding: '20px' }}>
      <div style={{
        maxHeight: '600px',
        overflowY: 'auto',
      }}>
        {scenes.map((scene, index) => (
          <div
            key={scene.artifact_id}
            onClick={() => jumpToScene(index)}
            style={{
              padding: '12px',
              borderBottom: '1px solid #333',
              cursor: 'pointer',
              backgroundColor: index === currentSceneIndex ? '#2a4a6a' : 'transparent',
              fontWeight: index === currentSceneIndex ? '600' : '400',
              transition: 'background-color 0.2s',
            }}
          >
            <div style={{ color: '#fff' }}>
              Scene {index + 1}: {formatTime(scene.span_start_ms)} - {formatTime(scene.span_end_ms)}
            </div>
            <div style={{ fontSize: '12px', color: '#999', marginTop: '4px' }}>
              Duration: {formatTime(scene.duration_ms)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
