import { useRef, useState, useEffect } from 'react';
import SceneDetectionViewer from './SceneDetectionViewer';
import FaceDetectionOverlay from './FaceDetectionViewer';
import ObjectDetectionOverlay from './ObjectDetectionOverlay';
import OCROverlay from './OCROverlay';
import FaceDetectionListViewer from './FaceDetectionListViewer';
import TaskStatusViewer from './TaskStatusViewer';
import TranscriptViewer from './TranscriptViewer';
import ObjectDetectionViewer from './ObjectDetectionViewer';
import OCRViewer from './OCRViewer';
import PlaceDetectionViewer from './PlaceDetectionViewer';
import MetadataViewer from './MetadataViewer';
import JumpNavigationControl from './JumpNavigationControl';

interface Props {
  videoId: string;
  apiUrl?: string;
  onBack: () => void;
}

type ArtifactView = 'scenes' | 'transcript' | 'objects' | 'ocr' | 'places' | 'faces' | 'metadata';

export default function VideoPlayer({ videoId, apiUrl = 'http://localhost:8080', onBack }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [activeView, setActiveView] = useState<ArtifactView>('transcript');
  const [showFaces, setShowFaces] = useState(false);
  const [showObjects, setShowObjects] = useState(false);
  const [showOCR, setShowOCR] = useState(false);
  const [videoName, setVideoName] = useState<string>('');

  useEffect(() => {
    fetch(`${apiUrl}/api/v1/videos/${videoId}`)
      .then(res => res.json())
      .then(data => {
        setVideoName(data.filename || 'Video');
      })
      .catch(err => {
        console.error('Failed to fetch video info:', err);
        setVideoName('Video');
      });
  }, [videoId, apiUrl]);

  useEffect(() => {
    // Push a new history state when entering the video player
    window.history.pushState({ videoPlayer: true }, '');

    const handlePopState = (event: PopStateEvent) => {
      // Only handle if this is our video player state
      if (event.state?.videoPlayer) {
        return;
      }
      onBack();
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [onBack]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', backgroundColor: '#000' }}>
      {/* Header */}
      <div style={{ padding: '10px 20px', backgroundColor: '#1a1a1a', borderBottom: '1px solid #333', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <button
            onClick={onBack}
            style={{
              padding: '8px 16px',
              backgroundColor: '#333',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px',
            }}
          >
            ← Back
          </button>
          {videoName && (
            <h2 style={{ margin: 0, color: '#fff', fontSize: '16px', fontWeight: '600' }}>
              {videoName}
            </h2>
          )}
        </div>
        <TaskStatusViewer videoId={videoId} apiUrl={apiUrl} />
      </div>

      {/* Main content */}
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        {/* Video player */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            backgroundColor: '#000',
            minHeight: '400px',
            overflow: 'hidden',
          }}
        >
          <video
            ref={videoRef}
            controls
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
            }}
            src={`${apiUrl}/api/v1/videos/${videoId}/stream`}
          />
          {(showFaces || showObjects || showOCR) && (
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
              }}
            />
          )}
          <FaceDetectionOverlay
            videoId={videoId}
            videoRef={videoRef}
            canvasRef={canvasRef}
            enabled={showFaces}
            apiUrl={apiUrl}
          />
          <ObjectDetectionOverlay
            videoId={videoId}
            videoRef={videoRef}
            canvasRef={canvasRef}
            enabled={showObjects}
            apiUrl={apiUrl}
          />
          <OCROverlay
            videoId={videoId}
            videoRef={videoRef}
            canvasRef={canvasRef}
            enabled={showOCR}
            apiUrl={apiUrl}
          />
        </div>

        {/* Jump navigation control */}
        <JumpNavigationControl videoId={videoId} videoRef={videoRef} apiUrl={apiUrl} />

        {/* Artifact tabs */}
        <div
          style={{
            display: 'flex',
            gap: '0',
            backgroundColor: '#1a1a1a',
            borderTop: '1px solid #333',
            padding: '0',
            overflowX: 'auto',
          }}
        >
          {(['transcript', 'scenes', 'objects', 'ocr', 'places', 'faces', 'metadata'] as const).map(view => (
            <button
              key={view}
              onClick={() => setActiveView(view)}
              style={{
                padding: '12px 16px',
                backgroundColor: activeView === view ? '#333' : '#1a1a1a',
                color: activeView === view ? '#fff' : '#999',
                border: 'none',
                borderBottom: activeView === view ? '2px solid #1976d2' : '2px solid transparent',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: activeView === view ? '600' : '400',
                transition: 'all 0.2s',
                whiteSpace: 'nowrap',
              }}
            >
              {view === 'transcript' && 'Transcript'}
              {view === 'scenes' && 'Scenes'}
              {view === 'objects' && 'Objects'}
              {view === 'ocr' && 'OCR'}
              {view === 'places' && 'Places'}
              {view === 'faces' && 'Faces'}
              {view === 'metadata' && 'Metadata'}
            </button>
          ))}
          <div style={{ marginLeft: 'auto', display: 'flex', gap: '0', alignItems: 'center' }}>
            <label
              style={{
                padding: '12px 16px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                cursor: 'pointer',
                borderBottom: showFaces ? '2px solid #1976d2' : '2px solid transparent',
                transition: 'all 0.2s',
              }}
            >
              <input
                type="checkbox"
                checked={showFaces}
                onChange={(e) => setShowFaces(e.target.checked)}
                style={{ cursor: 'pointer' }}
              />
              <span style={{ color: showFaces ? '#fff' : '#999', fontSize: '14px' }}>Show Faces</span>
            </label>
            <label
              style={{
                padding: '12px 16px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                cursor: 'pointer',
                borderBottom: showObjects ? '2px solid #1976d2' : '2px solid transparent',
                transition: 'all 0.2s',
              }}
            >
              <input
                type="checkbox"
                checked={showObjects}
                onChange={(e) => setShowObjects(e.target.checked)}
                style={{ cursor: 'pointer' }}
              />
              <span style={{ color: showObjects ? '#fff' : '#999', fontSize: '14px' }}>Objects</span>
            </label>
            <label
              style={{
                padding: '12px 16px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                cursor: 'pointer',
                borderBottom: showOCR ? '2px solid #00c864' : '2px solid transparent',
                transition: 'all 0.2s',
              }}
            >
              <input
                type="checkbox"
                checked={showOCR}
                onChange={(e) => setShowOCR(e.target.checked)}
                style={{ cursor: 'pointer' }}
              />
              <span style={{ color: showOCR ? '#fff' : '#999', fontSize: '14px' }}>OCR</span>
            </label>
          </div>
        </div>

        {/* Artifact viewer */}
        <div
          style={{
            backgroundColor: '#1a1a1a',
            borderTop: '1px solid #333',
            overflow: 'auto',
            flex: 1,
            minHeight: '300px',
          }}
        >
          {activeView === 'transcript' && (
            <TranscriptViewer videoId={videoId} videoRef={videoRef} apiUrl={apiUrl} />
          )}
          {activeView === 'scenes' && (
            <SceneDetectionViewer videoId={videoId} videoRef={videoRef} apiUrl={apiUrl} />
          )}
          {activeView === 'objects' && (
            <ObjectDetectionViewer videoId={videoId} videoRef={videoRef} apiUrl={apiUrl} />
          )}
          {activeView === 'ocr' && (
            <OCRViewer videoId={videoId} videoRef={videoRef} apiUrl={apiUrl} />
          )}
          {activeView === 'places' && (
            <PlaceDetectionViewer videoId={videoId} videoRef={videoRef} apiUrl={apiUrl} />
          )}
          {activeView === 'faces' && (
            <FaceDetectionListViewer videoId={videoId} videoRef={videoRef} apiUrl={apiUrl} />
          )}
          {activeView === 'metadata' && (
            <MetadataViewer videoId={videoId} apiUrl={apiUrl} />
          )}
        </div>
      </div>
    </div>
  );
}
