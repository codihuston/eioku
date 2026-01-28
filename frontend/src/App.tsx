import { useState } from 'react';
import SceneDetectionViewer from './components/SceneDetectionViewer';
import FaceDetectionViewer from './components/FaceDetectionViewer';

function App() {
  const [view, setView] = useState<'home' | 'scenes' | 'faces'>('home');
  const [videoId, setVideoId] = useState('');
  const [inputVideoId, setInputVideoId] = useState('');

  const handleViewScenes = () => {
    if (inputVideoId.trim()) {
      setVideoId(inputVideoId);
      setView('scenes');
    }
  };

  const handleViewFaces = () => {
    if (inputVideoId.trim()) {
      setVideoId(inputVideoId);
      setView('faces');
    }
  };

  if (view === 'scenes' && videoId) {
    return (
      <div>
        <button onClick={() => setView('home')} style={{ margin: '10px' }}>
          ← Back to Home
        </button>
        <SceneDetectionViewer videoId={videoId} />
      </div>
    );
  }

  if (view === 'faces' && videoId) {
    return (
      <div>
        <button onClick={() => setView('home')} style={{ margin: '10px' }}>
          ← Back to Home
        </button>
        <FaceDetectionViewer videoId={videoId} />
      </div>
    );
  }

  return (
    <div style={{ padding: '40px', maxWidth: '600px', margin: '0 auto' }}>
      <h1>Eioku</h1>
      <p>Semantic Video Search Platform</p>

      <div style={{ marginTop: '40px', padding: '20px', border: '1px solid #ccc', borderRadius: '8px' }}>
        <h2>Video Viewers</h2>
        
        <div style={{ marginBottom: '20px' }}>
          <label>
            Video ID:
            <input
              type="text"
              value={inputVideoId}
              onChange={(e) => setInputVideoId(e.target.value)}
              placeholder="Enter video ID"
              style={{
                marginLeft: '10px',
                padding: '8px',
                width: '300px',
                fontSize: '14px'
              }}
            />
          </label>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={handleViewScenes}
            disabled={!inputVideoId.trim()}
            style={{
              padding: '10px 20px',
              fontSize: '16px',
              cursor: inputVideoId.trim() ? 'pointer' : 'not-allowed',
              opacity: inputVideoId.trim() ? 1 : 0.5
            }}
          >
            View Scenes
          </button>
          <button
            onClick={handleViewFaces}
            disabled={!inputVideoId.trim()}
            style={{
              padding: '10px 20px',
              fontSize: '16px',
              cursor: inputVideoId.trim() ? 'pointer' : 'not-allowed',
              opacity: inputVideoId.trim() ? 1 : 0.5
            }}
          >
            View Faces
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
