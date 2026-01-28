import { useState } from 'react';
import VideoGallery from './components/VideoGallery';
import VideoPlayer from './components/VideoPlayer';

function App() {
  const [view, setView] = useState<'gallery' | 'player'>('gallery');
  const [selectedVideoId, setSelectedVideoId] = useState('');

  const handleSelectVideo = (videoId: string) => {
    setSelectedVideoId(videoId);
    setView('player');
  };

  const handleBack = () => {
    setSelectedVideoId('');
    setView('gallery');
  };

  if (view === 'player' && selectedVideoId) {
    return <VideoPlayer videoId={selectedVideoId} onBack={handleBack} />;
  }

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#fff' }}>
      <div style={{ padding: '20px', borderBottom: '1px solid #eee' }}>
        <h1 style={{ margin: '0 0 5px 0', fontSize: '28px' }}>Eioku</h1>
        <p style={{ margin: '0', color: '#666', fontSize: '14px' }}>Semantic Video Search Platform</p>
      </div>
      <VideoGallery onSelectVideo={handleSelectVideo} />
    </div>
  );
}

export default App;
