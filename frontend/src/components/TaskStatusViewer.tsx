import { useEffect, useState } from 'react';

interface Task {
  task_id: string;
  video_id: string;
  task_type: string;
  status: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  language?: string;
}

interface Props {
  videoId: string;
  apiUrl?: string;
}

export default function TaskStatusViewer({ videoId, apiUrl = 'http://localhost:8080' }: Props) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTasks = async () => {
      try {
        const response = await fetch(`${apiUrl}/api/v1/videos/${videoId}/tasks`);
        if (!response.ok) {
          throw new Error('Failed to fetch tasks');
        }
        const data = await response.json();
        setTasks(data);
        setLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        setLoading(false);
      }
    };

    fetchTasks();
    const interval = setInterval(fetchTasks, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, [videoId, apiUrl]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return '#4caf50'; // green
      case 'running':
        return '#2196f3'; // blue
      case 'pending':
        return '#ff9800'; // orange
      case 'failed':
        return '#f44336'; // red
      default:
        return '#999';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return '✓';
      case 'running':
        return '⟳';
      case 'pending':
        return '⋯';
      case 'failed':
        return '✕';
      default:
        return '?';
    }
  };

  const taskTypeLabels: Record<string, string> = {
    object_detection: 'Objects',
    face_detection: 'Faces',
    transcription: 'Transcript',
    ocr: 'OCR',
    place_detection: 'Places',
    scene_detection: 'Scenes',
  };

  if (loading) {
    return <div style={{ padding: '10px', fontSize: '12px', color: '#999' }}>Loading tasks...</div>;
  }

  if (error) {
    return <div style={{ padding: '10px', fontSize: '12px', color: '#ff6b6b' }}>Error: {error}</div>;
  }

  if (tasks.length === 0) {
    return <div style={{ padding: '10px', fontSize: '12px', color: '#999' }}>No tasks</div>;
  }

  const completedCount = tasks.filter(t => t.status === 'completed').length;

  return (
    <div style={{ padding: '10px' }}>
      <div style={{ fontSize: '12px', color: '#999', marginBottom: '8px' }}>
        Tasks: {completedCount}/{tasks.length} completed
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
        {tasks.map(task => (
          <div
            key={task.task_id}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              padding: '4px 8px',
              backgroundColor: '#2a2a2a',
              borderRadius: '3px',
              border: `1px solid ${getStatusColor(task.status)}`,
              fontSize: '11px',
            }}
            title={`${taskTypeLabels[task.task_type as keyof typeof taskTypeLabels] || task.task_type}: ${task.status}${task.language ? ` (${task.language})` : ''}`}
          >
            <span style={{ color: getStatusColor(task.status), fontWeight: 'bold' }}>
              {getStatusIcon(task.status)}
            </span>
            <span style={{ color: '#fff' }}>
              {taskTypeLabels[task.task_type as keyof typeof taskTypeLabels] || task.task_type}
              {task.language && <span style={{ color: '#999', marginLeft: '4px' }}>({task.language})</span>}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
