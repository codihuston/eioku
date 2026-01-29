import { useEffect, useRef, useState } from 'react';

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
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [openMenuTaskId, setOpenMenuTaskId] = useState<string | null>(null);
  const [requeueing, setRequeueing] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const fetchTasks = async () => {
    try {
      setIsRefreshing(true);
      const response = await fetch(`${apiUrl}/api/v1/videos/${videoId}/tasks`);
      if (!response.ok) {
        throw new Error('Failed to fetch tasks');
      }
      const data = await response.json();
      setTasks(data);
      setLoading(false);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setLoading(false);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleRequeue = async (taskId: string) => {
    try {
      setRequeueing(taskId);
      const response = await fetch(`${apiUrl}/api/v1/tasks/${taskId}/retry`, {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Failed to requeue task');
      }
      setOpenMenuTaskId(null);
      await fetchTasks();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to requeue task');
    } finally {
      setRequeueing(null);
    }
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setOpenMenuTaskId(null);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    fetchTasks();
    const interval = setInterval(fetchTasks, 10000); // Refresh every 10 seconds

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  const formatTime = (isoString: string | undefined) => {
    if (!isoString) return null;
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const calculateDuration = (startTime: string | undefined, endTime: string | undefined) => {
    if (!startTime || !endTime) return null;
    const start = new Date(startTime).getTime();
    const end = new Date(endTime).getTime();
    const seconds = Math.floor((end - start) / 1000);
    
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  const calculateQueueTime = (createdTime: string, startTime: string | undefined) => {
    if (!startTime) return null;
    const created = new Date(createdTime).getTime();
    const started = new Date(startTime).getTime();
    const seconds = Math.floor((started - created) / 1000);
    
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
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
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
        <button
          onClick={fetchTasks}
          disabled={isRefreshing}
          style={{
            padding: '4px 8px',
            fontSize: '12px',
            backgroundColor: '#2a2a2a',
            color: '#fff',
            border: '1px solid #444',
            borderRadius: '3px',
            cursor: isRefreshing ? 'not-allowed' : 'pointer',
            opacity: isRefreshing ? 0.6 : 1,
            transition: 'opacity 0.2s',
          }}
          title="Refresh task status"
        >
          {isRefreshing ? '⟳ Refreshing...' : '⟳ Refresh'}
        </button>
        <div style={{ fontSize: '12px', color: '#999' }}>
          Tasks: {completedCount}/{tasks.length} completed
        </div>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
        {tasks.map(task => {
          const queueTime = calculateQueueTime(task.created_at, task.started_at);
          const duration = calculateDuration(task.started_at, task.completed_at);
          const startTimeStr = formatTime(task.started_at);
          const endTimeStr = formatTime(task.completed_at);
          
          const tooltipLines = [
            `${taskTypeLabels[task.task_type as keyof typeof taskTypeLabels] || task.task_type}: ${task.status}${task.language ? ` (${task.language})` : ''}`,
            `Queued: ${formatTime(task.created_at)}`,
            ...(startTimeStr ? [`Started: ${startTimeStr}`] : []),
            ...(endTimeStr ? [`Ended: ${endTimeStr}`] : []),
            ...(queueTime ? [`Queue time: ${queueTime}`] : []),
            ...(duration ? [`Duration: ${duration}`] : []),
          ];
          
          return (
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
                position: 'relative',
              }}
              title={tooltipLines.join('\n')}
            >
              <span style={{ color: getStatusColor(task.status), fontWeight: 'bold' }}>
                {getStatusIcon(task.status)}
              </span>
              <span style={{ color: '#fff' }}>
                {taskTypeLabels[task.task_type as keyof typeof taskTypeLabels] || task.task_type}
                {task.language && <span style={{ color: '#999', marginLeft: '4px' }}>({task.language})</span>}
              </span>
              {duration && <span style={{ color: '#999', marginLeft: '4px', fontSize: '10px' }}>({duration})</span>}
              
              {task.status === 'failed' && (
                <div style={{ position: 'relative', marginLeft: '4px' }}>
                  <button
                    onClick={() => setOpenMenuTaskId(openMenuTaskId === task.task_id ? null : task.task_id)}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#999',
                      cursor: 'pointer',
                      padding: '0 2px',
                      fontSize: '14px',
                      display: 'flex',
                      alignItems: 'center',
                    }}
                    title="Task options"
                  >
                    ⋮
                  </button>
                  
                  {openMenuTaskId === task.task_id && (
                    <div
                      ref={menuRef}
                      style={{
                        position: 'absolute',
                        top: '100%',
                        right: 0,
                        backgroundColor: '#1a1a1a',
                        border: '1px solid #444',
                        borderRadius: '3px',
                        minWidth: '120px',
                        zIndex: 1000,
                        boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
                      }}
                    >
                      <button
                        onClick={() => handleRequeue(task.task_id)}
                        disabled={requeueing === task.task_id}
                        style={{
                          width: '100%',
                          padding: '8px 12px',
                          backgroundColor: 'transparent',
                          color: requeueing === task.task_id ? '#666' : '#fff',
                          border: 'none',
                          textAlign: 'left',
                          cursor: requeueing === task.task_id ? 'not-allowed' : 'pointer',
                          fontSize: '12px',
                          transition: 'background-color 0.2s',
                        }}
                        onMouseEnter={(e) => {
                          if (requeueing !== task.task_id) {
                            (e.target as HTMLButtonElement).style.backgroundColor = '#333';
                          }
                        }}
                        onMouseLeave={(e) => {
                          (e.target as HTMLButtonElement).style.backgroundColor = 'transparent';
                        }}
                      >
                        {requeueing === task.task_id ? '⟳ Requeuing...' : '⟳ Requeue'}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
