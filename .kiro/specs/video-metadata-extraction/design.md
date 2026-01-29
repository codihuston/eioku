# Design Document: Video Metadata Extraction

## Overview

The Video Metadata Extraction feature integrates metadata extraction into the video discovery pipeline. When videos are discovered, a language-agnostic `metadata_extraction` task is automatically created and enqueued. The ML worker processes this task using pyexiftool to extract standardized Composite fields from video files. Extracted metadata is persisted as artifact envelopes with type `video.metadata`, and GPS coordinates are normalized into a `video_locations` projection table for future geo-spatial queries. The frontend displays extracted metadata in a new "Metadata" tab in the video player.

## Architecture

### System Components

```
Video Discovery Pipeline
    ↓
[VideoDiscoveryService] - Detects new videos, creates metadata_extraction task
    ↓
[JobProducer] - Enqueues task to Redis job queue
    ↓
[ML Worker] - Processes metadata_extraction task
    ↓
[MetadataExtractor] - Runs pyexiftool, extracts Composite fields
    ↓
[ArtifactTransformer] - Converts extraction results to artifact envelopes
    ↓
[ArtifactRepository] - Persists artifacts to PostgreSQL
    ↓
[ProjectionSyncService] - Syncs GPS data to video_locations table
    ↓
[Frontend] - Displays metadata in VideoPlayer UI
```

### Data Flow

1. **Discovery**: VideoDiscoveryService discovers video file
2. **Task Creation**: Creates metadata_extraction task with language=NULL
3. **Enqueueing**: JobProducer enqueues task to Redis
4. **Extraction**: ML worker runs MetadataExtractor on video file
5. **Transformation**: ArtifactTransformer converts results to artifact envelope
6. **Persistence**: ArtifactRepository saves artifact to artifacts table
7. **Projection Sync**: ProjectionSyncService creates video_locations entry if GPS exists
8. **Display**: Frontend fetches artifact and displays in MetadataViewer component

## Components and Interfaces

### Backend Components

#### 1. MetadataExtractor (ML Service)

Extracts standardized metadata from video files using pyexiftool.

```python
class MetadataExtractor:
    """Extract metadata from video files using pyexiftool."""
    
    def extract(self, video_path: str) -> dict:
        """
        Extract metadata from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with extracted metadata fields (no null values)
            
        Raises:
            MetadataExtractionError: If extraction fails
        """
        # Implementation details in tasks
```

**Composite Fields Extracted**:
- GPS: Composite:GPSLatitude, Composite:GPSLongitude, Composite:GPSAltitude
- Image: Composite:ImageSize, Composite:Megapixels, Composite:Rotation
- Audio/Video: Composite:AvgBitrate, Composite:Duration, Composite:VideoFrameRate, Composite:VideoCodec
- File: Composite:FileSize, Composite:FileType, Composite:MIMEType
- Camera: Composite:Make, Composite:Model
- Temporal: Composite:CreateDate
- QuickTime Fallbacks: QuickTime:GPSCoordinates, QuickTime:Duration, QuickTime:CreateDate

**Output Format**:
```python
{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "altitude": 10.5,
    "image_size": "1920x1080",
    "megapixels": 2.07,
    "rotation": 0,
    "avg_bitrate": "5000k",
    "duration_seconds": 120.5,
    "frame_rate": 29.97,
    "codec": "h264",
    "file_size": 75000000,
    "file_type": "video",
    "mime_type": "video/mp4",
    "camera_make": "Canon",
    "camera_model": "EOS R5",
    "create_date": "2024-01-15T10:30:00Z"
}
```

#### 2. Metadata Artifact Schema (v1)

Defines the structure of video.metadata artifacts.

```python
class MetadataV1(BaseModel):
    """Schema for video.metadata artifact payload."""
    
    # GPS coordinates (optional)
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    
    # Image properties (optional)
    image_size: str | None = None
    megapixels: float | None = None
    rotation: int | None = None
    
    # Audio/Video properties (optional)
    avg_bitrate: str | None = None
    duration_seconds: float | None = None
    frame_rate: float | None = None
    codec: str | None = None
    
    # File properties (optional)
    file_size: int | None = None
    file_type: str | None = None
    mime_type: str | None = None
    
    # Camera properties (optional)
    camera_make: str | None = None
    camera_model: str | None = None
    
    # Temporal properties (optional)
    create_date: str | None = None
```

#### 3. VideoDiscoveryService Enhancement

Adds metadata_extraction to ACTIVE_TASK_TYPES and creates tasks on discovery.

```python
# In video_discovery_service.py
ACTIVE_TASK_TYPES = [
    "place_detection",
    "metadata_extraction",  # NEW
]

async def discover_and_queue_tasks(self, video_path: str) -> str:
    """
    Enhanced to include metadata_extraction task creation.
    
    For metadata_extraction:
    - language=NULL (language-agnostic)
    - Uses default config from content_creator.json
    - Enqueued like other tasks
    """
```

#### 4. ProjectionSyncService Enhancement

Syncs GPS data from video.metadata artifacts to video_locations table.

```python
class ProjectionSyncService:
    """Synchronize artifacts to projection tables."""
    
    def sync_artifact(self, artifact: ArtifactEnvelope) -> None:
        """
        Sync artifact to appropriate projection table.
        
        For video.metadata artifacts:
        - Extract GPS coordinates from payload
        - Create video_locations entry if GPS exists
        """
        
    def _sync_video_metadata(self, artifact: ArtifactEnvelope) -> None:
        """
        Synchronize video.metadata artifact to video_locations projection.
        
        Args:
            artifact: The video.metadata artifact to synchronize
        """
```

#### 5. Video Locations Projection Table

New PostgreSQL table for GPS normalization.

```sql
CREATE TABLE video_locations (
    id SERIAL PRIMARY KEY,
    artifact_id UUID NOT NULL UNIQUE,
    asset_id UUID NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    altitude FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id),
    FOREIGN KEY (asset_id) REFERENCES videos(video_id)
);

CREATE INDEX idx_video_locations_asset_id ON video_locations(asset_id);
CREATE INDEX idx_video_locations_latitude ON video_locations(latitude);
CREATE INDEX idx_video_locations_longitude ON video_locations(longitude);
```

### Frontend Components

#### 1. MetadataViewer Component

Displays extracted metadata in organized sections.

```typescript
interface MetadataViewerProps {
  artifact: ArtifactEnvelope;
  videoId: string;
}

export function MetadataViewer({ artifact, videoId }: MetadataViewerProps) {
  // Renders metadata in sections:
  // - GPS coordinates (if available)
  // - Camera info
  // - File info
  // - Temporal info
  // - Image info
}
```

#### 2. VideoPlayer Enhancement

Adds "Metadata" tab to VideoPlayer component.

```typescript
export function VideoPlayer({ videoId }: VideoPlayerProps) {
  const [activeTab, setActiveTab] = useState<'transcript' | 'metadata'>('transcript');
  
  return (
    <div>
      <div className="tabs">
        <button onClick={() => setActiveTab('transcript')}>Transcript</button>
        <button onClick={() => setActiveTab('metadata')}>Metadata</button>
      </div>
      
      {activeTab === 'metadata' && <MetadataViewer videoId={videoId} />}
    </div>
  );
}
```

## Data Models

### Artifact Envelope for Metadata

```python
artifact = ArtifactEnvelope(
    artifact_id="uuid",
    asset_id="video_id",
    artifact_type="video.metadata",
    schema_version=1,
    span_start_ms=0,
    span_end_ms=video_duration_ms,
    payload_json=json.dumps({
        "latitude": 40.7128,
        "longitude": -74.0060,
        # ... other fields
    }),
    producer="pyexiftool",
    producer_version="0.4.12",
    model_profile="balanced",
    run_id="run_uuid",
    created_at=datetime.now()
)
```

### Task Model for Metadata Extraction

```python
task = Task(
    task_id="uuid",
    video_id="video_id",
    task_type="metadata_extraction",
    language=None,  # Language-agnostic
    status="pending",
    priority=1
)
```

### Video Locations Projection Entry

```python
video_location = {
    "artifact_id": "artifact_uuid",
    "asset_id": "video_id",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "altitude": 10.5,
    "created_at": datetime.now()
}
```

## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Metadata Task Creation on Discovery

*For any* video discovered through the discovery service, a metadata_extraction task with language=NULL should be created and enqueued to the job queue.

**Validates: Requirements 4.1, 4.2, 4.4**

### Property 2: Metadata Artifact Envelope Structure

*For any* metadata extraction result, the resulting artifact envelope should have artifact_type="video.metadata", span_start_ms=0, span_end_ms equal to video duration, producer="pyexiftool", and model_profile="balanced".

**Validates: Requirements 2.1, 2.2, 2.4, 2.5**

### Property 3: No Null Values in Metadata Payload

*For any* extracted metadata payload, all fields present in the payload should have non-null values (missing fields are omitted, not set to null).

**Validates: Requirements 1.4, 2.3**

### Property 4: GPS Projection Table Creation

*For any* metadata artifact containing GPS coordinates (latitude and longitude), a corresponding entry should exist in the video_locations projection table with matching artifact_id, asset_id, latitude, and longitude.

**Validates: Requirements 3.1, 3.2, 3.5**

### Property 5: No GPS Projection Without Coordinates

*For any* metadata artifact lacking GPS coordinates, no entry should exist in the video_locations projection table for that artifact_id.

**Validates: Requirements 3.4**

### Property 6: Metadata Extraction Handles Missing Fields

*For any* video file, the metadata extraction should complete successfully (either with extracted fields or empty payload) without raising exceptions, even if the file lacks certain metadata fields.

**Validates: Requirements 1.4, 1.5, 6.1, 6.2**

### Property 7: Metadata Viewer Displays Available Fields

*For any* metadata artifact, the MetadataViewer component should display all non-null fields from the payload organized into their respective sections (GPS, Camera, File, Temporal, Image).

**Validates: Requirements 5.2, 5.3, 5.5**

### Property 8: GPS Coordinate Formatting

*For any* metadata artifact with GPS coordinates, the MetadataViewer should display latitude and longitude in user-friendly format (e.g., "40.7128°N, 74.0060°W") rather than raw decimal values.

**Validates: Requirements 5.4**

### Property 9: Metadata Artifact Persistence

*For any* metadata artifact created by the extraction process, querying the artifacts table by artifact_id should return the artifact with schema_version=1 and all extracted fields in payload_json.

**Validates: Requirements 2.6**

### Property 10: Metadata Task Configuration Loading

*For any* metadata_extraction task created, the task configuration should be loaded from content_creator.json (or use defaults if not configured), and the task should be enqueued with this configuration.

**Validates: Requirements 4.3**

## Error Handling

### Extraction Failures

- **Unreadable Video File**: Log error, create empty metadata artifact with empty payload
- **pyexiftool Failure**: Log error, mark task as completed with empty payload
- **Missing Composite Fields**: Omit field from payload (no null values)

### Projection Sync Failures

- **Database Error**: Log error, continue processing other artifacts
- **Constraint Violation**: Log error, skip video_locations entry creation

### Task Failures

- **Task Execution Error**: Mark task status as "failed", log error details
- **Retry Mechanism**: Failed tasks can be retried via task retry endpoint

## Testing Strategy

### Unit Tests

- Test MetadataExtractor with various video formats
- Test MetadataV1 schema validation
- Test ProjectionSyncService GPS extraction logic
- Test MetadataViewer component rendering with various payloads
- Test error handling for missing/corrupted files

### Property-Based Tests

- Property 1: Verify metadata task creation on discovery
- Property 2: Verify artifact envelope structure
- Property 3: Verify no null values in payload
- Property 4: Verify GPS projection table creation
- Property 5: Verify no projection without GPS
- Property 6: Verify extraction handles missing fields
- Property 7: Verify MetadataViewer displays available fields
- Property 8: Verify GPS coordinate formatting
- Property 9: Verify artifact persistence
- Property 10: Verify task configuration loading

### Integration Tests

- End-to-end: Discover video → Extract metadata → Display in UI
- Test with videos containing GPS data
- Test with videos lacking GPS data
- Test with various video formats (MP4, MOV, MKV, AVI)
- Test error scenarios (unreadable files, corrupted metadata)

