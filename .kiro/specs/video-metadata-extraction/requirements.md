# Requirements Document: Video Metadata Extraction

## Introduction

The Video Metadata Extraction feature adds automated metadata extraction from video files to the content discovery pipeline. When videos are discovered, a metadata extraction task automatically runs to extract standardized metadata fields using pyexiftool. The extracted metadata is persisted as artifact envelopes with type `video.metadata`, enabling metadata display in the video player UI and supporting future geo-spatial queries through a GPS projection table.

## Glossary

- **Metadata**: Standardized information extracted from video files (GPS, camera info, file info, temporal info, image info)
- **Artifact**: A structured result from an ML task, stored as an envelope with type, payload, and metadata
- **Artifact Envelope**: Container for an artifact with artifact_type, span_start_ms, span_end_ms, payload_json, producer, producer_version, model_profile
- **Projection Table**: Denormalized database table for efficient querying of specific artifact data (e.g., GPS coordinates)
- **GPS Coordinates**: Latitude, longitude, and altitude extracted from video metadata
- **Composite Fields**: Format-agnostic metadata fields from pyexiftool that work across video formats
- **QuickTime Fallback**: Alternative metadata field names for QuickTime-based video formats
- **Metadata Task**: Language-agnostic ML task type that extracts metadata from video files
- **Producer**: The tool/service that created an artifact (e.g., "pyexiftool")
- **Model Profile**: Configuration profile for the extraction (e.g., "balanced")

## Requirements

### Requirement 1: Metadata Extraction from Video Files

**User Story:** As a content creator, I want video metadata to be automatically extracted from my video files, so that I can view and search by camera information, GPS location, and file properties.

#### Acceptance Criteria

1. WHEN a video is discovered, THE Metadata_Extractor SHALL automatically create a metadata extraction task
2. WHEN a metadata extraction task runs, THE Metadata_Extractor SHALL extract Composite fields from the video file using pyexiftool
3. WHEN extracting metadata, THE Metadata_Extractor SHALL extract the following standardized Composite fields from pyexiftool:
   - GPS: Composite:GPSLatitude, Composite:GPSLongitude, Composite:GPSAltitude
   - Image: Composite:ImageSize, Composite:Megapixels, Composite:Rotation
   - Audio/Video: Composite:AvgBitrate, Composite:Duration, Composite:VideoFrameRate, Composite:VideoCodec
   - File: Composite:FileSize, Composite:FileType, Composite:MIMEType
   - Camera: Composite:Make, Composite:Model
   - Temporal: Composite:CreateDate
   - With QuickTime fallbacks: QuickTime:GPSCoordinates, QuickTime:Duration, QuickTime:CreateDate
4. WHEN a video file lacks a metadata field, THE Metadata_Extractor SHALL omit that field from the payload (no null values)
5. WHEN extracting metadata, THE Metadata_Extractor SHALL use format-agnostic Composite fields with QuickTime fallbacks for compatibility

### Requirement 2: Artifact Storage with Standardized Envelope

**User Story:** As a system architect, I want metadata to be stored as standardized artifact envelopes, so that metadata integrates seamlessly with the existing artifact system.

#### Acceptance Criteria

1. WHEN metadata is extracted, THE Artifact_Storage SHALL create an artifact envelope with artifact_type "video.metadata"
2. WHEN creating a metadata artifact, THE Artifact_Storage SHALL set span_start_ms to 0 and span_end_ms to video duration in milliseconds
3. WHEN creating a metadata artifact, THE Artifact_Storage SHALL include all extracted metadata fields in payload_json
4. WHEN creating a metadata artifact, THE Artifact_Storage SHALL set producer to "pyexiftool" and producer_version to the current pyexiftool version
5. WHEN creating a metadata artifact, THE Artifact_Storage SHALL set model_profile to "balanced"
6. WHEN a metadata artifact is created, THE Artifact_Storage SHALL persist it to the artifacts table with schema_version 1

### Requirement 3: GPS Normalization and Projection Table

**User Story:** As a system architect, I want GPS coordinates to be normalized into a projection table, so that future geo-spatial queries can efficiently find videos by location.

#### Acceptance Criteria

1. WHEN a metadata artifact contains GPS coordinates, THE GPS_Normalizer SHALL create an entry in the video_locations projection table
2. WHEN creating a video_locations entry, THE GPS_Normalizer SHALL store artifact_id, asset_id, latitude, longitude, altitude, and created_at
3. WHEN creating a video_locations entry, THE GPS_Normalizer SHALL index latitude and longitude columns for efficient geo-queries
4. WHEN a metadata artifact lacks GPS coordinates, THE GPS_Normalizer SHALL not create a video_locations entry
5. WHEN a video_locations entry is created, THE GPS_Normalizer SHALL set asset_id to the video_id for efficient asset-based queries

### Requirement 4: Task Integration with Video Discovery

**User Story:** As a system operator, I want metadata extraction to run automatically on video discovery, so that metadata is available immediately after discovery.

#### Acceptance Criteria

1. WHEN a video is discovered, THE Video_Discovery_Service SHALL add "metadata_extraction" to the active task types
2. WHEN creating a metadata extraction task, THE Video_Discovery_Service SHALL use language=NULL (language-agnostic)
3. WHEN creating a metadata extraction task, THE Video_Discovery_Service SHALL load default configuration from content_creator.json
4. WHEN a metadata extraction task is created, THE Video_Discovery_Service SHALL enqueue it to the job queue for processing

### Requirement 5: Metadata Display in Video Player UI

**User Story:** As a content creator, I want to view extracted metadata in the video player, so that I can see camera information, GPS location, and file properties while watching.

#### Acceptance Criteria

1. WHEN the video player loads, THE Video_Player SHALL display a "Metadata" tab alongside other tabs
2. WHEN the Metadata tab is selected, THE Metadata_Viewer SHALL display extracted metadata in readable format
3. WHEN displaying metadata, THE Metadata_Viewer SHALL organize information into sections:
   - GPS coordinates (if available): latitude, longitude, altitude
   - Camera info: camera_make, camera_model
   - File info: file_size, file_type, mime_type, codec
   - Temporal info: duration_seconds, frame_rate, create_date
   - Image info: image_size, megapixels, rotation
4. WHEN GPS coordinates are available, THE Metadata_Viewer SHALL display them in a user-friendly format (e.g., "40.7128°N, 74.0060°W")
5. WHEN a metadata field is unavailable, THE Metadata_Viewer SHALL not display that field

### Requirement 6: Metadata Extraction Error Handling

**User Story:** As a system operator, I want metadata extraction to handle errors gracefully, so that missing or corrupted metadata doesn't break the pipeline.

#### Acceptance Criteria

1. IF a video file cannot be read by pyexiftool, THEN THE Metadata_Extractor SHALL log the error and create an empty metadata artifact
2. IF pyexiftool fails to extract metadata, THEN THE Metadata_Extractor SHALL mark the task as completed with an empty payload
3. WHEN a metadata extraction task fails, THEN THE Task_Handler SHALL mark the task status as "failed" and log the error
4. IF a video_locations entry creation fails, THEN THE GPS_Normalizer SHALL log the error and continue processing other artifacts

