# Design Document

## Overview

This design document specifies the implementation of Phase 1 of the LEAP PLAYER artifact envelope architecture. The system transitions from a "one type = one table" pattern to a unified artifact envelope model that provides:

- **Canonical time-aligned artifacts** for deterministic navigation
- **Multiple model strengths** coexisting without overwriting historical data
- **Unified storage** reducing the "new artifact tax" (no new tables per ML model)
- **Derived projections** for query performance
- **Backward compatibility** through dual-write mode during migration

The architecture separates concerns:
- **Artifacts table**: Canonical source of truth for all ML outputs
- **Projection tables**: Denormalized views optimized for specific query patterns
- **Selection policies**: User/system preferences for which artifact version to use
- **Schema registry**: Centralized validation and versioning

This phase migrates all video processing tasks (transcripts, scenes, objects, faces, places, OCR) to the unified model while maintaining backward compatibility with existing APIs.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Video Processing Pipeline                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Artifact Producers                        │
│  (Whisper, PySceneDetect, YOLO, ResNet, EasyOCR)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Schema Registry                            │
│         (Validates payloads against schemas)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Artifact Repository                         │
│              (Handles CRUD + Selection)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────┐          ┌──────────────┐
│   Artifacts  │          │  Projections │
│    Table     │──────────▶│   (FTS5,    │
│  (Canonical) │          │   Labels,    │
│              │          │   Clusters)  │
└──────────────┘          └──────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer                               │
│         /jump, /find, /artifacts, /search                    │
└─────────────────────────────────────────────────────────────┘
```


### Data Flow

1. **Ingestion**: Video processing pipeline runs ML models
2. **Production**: Each model produces typed artifacts with envelope metadata
3. **Validation**: Schema registry validates payloads before storage
4. **Storage**: Artifacts stored in unified table with run tracking
5. **Projection**: Triggers update derived projection tables (FTS, labels, clusters)
6. **Selection**: Queries apply selection policies to choose artifact versions
7. **Retrieval**: API layer serves artifacts through repository abstraction

### Cutover Strategy

**Immediate Cutover**: The system will immediately switch to the artifact model:
- All new processing writes only to artifacts table
- Legacy tables remain for historical data reference only
- No migration of legacy data (clean slate approach)

**Deployment Path**:
1. Deploy artifact infrastructure (tables, schemas, repositories, projections)
2. Deploy new code that uses artifact-based storage and retrieval
3. Process new videos using artifact model
4. Legacy tables remain read-only for historical reference
5. Optional: Gradually reprocess old videos to populate artifacts

## Components and Interfaces

### 1. Artifact Envelope (Core Data Model)

```python
@dataclass
class ArtifactEnvelope:
    artifact_id: str
    asset_id: str  # video_id
    artifact_type: str  # e.g., "transcript.segment", "scene", "object.detection"
    schema_version: int
    span_start_ms: int
    span_end_ms: int
    payload_json: str  # JSON-serialized payload
    producer: str  # e.g., "whisper", "pyscenedetect"
    producer_version: str  # e.g., "3.0.0"
    model_profile: str  # "fast" | "balanced" | "high_quality"
    config_hash: str  # Hash of configuration used
    input_hash: str  # Hash of input data
    run_id: str
    created_at: datetime
```


### 2. Schema Registry

```python
class SchemaRegistry:
    """Central registry for artifact schemas."""
    
    _schemas: dict[tuple[str, int], Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, artifact_type: str, schema_version: int, schema: Type[BaseModel]):
        """Register a schema for an artifact type and version."""
        key = (artifact_type, schema_version)
        cls._schemas[key] = schema
    
    @classmethod
    def get_schema(cls, artifact_type: str, schema_version: int) -> Type[BaseModel]:
        """Get schema for artifact type and version."""
        key = (artifact_type, schema_version)
        if key not in cls._schemas:
            raise SchemaNotFoundError(f"No schema for {artifact_type} v{schema_version}")
        return cls._schemas[key]
    
    @classmethod
    def validate(cls, artifact_type: str, schema_version: int, payload: dict) -> BaseModel:
        """Validate and parse payload."""
        schema = cls.get_schema(artifact_type, schema_version)
        return schema(**payload)
    
    @classmethod
    def serialize(cls, artifact_type: str, schema_version: int, payload: BaseModel) -> str:
        """Serialize payload to JSON string."""
        return payload.model_dump_json()
```

### 3. Artifact Schemas

#### Transcript Segment Schema (v1)

```python
class TranscriptSegmentV1(BaseModel):
    text: str
    speaker: str | None = None
    confidence: float
    language: str = "en"
```

#### Scene Schema (v1)

```python
class SceneV1(BaseModel):
    scene_index: int
    method: str  # "content", "threshold", etc.
    score: float
    frame_number: int
```

#### Object Detection Schema (v1)

```python
class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class ObjectDetectionV1(BaseModel):
    label: str
    confidence: float
    bounding_box: BoundingBox
    frame_number: int
```

#### Face Detection Schema (v1)

```python
class FaceDetectionV1(BaseModel):
    confidence: float
    bounding_box: BoundingBox
    cluster_id: str | None = None
    frame_number: int
```

#### Place Classification Schema (v1)

```python
class AlternativeLabel(BaseModel):
    label: str
    confidence: float

class PlaceClassificationV1(BaseModel):
    label: str
    confidence: float
    alternative_labels: list[AlternativeLabel]
    frame_number: int
```

#### OCR Text Schema (v1)

```python
class PolygonPoint(BaseModel):
    x: float
    y: float

class OcrTextV1(BaseModel):
    text: str
    confidence: float
    bounding_box: list[PolygonPoint]  # Polygon points
    language: str
    frame_number: int
```


### 2. Schema Registry

The schema registry maps `(artifact_type, schema_version)` to Pydantic models for validation.

```python
class SchemaRegistry:
    """Central registry for artifact schemas."""
    
    _schemas: dict[tuple[str, int], Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, artifact_type: str, schema_version: int, model: Type[BaseModel]):
        """Register a schema for an artifact type."""
        key = (artifact_type, schema_version)
        if key in cls._schemas:
            raise ValueError(f"Schema already registered: {key}")
        cls._schemas[key] = model
    
    @classmethod
    def get_schema(cls, artifact_type: str, schema_version: int) -> Type[BaseModel]:
        """Get schema for artifact type and version."""
        key = (artifact_type, schema_version)
        if key not in cls._schemas:
            raise SchemaNotFoundError(f"No schema registered for {key}")
        return cls._schemas[key]
    
    @classmethod
    def validate(cls, artifact_type: str, schema_version: int, payload: dict) -> BaseModel:
        """Validate and parse payload."""
        schema = cls.get_schema(artifact_type, schema_version)
        return schema(**payload)
```

### 3. Artifact Type Schemas

Each artifact type has a versioned Pydantic schema:

```python
# domain/artifacts/schemas/transcript_segment_v1.py
class TranscriptSegmentV1(BaseModel):
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None
    language: str = "en"

# domain/artifacts/schemas/scene_v1.py
class SceneV1(BaseModel):
    scene_index: int
    method: str  # "content", "threshold", etc.
    score: float
    frame_number: int

# domain/artifacts/schemas/object_detection_v1.py
class ObjectDetectionV1(BaseModel):
    label: str
    confidence: float
    bounding_box: dict[str, float]  # {x, y, width, height}
    frame_number: int

# domain/artifacts/schemas/face_detection_v1.py
class FaceDetectionV1(BaseModel):
    confidence: float
    bounding_box: dict[str, float]
    cluster_id: Optional[str] = None
    frame_number: int

# domain/artifacts/schemas/place_classification_v1.py
class PlaceClassificationV1(BaseModel):
    label: str
    confidence: float
    alternative_labels: list[dict[str, Any]]  # [{label, confidence}]
    frame_number: int

# domain/artifacts/schemas/ocr_text_v1.py
class OcrTextV1(BaseModel):
    text: str
    confidence: float
    bounding_box: list[list[float]]  # Polygon points
    language: Optional[str] = None
    frame_number: int
```

Schema registration happens at application startup:

```python
# Register all schemas
SchemaRegistry.register("transcript.segment", 1, TranscriptSegmentV1)
SchemaRegistry.register("scene", 1, SceneV1)
SchemaRegistry.register("object.detection", 1, ObjectDetectionV1)
SchemaRegistry.register("face.detection", 1, FaceDetectionV1)
SchemaRegistry.register("place.classification", 1, PlaceClassificationV1)
SchemaRegistry.register("ocr.text", 1, OcrTextV1)
```


### 4. Artifact Repository

The repository provides a clean interface for artifact CRUD operations with selection policy support.

```python
class ArtifactRepository:
    """Repository for artifact storage and retrieval."""
    
    def __init__(self, session: Session, schema_registry: SchemaRegistry):
        self.session = session
        self.schema_registry = schema_registry
    
    def create(self, artifact: ArtifactEnvelope) -> ArtifactEnvelope:
        """Create a new artifact with validation."""
        # Validate payload against schema
        payload_dict = json.loads(artifact.payload_json)
        self.schema_registry.validate(
            artifact.artifact_type, 
            artifact.schema_version, 
            payload_dict
        )
        
        # Store in database
        entity = self._to_entity(artifact)
        self.session.add(entity)
        self.session.commit()
        self.session.refresh(entity)
        return self._to_domain(entity)
    
    def get_by_id(self, artifact_id: str) -> Optional[ArtifactEnvelope]:
        """Get artifact by ID."""
        entity = self.session.query(ArtifactEntity).filter(
            ArtifactEntity.artifact_id == artifact_id
        ).first()
        return self._to_domain(entity) if entity else None
    
    def get_by_asset(
        self,
        asset_id: str,
        artifact_type: Optional[str] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        selection: Optional[SelectionPolicy] = None
    ) -> list[ArtifactEnvelope]:
        """Get artifacts for an asset with optional filtering."""
        query = self.session.query(ArtifactEntity).filter(
            ArtifactEntity.asset_id == asset_id
        )
        
        if artifact_type:
            query = query.filter(ArtifactEntity.artifact_type == artifact_type)
        
        if start_ms is not None:
            query = query.filter(ArtifactEntity.span_start_ms >= start_ms)
        
        if end_ms is not None:
            query = query.filter(ArtifactEntity.span_end_ms <= end_ms)
        
        # Apply selection policy
        if selection:
            query = self._apply_selection_policy(query, asset_id, artifact_type, selection)
        
        query = query.order_by(ArtifactEntity.span_start_ms)
        entities = query.all()
        return [self._to_domain(e) for e in entities]
    
    def get_by_span(
        self,
        asset_id: str,
        artifact_type: str,
        span_start_ms: int,
        span_end_ms: int,
        selection: Optional[SelectionPolicy] = None
    ) -> list[ArtifactEnvelope]:
        """Get artifacts overlapping a time span."""
        query = self.session.query(ArtifactEntity).filter(
            ArtifactEntity.asset_id == asset_id,
            ArtifactEntity.artifact_type == artifact_type,
            ArtifactEntity.span_start_ms < span_end_ms,
            ArtifactEntity.span_end_ms > span_start_ms
        )
        
        if selection:
            query = self._apply_selection_policy(query, asset_id, artifact_type, selection)
        
        entities = query.all()
        return [self._to_domain(e) for e in entities]
    
    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact."""
        deleted = self.session.query(ArtifactEntity).filter(
            ArtifactEntity.artifact_id == artifact_id
        ).delete()
        self.session.commit()
        return deleted > 0
    
    def _apply_selection_policy(self, query, asset_id, artifact_type, policy):
        """Apply selection policy to query."""
        if policy.mode == "pinned" and policy.pinned_run_id:
            query = query.filter(ArtifactEntity.run_id == policy.pinned_run_id)
        elif policy.mode == "profile" and policy.preferred_profile:
            query = query.filter(ArtifactEntity.model_profile == policy.preferred_profile)
        elif policy.mode == "latest":
            # Get most recent run_id for this asset/type
            subquery = (
                self.session.query(ArtifactEntity.run_id)
                .filter(
                    ArtifactEntity.asset_id == asset_id,
                    ArtifactEntity.artifact_type == artifact_type
                )
                .order_by(ArtifactEntity.created_at.desc())
                .limit(1)
                .scalar_subquery()
            )
            query = query.filter(ArtifactEntity.run_id == subquery)
        
        return query
```


### 5. Projection Tables

Projection tables denormalize artifact data for query performance.

#### Transcript FTS Projection (PostgreSQL)

```sql
-- Table for transcript full-text search
CREATE TABLE transcript_fts (
    artifact_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
);

-- GIN index for fast full-text search
CREATE INDEX idx_transcript_fts_tsv ON transcript_fts USING GIN (text_tsv);

-- Index for asset filtering
CREATE INDEX idx_transcript_fts_asset ON transcript_fts(asset_id, start_ms);
```

**Query Pattern**:
```sql
-- Search by text (uses GIN index on tsvector)
SELECT 
    artifact_id, 
    asset_id, 
    start_ms, 
    end_ms,
    ts_headline('english', text, to_tsquery('english', 'password & reset')) as snippet
FROM transcript_fts
WHERE text_tsv @@ to_tsquery('english', 'password & reset')
  AND asset_id = 'video_123'
ORDER BY start_ms;
```

**Benefits**:
- `tsvector` is PostgreSQL's optimized full-text search type
- GIN index provides fast search across millions of rows
- `ts_headline` generates highlighted snippets
- Supports stemming, stop words, and ranking

Synchronization trigger:
```python
def sync_transcript_fts(artifact: ArtifactEnvelope):
    """Sync transcript artifact to FTS table."""
    if artifact.artifact_type != "transcript.segment":
        return
    
    payload = json.loads(artifact.payload_json)
    session.execute(
        text("""
            INSERT INTO transcript_fts (artifact_id, asset_id, start_ms, end_ms, text)
            VALUES (:artifact_id, :asset_id, :start_ms, :end_ms, :text)
        """),
        {
            "artifact_id": artifact.artifact_id,
            "asset_id": artifact.asset_id,
            "start_ms": artifact.span_start_ms,
            "end_ms": artifact.span_end_ms,
            "text": payload["text"]
        }
    )
```

#### Object Labels Projection

```sql
CREATE TABLE object_labels (
    artifact_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
);

CREATE INDEX idx_object_labels_asset_label_start 
ON object_labels(asset_id, label, start_ms);
```

#### Face Clusters Projection

```sql
CREATE TABLE face_clusters (
    artifact_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    cluster_id TEXT,
    confidence REAL NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
);

CREATE INDEX idx_face_clusters_asset_cluster_start 
ON face_clusters(asset_id, cluster_id, start_ms);
```

#### Scene Ranges Projection

```sql
CREATE TABLE scene_ranges (
    artifact_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    scene_index INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
);

CREATE INDEX idx_scene_ranges_asset_index 
ON scene_ranges(asset_id, scene_index);
```

#### OCR Text FTS Projection (PostgreSQL)

```sql
-- Table for OCR full-text search
CREATE TABLE ocr_fts (
    artifact_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
);

-- GIN index for fast full-text search
CREATE INDEX idx_ocr_fts_tsv ON ocr_fts USING GIN (text_tsv);

-- Index for asset filtering
CREATE INDEX idx_ocr_fts_asset ON ocr_fts(asset_id, start_ms);
```

**Scalability Note**:
- OCR can generate many artifacts (one per text detection)
- PostgreSQL GIN indexes handle millions of rows efficiently
- Consider periodic cleanup of low-confidence OCR results
- Archive old OCR data after N days
- Partition ocr_fts by asset_id if it grows very large


### 6. Selection Policy Manager

Manages artifact selection policies for assets.

```python
@dataclass
class SelectionPolicy:
    asset_id: str
    artifact_type: str
    mode: str  # "default" | "pinned" | "latest" | "profile" | "best_quality"
    preferred_profile: Optional[str] = None
    pinned_run_id: Optional[str] = None
    pinned_artifact_id: Optional[str] = None
    updated_at: Optional[datetime] = None

class SelectionPolicyManager:
    """Manages selection policies for artifacts."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_policy(self, asset_id: str, artifact_type: str) -> Optional[SelectionPolicy]:
        """Get selection policy for asset and artifact type."""
        entity = self.session.query(SelectionPolicyEntity).filter(
            SelectionPolicyEntity.asset_id == asset_id,
            SelectionPolicyEntity.artifact_type == artifact_type
        ).first()
        return self._to_domain(entity) if entity else None
    
    def set_policy(self, policy: SelectionPolicy) -> SelectionPolicy:
        """Set or update selection policy."""
        entity = self._to_entity(policy)
        existing = self.session.query(SelectionPolicyEntity).filter(
            SelectionPolicyEntity.asset_id == policy.asset_id,
            SelectionPolicyEntity.artifact_type == policy.artifact_type
        ).first()
        
        if existing:
            for key, value in entity.__dict__.items():
                if not key.startswith("_"):
                    setattr(existing, key, value)
            existing.updated_at = datetime.utcnow()
            self.session.commit()
            return self._to_domain(existing)
        else:
            self.session.add(entity)
            self.session.commit()
            return self._to_domain(entity)
    
    def get_default_policy(self) -> SelectionPolicy:
        """Get default selection policy (latest)."""
        return SelectionPolicy(
            asset_id="",
            artifact_type="",
            mode="latest"
        )
```

### 7. Jump Navigation Service

Provides deterministic next/prev navigation across artifact types.

```python
class JumpNavigationService:
    """Service for jump navigation across artifacts."""
    
    def __init__(self, artifact_repo: ArtifactRepository, policy_manager: SelectionPolicyManager):
        self.artifact_repo = artifact_repo
        self.policy_manager = policy_manager
    
    def jump_next(
        self,
        asset_id: str,
        artifact_type: str,
        from_ms: int,
        label: Optional[str] = None,
        cluster_id: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> Optional[dict]:
        """Jump to next artifact occurrence."""
        policy = self.policy_manager.get_policy(asset_id, artifact_type) or \
                 self.policy_manager.get_default_policy()
        
        artifacts = self.artifact_repo.get_by_asset(
            asset_id=asset_id,
            artifact_type=artifact_type,
            start_ms=from_ms,
            selection=policy
        )
        
        # Filter by label/cluster if specified
        filtered = self._filter_artifacts(artifacts, label, cluster_id, min_confidence)
        
        if not filtered:
            return None
        
        # Return first match
        artifact = filtered[0]
        return {
            "jump_to": {
                "start_ms": artifact.span_start_ms,
                "end_ms": artifact.span_end_ms
            },
            "artifact_ids": [artifact.artifact_id]
        }
    
    def jump_prev(
        self,
        asset_id: str,
        artifact_type: str,
        from_ms: int,
        label: Optional[str] = None,
        cluster_id: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> Optional[dict]:
        """Jump to previous artifact occurrence."""
        policy = self.policy_manager.get_policy(asset_id, artifact_type) or \
                 self.policy_manager.get_default_policy()
        
        artifacts = self.artifact_repo.get_by_asset(
            asset_id=asset_id,
            artifact_type=artifact_type,
            end_ms=from_ms,
            selection=policy
        )
        
        # Filter and reverse order
        filtered = self._filter_artifacts(artifacts, label, cluster_id, min_confidence)
        filtered.reverse()
        
        if not filtered:
            return None
        
        # Return first match (last in original order)
        artifact = filtered[0]
        return {
            "jump_to": {
                "start_ms": artifact.span_start_ms,
                "end_ms": artifact.span_end_ms
            },
            "artifact_ids": [artifact.artifact_id]
        }
    
    def _filter_artifacts(self, artifacts, label, cluster_id, min_confidence):
        """Filter artifacts by label, cluster, and confidence."""
        filtered = []
        for artifact in artifacts:
            payload = json.loads(artifact.payload_json)
            
            # Check confidence
            if "confidence" in payload and payload["confidence"] < min_confidence:
                continue
            
            # Check label
            if label and payload.get("label") != label:
                continue
            
            # Check cluster
            if cluster_id and payload.get("cluster_id") != cluster_id:
                continue
            
            filtered.append(artifact)
        
        return filtered
```


### 8. Find Within Video Service

Provides keyword search within a video using FTS projections.

```python
class FindWithinVideoService:
    """Service for keyword search within videos."""
    
    def __init__(self, session: Session, policy_manager: SelectionPolicyManager):
        self.session = session
        self.policy_manager = policy_manager
    
    def find_next(
        self,
        asset_id: str,
        query: str,
        from_ms: int,
        source: str = "all"  # "transcript" | "ocr" | "all"
    ) -> list[dict]:
        """Find next occurrence of query text."""
        results = []
        
        if source in ["transcript", "all"]:
            results.extend(self._search_transcript_fts(asset_id, query, from_ms, "next"))
        
        if source in ["ocr", "all"]:
            results.extend(self._search_ocr_fts(asset_id, query, from_ms, "next"))
        
        # Sort by timestamp
        results.sort(key=lambda x: x["jump_to"]["start_ms"])
        return results
    
    def find_prev(
        self,
        asset_id: str,
        query: str,
        from_ms: int,
        source: str = "all"
    ) -> list[dict]:
        """Find previous occurrence of query text."""
        results = []
        
        if source in ["transcript", "all"]:
            results.extend(self._search_transcript_fts(asset_id, query, from_ms, "prev"))
        
        if source in ["ocr", "all"]:
            results.extend(self._search_ocr_fts(asset_id, query, from_ms, "prev"))
        
        # Sort by timestamp descending
        results.sort(key=lambda x: x["jump_to"]["start_ms"], reverse=True)
        return results
    
    def _search_transcript_fts(self, asset_id, query, from_ms, direction):
        """Search transcript FTS table using PostgreSQL full-text search."""
        operator = ">" if direction == "next" else "<"
        order = "ASC" if direction == "next" else "DESC"
        
        sql = text(f"""
            SELECT 
                artifact_id,
                start_ms,
                end_ms,
                ts_headline('english', text, to_tsquery('english', :query)) as snippet
            FROM transcript_fts
            WHERE text_tsv @@ to_tsquery('english', :query)
              AND asset_id = :asset_id
              AND start_ms {operator} :from_ms
            ORDER BY start_ms {order}
            LIMIT 10
        """)
        
        rows = self.session.execute(sql, {
            "query": query.replace(" ", " & "),  # Convert to tsquery format
            "asset_id": asset_id,
            "from_ms": from_ms
        }).fetchall()
        
        return [
            {
                "jump_to": {"start_ms": row.start_ms, "end_ms": row.end_ms},
                "artifact_id": row.artifact_id,
                "snippet": row.snippet,
                "source": "transcript"
            }
            for row in rows
        ]
    
    def _search_ocr_fts(self, asset_id, query, from_ms, direction):
        """Search OCR FTS table using PostgreSQL full-text search."""
        operator = ">" if direction == "next" else "<"
        order = "ASC" if direction == "next" else "DESC"
        
        sql = text(f"""
            SELECT 
                artifact_id,
                start_ms,
                end_ms,
                ts_headline('english', text, to_tsquery('english', :query)) as snippet
            FROM ocr_fts
            WHERE text_tsv @@ to_tsquery('english', :query)
              AND asset_id = :asset_id
              AND start_ms {operator} :from_ms
            ORDER BY start_ms {order}
            LIMIT 10
        """)
        
        rows = self.session.execute(sql, {
            "query": query.replace(" ", " & "),  # Convert to tsquery format
            "asset_id": asset_id,
            "from_ms": from_ms
        }).fetchall()
        
        return [
            {
                "jump_to": {"start_ms": row.start_ms, "end_ms": row.end_ms},
                "artifact_id": row.artifact_id,
                "snippet": row.snippet,
                "source": "ocr"
            }
            for row in rows
        ]
```



## Data Models

### Database Schema (PostgreSQL)

```sql
-- Artifacts table (canonical storage)
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    span_start_ms INTEGER NOT NULL,
    span_end_ms INTEGER NOT NULL,
    payload_json JSONB NOT NULL,  -- JSONB for efficient querying
    producer TEXT NOT NULL,
    producer_version TEXT NOT NULL,
    model_profile TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    run_id TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (asset_id) REFERENCES videos(video_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Composite indexes for common query patterns
CREATE INDEX idx_artifacts_asset_type_start 
ON artifacts(asset_id, artifact_type, span_start_ms);

CREATE INDEX idx_artifacts_asset_type_profile_start 
ON artifacts(asset_id, artifact_type, model_profile, span_start_ms);

CREATE INDEX idx_artifacts_type_created 
ON artifacts(artifact_type, created_at);

CREATE INDEX idx_artifacts_run 
ON artifacts(run_id);

-- GIN index for JSONB payload queries (optional, for advanced filtering)
CREATE INDEX idx_artifacts_payload_gin 
ON artifacts USING GIN (payload_json);

-- Runs table (pipeline execution tracking)
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL,
    pipeline_profile TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    status TEXT NOT NULL,
    error TEXT,
    FOREIGN KEY (asset_id) REFERENCES videos(video_id)
);

CREATE INDEX idx_runs_asset_status 
ON runs(asset_id, status);

-- Selection policies table
CREATE TABLE artifact_selections (
    asset_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    selection_mode TEXT NOT NULL,
    preferred_profile TEXT,
    pinned_run_id TEXT,
    pinned_artifact_id TEXT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (asset_id, artifact_type),
    FOREIGN KEY (asset_id) REFERENCES videos(video_id)
);
```

### Scalability Considerations (PostgreSQL)

**Artifacts Table Growth**:
- Expected: 100-1000 artifacts per video depending on length and processing
- For 10,000 videos: 1-10 million rows
- For 100,000 videos: 10-100 million rows

**PostgreSQL Optimization Strategies**:

1. **JSONB Benefits**:
   - Binary format, faster than JSON text
   - Supports indexing with GIN
   - Can query nested fields efficiently
   - Example: `WHERE payload_json->>'label' = 'dog'`

2. **Projection-First Queries**:
   - Always query projections first (smaller, indexed)
   - Only fetch full artifacts when needed
   - Projections are much smaller (10-100x reduction)

3. **Vacuum and Maintenance**:
   - Auto-vacuum configured for high-write tables
   - Periodic ANALYZE for query planner
   - Monitor table sizes and query performance

**Query Performance**:
- Indexed queries on asset_id: <50ms for single-video queries
- Cross-video queries: <500ms with proper indexing
- JSONB GIN index: Fast filtering on payload fields
- Projection tables: 10-100x faster for common queries


### API Request/Response Models

```python
# Jump Navigation Request
class JumpRequest(BaseModel):
    kind: str  # "scene" | "transcript" | "object" | "face" | "place" | "ocr"
    direction: str  # "next" | "prev"
    from_ms: int
    label: Optional[str] = None
    face_cluster_id: Optional[str] = None
    min_confidence: float = 0.0
    selection: str = "default"
    profile: Optional[str] = None

# Jump Navigation Response
class JumpResponse(BaseModel):
    video_id: str
    jump_to: dict[str, int]  # {start_ms, end_ms}
    artifact_ids: list[str]

# Find Within Video Request
class FindRequest(BaseModel):
    q: str
    direction: str = "next"
    from_ms: int = 0
    source: str = "all"  # "transcript" | "ocr" | "all"
    selection: str = "default"
    profile: Optional[str] = None

# Find Within Video Response
class FindResponse(BaseModel):
    video_id: str
    matches: list[dict]  # [{jump_to, artifact_id, snippet, source}]

# Artifacts Query Request
class ArtifactsQueryRequest(BaseModel):
    type: Optional[str] = None
    from_ms: int = 0
    to_ms: Optional[int] = None
    selection: str = "default"
    profile: Optional[str] = None

# Artifacts Query Response
class ArtifactsQueryResponse(BaseModel):
    video_id: str
    artifacts: list[dict]  # Artifact envelopes with payloads
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Artifact Envelope Completeness

*For any* artifact stored in the system, all envelope fields (artifact_id, asset_id, artifact_type, schema_version, span_start_ms, span_end_ms, payload_json, producer, producer_version, model_profile, config_hash, input_hash, run_id, created_at) must be present and non-null.

**Validates: Requirements 1.1, 12.1, 12.2, 12.3**

### Property 2: Schema Validation Round-Trip

*For any* artifact with a registered schema, serializing the payload to JSON and then deserializing it using the schema should produce an equivalent object.

**Validates: Requirements 1.2, 3.2, 3.3**

### Property 3: Schema Version Coexistence

*For any* artifact_type with multiple schema versions, artifacts with different schema versions should coexist in the database without conflicts, and each should be retrievable and deserializable using its respective schema.

**Validates: Requirements 1.3, 3.5**

### Property 4: Run Lifecycle Tracking

*For any* pipeline run, the run record should transition through valid states (running → completed/failed), and all artifacts created during that run should reference the run_id.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

### Property 5: Schema Registry Lookup

*For any* registered (artifact_type, schema_version) pair, the schema registry should return the correct Pydantic model, and for any unregistered pair, it should raise SchemaNotFoundError.

**Validates: Requirements 3.1, 3.4**

### Property 6: Artifact Type Storage Consistency

*For any* artifact type (transcript.segment, scene, object.detection, face.detection, place.classification, ocr.text), when an artifact is created, it should be stored with all required envelope fields and payload fields matching the registered schema.

**Validates: Requirements 4.2, 13.2, 14.2, 15.2, 16.2, 17.2**

### Property 7: Projection Synchronization

*For any* artifact that has a corresponding projection table (transcript_fts, object_labels, face_clusters, scene_ranges, ocr_fts), creating the artifact should trigger an insert into the projection table with matching artifact_id and time span.

**Validates: Requirements 5.2, 18.2, 19.2, 20.2, 21.2**

### Property 8: FTS Search Correctness

*For any* keyword query against transcript_fts or ocr_fts, all returned results should contain the query term (case-insensitive), and the snippet should highlight the matched term.

**Validates: Requirements 5.3, 5.4, 9.5, 21.3, 21.4**

### Property 9: Selection Policy Application

*For any* artifact query with a selection policy, the returned artifacts should match the policy criteria: if mode is "profile", all artifacts should have the specified model_profile; if mode is "pinned", all artifacts should have the specified run_id; if mode is "latest", all artifacts should be from the most recent run.

**Validates: Requirements 6.3, 6.4, 6.5, 7.4, 11.3**

### Property 10: Jump Navigation Determinism

*For any* asset and artifact_type, calling jump_next from a given timestamp should return the artifact with the smallest span_start_ms greater than the timestamp, and calling jump_prev should return the artifact with the largest span_end_ms less than the timestamp.

**Validates: Requirements 8.2, 8.3, 22.1, 22.2, 22.3, 22.4, 22.5**

### Property 11: Find Direction Ordering

*For any* find query, when direction is "next", results should be ordered by start_ms ascending, and when direction is "prev", results should be ordered by start_ms descending.

**Validates: Requirements 9.2, 9.3, 23.2, 23.3**

### Property 12: Multi-Source Find Merging

*For any* find query with source="all", the results should include matches from both transcript_fts and ocr_fts, each tagged with its source, and the combined results should be ordered by timestamp according to the direction parameter.

**Validates: Requirements 23.1, 23.3, 23.4, 23.5**

### Property 13: Legacy Data Access

*For any* video with legacy data, querying legacy endpoints should return historical data, and the response should clearly indicate it is legacy data (not artifact-based).

**Validates: Requirements 10.1, 10.2, 10.3**

### Property 14: Multi-Profile Preservation

*For any* asset and artifact_type, running multiple pipelines with different model_profiles should result in artifacts from all profiles being stored, and no profile should overwrite another.

**Validates: Requirements 11.1, 11.4**

### Property 15: Time Span Overlap Query

*For any* time span query [start_ms, end_ms], the returned artifacts should be exactly those where the artifact's span overlaps the query span (artifact.span_start_ms < end_ms AND artifact.span_end_ms > start_ms).

**Validates: Requirements 7.3**


## Error Handling

### Error Types

```python
class ArtifactError(Exception):
    """Base exception for artifact-related errors."""
    pass

class SchemaNotFoundError(ArtifactError):
    """Raised when schema is not registered."""
    pass

class SchemaValidationError(ArtifactError):
    """Raised when payload fails schema validation."""
    pass

class ArtifactNotFoundError(ArtifactError):
    """Raised when artifact is not found."""
    pass

class SelectionPolicyError(ArtifactError):
    """Raised when selection policy is invalid."""
    pass

class ProjectionSyncError(ArtifactError):
    """Raised when projection synchronization fails."""
    pass
```

### Error Handling Strategy

1. **Schema Validation Errors**: Return 400 Bad Request with validation details
2. **Not Found Errors**: Return 404 Not Found with helpful message
3. **Selection Policy Errors**: Return 400 Bad Request with policy details
4. **Projection Sync Errors**: Log error, continue (projections are derived, can be rebuilt)
5. **Database Errors**: Return 500 Internal Server Error, log full details

### Error Response Format

```python
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[dict] = None
    timestamp: datetime
```

### Retry Strategy

- **Transient database errors**: Retry up to 3 times with exponential backoff
- **Schema validation errors**: No retry (client error)
- **Projection sync errors**: Retry once, then log and continue
- **Not found errors**: No retry (expected condition)

### Logging

```python
# Log all artifact operations
logger.info(f"Created artifact: {artifact_id}, type: {artifact_type}, run: {run_id}")

# Log dual-write warnings
logger.warning(f"Dual-write enabled for {artifact_type}, writing to legacy table")

# Log legacy read warnings
logger.warning(f"Reading {artifact_type} from legacy table, consider migration")

# Log projection sync failures
logger.error(f"Failed to sync projection for artifact {artifact_id}: {error}")

# Log selection policy changes
logger.info(f"Updated selection policy for {asset_id}/{artifact_type}: {mode}")
```

## Testing Strategy

### Unit Tests

Unit tests verify specific examples, edge cases, and error conditions:

- **Schema validation**: Test valid and invalid payloads for each schema
- **Repository CRUD**: Test create, read, update, delete operations
- **Selection policy logic**: Test each selection mode with specific examples
- **Jump navigation edge cases**: Test boundary conditions (first/last artifact, no matches)
- **Find search edge cases**: Test empty results, special characters, long queries
- **Dual-write adapter**: Test flag combinations and legacy compatibility
- **Error handling**: Test each error type is raised correctly

### Property-Based Tests

Property-based tests verify universal properties across all inputs. Each test should run a minimum of 100 iterations.

**Property 1: Artifact Envelope Completeness**
- Generate random artifacts with all envelope fields
- Verify all fields are present and non-null after storage and retrieval
- Tag: `Feature: artifact-envelope-architecture, Property 1: Artifact Envelope Completeness`

**Property 2: Schema Validation Round-Trip**
- Generate random valid payloads for each registered schema
- Serialize to JSON, deserialize, verify equivalence
- Tag: `Feature: artifact-envelope-architecture, Property 2: Schema Validation Round-Trip`

**Property 3: Schema Version Coexistence**
- Generate artifacts with different schema versions for same type
- Store all, retrieve each, verify no conflicts
- Tag: `Feature: artifact-envelope-architecture, Property 3: Schema Version Coexistence`

**Property 4: Run Lifecycle Tracking**
- Generate random runs with state transitions
- Verify all artifacts reference valid run_ids
- Tag: `Feature: artifact-envelope-architecture, Property 4: Run Lifecycle Tracking`

**Property 5: Schema Registry Lookup**
- Test registered and unregistered schema lookups
- Verify correct model returned or error raised
- Tag: `Feature: artifact-envelope-architecture, Property 5: Schema Registry Lookup`

**Property 6: Artifact Type Storage Consistency**
- Generate random artifacts for each type
- Store them, verify all envelope and payload fields present
- Tag: `Feature: artifact-envelope-architecture, Property 6: Artifact Type Storage Consistency`

**Property 7: Projection Synchronization**
- Generate random artifacts with projections
- Verify projection tables updated correctly
- Tag: `Feature: artifact-envelope-architecture, Property 7: Projection Synchronization`

**Property 8: FTS Search Correctness**
- Generate random text and search queries
- Verify all results contain query term
- Tag: `Feature: artifact-envelope-architecture, Property 8: FTS Search Correctness`

**Property 9: Selection Policy Application**
- Generate artifacts with different profiles/runs
- Apply random selection policies, verify filtering
- Tag: `Feature: artifact-envelope-architecture, Property 9: Selection Policy Application`

**Property 10: Jump Navigation Determinism**
- Generate random artifact sequences
- Test jump_next/prev from random timestamps
- Verify correct artifact returned
- Tag: `Feature: artifact-envelope-architecture, Property 10: Jump Navigation Determinism`

**Property 11: Find Direction Ordering**
- Generate random artifacts with text
- Test find with both directions
- Verify ordering is correct
- Tag: `Feature: artifact-envelope-architecture, Property 11: Find Direction Ordering`

**Property 12: Multi-Source Find Merging**
- Generate artifacts in both transcript and OCR
- Test find with source="all"
- Verify results merged and ordered correctly
- Tag: `Feature: artifact-envelope-architecture, Property 12: Multi-Source Find Merging`

**Property 13: Legacy Data Access**
- Query legacy endpoints for historical data
- Verify responses indicate legacy vs artifact-based data
- Tag: `Feature: artifact-envelope-architecture, Property 13: Legacy Data Access`

**Property 14: Multi-Profile Preservation**
- Generate artifacts with different profiles
- Verify all profiles preserved
- Tag: `Feature: artifact-envelope-architecture, Property 14: Multi-Profile Preservation`

**Property 15: Time Span Overlap Query**
- Generate random artifacts with time spans
- Query with random time ranges
- Verify only overlapping artifacts returned
- Tag: `Feature: artifact-envelope-architecture, Property 15: Time Span Overlap Query`

### Integration Tests

- **End-to-end artifact lifecycle**: Create artifact → query → update selection → query again
- **Dual-write mode**: Write through adapter, verify both storages
- **Migration script**: Run migration, verify data consistency
- **API endpoints**: Test /jump, /find, /artifacts endpoints
- **Projection rebuild**: Delete projection, rebuild from artifacts, verify consistency

### Testing Tools

- **pytest**: Unit and integration testing framework
- **Hypothesis**: Property-based testing library for Python
- **pytest-asyncio**: Async test support
- **SQLAlchemy test fixtures**: In-memory SQLite for fast tests
- **Factory patterns**: Generate test data with realistic variations

### Test Configuration

```python
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    property: Property-based tests
    integration: Integration tests
    slow: Slow-running tests

# Hypothesis settings for property tests
from hypothesis import settings
settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=20, deadline=None)
settings.load_profile("dev")
```

