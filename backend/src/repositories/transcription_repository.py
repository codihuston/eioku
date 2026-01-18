from sqlalchemy.orm import Session

from ..database.models import Transcription as TranscriptionEntity
from ..domain.models import Transcription
from .interfaces import TranscriptionRepository


class SqlTranscriptionRepository(TranscriptionRepository):
    """SQLAlchemy implementation of TranscriptionRepository."""

    def __init__(self, session: Session):
        self.session = session

    def save(self, transcription: Transcription) -> Transcription:
        """Save transcription to database."""
        entity = self._to_entity(transcription)

        # Check if exists (update) or new (create)
        existing = (
            self.session.query(TranscriptionEntity)
            .filter(TranscriptionEntity.segment_id == transcription.segment_id)
            .first()
        )

        if existing:
            # Update existing
            for key, value in entity.__dict__.items():
                if not key.startswith("_") and value is not None:
                    setattr(existing, key, value)
            self.session.commit()
            self.session.refresh(existing)
            return self._to_domain(existing)
        else:
            # Create new
            self.session.add(entity)
            self.session.commit()
            self.session.refresh(entity)
            return self._to_domain(entity)

    def find_by_video_id(self, video_id: str) -> list[Transcription]:
        """Find all transcriptions for a video."""
        entities = (
            self.session.query(TranscriptionEntity)
            .filter(TranscriptionEntity.video_id == video_id)
            .order_by(TranscriptionEntity.start)
            .all()
        )
        return [self._to_domain(entity) for entity in entities]

    def find_by_time_range(
        self, video_id: str, start: float, end: float
    ) -> list[Transcription]:
        """Find transcriptions within time range."""
        entities = (
            self.session.query(TranscriptionEntity)
            .filter(
                TranscriptionEntity.video_id == video_id,
                TranscriptionEntity.start >= start,
                TranscriptionEntity.end <= end,
            )
            .order_by(TranscriptionEntity.start)
            .all()
        )
        return [self._to_domain(entity) for entity in entities]

    def delete_by_video_id(self, video_id: str) -> bool:
        """Delete all transcriptions for a video."""
        deleted_count = (
            self.session.query(TranscriptionEntity)
            .filter(TranscriptionEntity.video_id == video_id)
            .delete()
        )
        self.session.commit()
        return deleted_count > 0

    def _to_entity(self, domain: Transcription) -> TranscriptionEntity:
        """Convert domain model to SQLAlchemy entity."""
        return TranscriptionEntity(
            segment_id=domain.segment_id,
            video_id=domain.video_id,
            text=domain.text,
            start=domain.start,
            end=domain.end,
            confidence=domain.confidence,
            speaker=domain.speaker,
        )

    def _to_domain(self, entity: TranscriptionEntity) -> Transcription:
        """Convert SQLAlchemy entity to domain model."""
        return Transcription(
            segment_id=entity.segment_id,
            video_id=entity.video_id,
            text=entity.text,
            start=entity.start,
            end=entity.end,
            confidence=entity.confidence,
            speaker=entity.speaker,
            created_at=entity.created_at,
        )
