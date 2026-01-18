import tempfile

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import PathConfig


def test_path_config_model_creation():
    """Test that PathConfig model can be created with unique constraint."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        engine = create_engine(f"sqlite:///{tmp.name}")
        Base.metadata.create_all(engine)

        session_class = sessionmaker(bind=engine)
        session = session_class()

        # Create path configuration
        path_config = PathConfig(
            path_id="path-1", path="/home/user/videos", recursive="true"
        )

        session.add(path_config)
        session.commit()

        # Query it back
        retrieved = session.query(PathConfig).filter_by(path_id="path-1").first()
        assert retrieved is not None
        assert retrieved.path == "/home/user/videos"
        assert retrieved.recursive == "true"
        assert retrieved.added_at is not None

        session.close()
