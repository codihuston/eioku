"""Test arq worker configuration."""


class TestWorkerSettings:
    """Test WorkerSettings class."""

    def test_worker_settings_has_required_attributes(self):
        """Test WorkerSettings has all required configuration attributes."""
        from src.workers.arq_worker import WorkerSettings

        settings = WorkerSettings()

        # Check required attributes exist
        assert hasattr(settings, "functions")
        assert hasattr(settings, "redis_settings")
        assert hasattr(settings, "queue_name")
        assert hasattr(settings, "max_jobs")
        assert hasattr(settings, "job_timeout")
        assert hasattr(settings, "max_tries")
        assert hasattr(settings, "allow_abort_jobs")

    def test_worker_settings_queue_name_is_jobs(self):
        """Test WorkerSettings uses single 'jobs' queue."""
        from src.workers.arq_worker import WorkerSettings

        settings = WorkerSettings()
        assert settings.queue_name == "ml_jobs"

    def test_worker_settings_default_max_jobs(self):
        """Test WorkerSettings uses default max_jobs=4."""
        from src.workers.arq_worker import WorkerSettings

        settings = WorkerSettings()
        assert settings.max_jobs == 4

    def test_worker_settings_default_job_timeout(self):
        """Test WorkerSettings uses default job_timeout=1800."""
        from src.workers.arq_worker import WorkerSettings

        settings = WorkerSettings()
        assert settings.job_timeout == 1800

    def test_worker_settings_max_tries(self):
        """Test WorkerSettings has max_tries=3."""
        from src.workers.arq_worker import WorkerSettings

        settings = WorkerSettings()
        assert settings.max_tries == 3

    def test_worker_settings_allow_abort_jobs(self):
        """Test WorkerSettings has allow_abort_jobs=True."""
        from src.workers.arq_worker import WorkerSettings

        settings = WorkerSettings()
        assert settings.allow_abort_jobs is True

    def test_worker_settings_redis_settings_default_host(self):
        """Test WorkerSettings uses default Redis host from config."""
        from src.workers.arq_worker import WorkerSettings

        settings = WorkerSettings()
        assert settings.redis_settings.host == "valkey"

    def test_worker_settings_redis_settings_default_port(self):
        """Test WorkerSettings uses default Redis port."""
        from src.workers.arq_worker import WorkerSettings

        settings = WorkerSettings()
        assert settings.redis_settings.port == 6379

    def test_worker_settings_redis_settings_default_database(self):
        """Test WorkerSettings uses default Redis database."""
        from src.workers.arq_worker import WorkerSettings

        settings = WorkerSettings()
        assert settings.redis_settings.database == 0

    def test_worker_settings_redis_settings_type(self):
        """Test WorkerSettings redis_settings is RedisSettings."""
        from arq.connections import RedisSettings

        from src.workers.arq_worker import WorkerSettings

        settings = WorkerSettings()
        assert isinstance(settings.redis_settings, RedisSettings)
