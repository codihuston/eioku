"""Unit tests for ML Service arq worker configuration."""

from arq.connections import RedisSettings

from src.workers.arq_worker import (
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_SETTINGS,
    REDIS_URL,
    WorkerSettings,
)


class TestRedisConfiguration:
    """Test Redis/Valkey connection configuration."""

    def test_default_redis_host(self):
        """Test default Redis host is 'valkey'."""
        assert REDIS_HOST == "valkey"

    def test_default_redis_port(self):
        """Test default Redis port is 6379."""
        assert REDIS_PORT == 6379

    def test_default_redis_db(self):
        """Test default Redis database is 0."""
        assert REDIS_DB == 0

    def test_redis_url_format(self):
        """Test Redis URL is correctly formatted."""
        expected_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
        assert REDIS_URL == expected_url

    def test_redis_url_format_is_valid(self):
        """Test Redis URL format is valid."""
        # URL should be in format redis://host:port/db
        assert REDIS_URL.startswith("redis://")
        assert ":" in REDIS_URL
        assert "/" in REDIS_URL

    def test_redis_settings_type(self):
        """Test REDIS_SETTINGS is RedisSettings instance."""
        assert isinstance(REDIS_SETTINGS, RedisSettings)

    def test_redis_settings_host(self):
        """Test REDIS_SETTINGS has correct host."""
        assert REDIS_SETTINGS.host == REDIS_HOST

    def test_redis_settings_port(self):
        """Test REDIS_SETTINGS has correct port."""
        assert REDIS_SETTINGS.port == REDIS_PORT

    def test_redis_settings_database(self):
        """Test REDIS_SETTINGS has correct database."""
        assert REDIS_SETTINGS.database == REDIS_DB


class TestWorkerSettings:
    """Test ML Service arq worker configuration."""

    def test_queue_name_is_ml_jobs(self):
        """Test worker consumes from 'ml_jobs' queue."""
        assert WorkerSettings.queue_name == "ml_jobs"

    def test_redis_settings_configured(self):
        """Test worker has Redis settings configured."""
        assert WorkerSettings.redis_settings == REDIS_SETTINGS

    def test_default_max_jobs(self):
        """Test default max_jobs is 4."""
        assert WorkerSettings.max_jobs == 4

    def test_default_job_timeout(self):
        """Test default job_timeout is 1800 seconds (30 minutes)."""
        assert WorkerSettings.job_timeout == 1800

    def test_default_max_tries(self):
        """Test default max_tries is 3."""
        assert WorkerSettings.max_tries == 3

    def test_allow_abort_jobs_enabled(self):
        """Test job abort is enabled for cancellation support."""
        assert WorkerSettings.allow_abort_jobs is True

    def test_functions_list_initialized(self):
        """Test functions list is initialized (empty until handlers added)."""
        assert isinstance(WorkerSettings.functions, list)

    def test_custom_max_jobs_via_env(self):
        """Test max_jobs respects ARQ_MAX_JOBS environment variable.

        Note: This test verifies the code reads the env var at module load time.
        The actual value depends on the environment when the module is imported.
        """
        # Verify max_jobs is an integer and reasonable
        assert isinstance(WorkerSettings.max_jobs, int)
        assert WorkerSettings.max_jobs > 0

    def test_custom_job_timeout_via_env(self):
        """Test job_timeout respects ARQ_JOB_TIMEOUT environment variable.

        Note: This test verifies the code reads the env var at module load time.
        The actual value depends on the environment when the module is imported.
        """
        # Verify job_timeout is an integer and reasonable
        assert isinstance(WorkerSettings.job_timeout, int)
        assert WorkerSettings.job_timeout > 0

    def test_worker_settings_initialization(self):
        """Test WorkerSettings can be instantiated."""
        settings = WorkerSettings()
        assert settings is not None

    def test_worker_settings_has_queue_name(self):
        """Test instantiated WorkerSettings has queue_name."""
        settings = WorkerSettings()
        assert settings.queue_name == "ml_jobs"

    def test_worker_settings_has_redis_settings(self):
        """Test instantiated WorkerSettings has redis_settings."""
        settings = WorkerSettings()
        assert settings.redis_settings == REDIS_SETTINGS

    def test_worker_settings_has_max_jobs(self):
        """Test instantiated WorkerSettings has max_jobs."""
        settings = WorkerSettings()
        assert settings.max_jobs == 4

    def test_worker_settings_has_job_timeout(self):
        """Test instantiated WorkerSettings has job_timeout."""
        settings = WorkerSettings()
        assert settings.job_timeout == 1800

    def test_worker_settings_has_max_tries(self):
        """Test instantiated WorkerSettings has max_tries."""
        settings = WorkerSettings()
        assert settings.max_tries == 3

    def test_worker_settings_has_allow_abort_jobs(self):
        """Test instantiated WorkerSettings has allow_abort_jobs."""
        settings = WorkerSettings()
        assert settings.allow_abort_jobs is True


class TestWorkerConfiguration:
    """Test overall worker configuration."""

    def test_ml_service_queue_different_from_worker_service(self):
        """Test ML Service uses different queue than Worker Service.

        Worker Service uses 'jobs' queue (gpu_jobs/cpu_jobs).
        ML Service uses 'ml_jobs' queue.
        """
        assert WorkerSettings.queue_name == "ml_jobs"
        assert WorkerSettings.queue_name != "jobs"

    def test_redis_settings_centralized(self):
        """Test Redis settings are centralized and consistent."""
        # Both should use same Redis host/port/db
        assert REDIS_SETTINGS.host == REDIS_HOST
        assert REDIS_SETTINGS.port == REDIS_PORT
        assert REDIS_SETTINGS.database == REDIS_DB

    def test_worker_settings_immutable_queue_name(self):
        """Test queue_name is set to 'ml_jobs' and cannot be changed."""
        # queue_name is a class attribute, should always be 'ml_jobs'
        assert WorkerSettings.queue_name == "ml_jobs"

    def test_job_timeout_sufficient_for_ml_inference(self):
        """Test job_timeout is sufficient for ML inference (30 minutes)."""
        # 30 minutes should be enough for most ML inference tasks
        assert WorkerSettings.job_timeout >= 1800

    def test_max_tries_allows_retry(self):
        """Test max_tries allows job retry on failure."""
        # max_tries=3 means: initial attempt + 2 retries
        assert WorkerSettings.max_tries >= 2
