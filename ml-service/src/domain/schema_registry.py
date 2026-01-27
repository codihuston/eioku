"""Schema registry for artifact payload validation."""


from pydantic import BaseModel


class SchemaNotFoundError(Exception):
    """Raised when schema is not registered."""

    pass


class SchemaRegistry:
    """
    Central registry for artifact schemas.

    Maps (artifact_type, schema_version) to Pydantic models for validation.
    Ensures type safety and consistent validation across the system.
    """

    _schemas: dict[tuple[str, int], type[BaseModel]] = {}

    @classmethod
    def register(
        cls, artifact_type: str, schema_version: int, schema: type[BaseModel]
    ) -> None:
        """
        Register a schema for an artifact type and version.

        Args:
            artifact_type: The artifact type (e.g., "transcript.segment")
            schema_version: The schema version number (must be >= 1)
            schema: The Pydantic model class for validation

        Raises:
            ValueError: If schema is already registered or invalid parameters
        """
        if not artifact_type:
            raise ValueError("artifact_type cannot be empty")
        if schema_version < 1:
            raise ValueError("schema_version must be >= 1")
        if not issubclass(schema, BaseModel):
            raise ValueError("schema must be a Pydantic BaseModel subclass")

        key = (artifact_type, schema_version)
        if key in cls._schemas:
            raise ValueError(
                f"Schema already registered for {artifact_type} v{schema_version}"
            )

        cls._schemas[key] = schema

    @classmethod
    def get_schema(cls, artifact_type: str, schema_version: int) -> type[BaseModel]:
        """
        Get schema for artifact type and version.

        Args:
            artifact_type: The artifact type
            schema_version: The schema version number

        Returns:
            The Pydantic model class for validation

        Raises:
            SchemaNotFoundError: If no schema is registered for the given type/version
        """
        key = (artifact_type, schema_version)
        if key not in cls._schemas:
            raise SchemaNotFoundError(
                f"No schema registered for {artifact_type} v{schema_version}"
            )
        return cls._schemas[key]

    @classmethod
    def validate(
        cls, artifact_type: str, schema_version: int, payload: dict
    ) -> BaseModel:
        """
        Validate and parse payload using registered schema.

        Args:
            artifact_type: The artifact type
            schema_version: The schema version number
            payload: The payload dictionary to validate

        Returns:
            Validated Pydantic model instance

        Raises:
            SchemaNotFoundError: If no schema is registered
            ValidationError: If payload fails validation
        """
        schema = cls.get_schema(artifact_type, schema_version)
        return schema(**payload)

    @classmethod
    def serialize(
        cls, artifact_type: str, schema_version: int, payload: BaseModel
    ) -> str:
        """
        Serialize payload to JSON string.

        Args:
            artifact_type: The artifact type
            schema_version: The schema version number
            payload: The validated Pydantic model instance

        Returns:
            JSON string representation of the payload

        Raises:
            SchemaNotFoundError: If no schema is registered
        """
        # Verify schema is registered (for consistency)
        cls.get_schema(artifact_type, schema_version)
        return payload.model_dump_json()

    @classmethod
    def is_registered(cls, artifact_type: str, schema_version: int) -> bool:
        """
        Check if a schema is registered.

        Args:
            artifact_type: The artifact type
            schema_version: The schema version number

        Returns:
            True if schema is registered, False otherwise
        """
        key = (artifact_type, schema_version)
        return key in cls._schemas

    @classmethod
    def list_registered_schemas(cls) -> list[tuple[str, int]]:
        """
        List all registered schemas.

        Returns:
            List of (artifact_type, schema_version) tuples
        """
        return list(cls._schemas.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered schemas.

        Primarily for testing purposes.
        """
        cls._schemas.clear()
