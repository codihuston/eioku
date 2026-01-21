"""Tests for schema registry."""

import pytest
from pydantic import BaseModel, Field

from src.domain.schema_registry import SchemaNotFoundError, SchemaRegistry


class TestSchemaV1(BaseModel):
    """Test schema for testing."""

    name: str = Field(..., description="Test name")
    value: int = Field(..., ge=0, description="Test value")


class TestSchemaRegistry:
    """Tests for SchemaRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        SchemaRegistry.clear()

    def test_register_schema(self):
        """Test registering a schema."""
        SchemaRegistry.register("test.type", 1, TestSchemaV1)

        assert SchemaRegistry.is_registered("test.type", 1) is True

    def test_register_duplicate_schema_raises_error(self):
        """Test that registering duplicate schema raises error."""
        SchemaRegistry.register("test.type", 1, TestSchemaV1)

        with pytest.raises(ValueError, match="Schema already registered"):
            SchemaRegistry.register("test.type", 1, TestSchemaV1)

    def test_get_schema(self):
        """Test getting a registered schema."""
        SchemaRegistry.register("test.type", 1, TestSchemaV1)

        schema = SchemaRegistry.get_schema("test.type", 1)
        assert schema == TestSchemaV1

    def test_get_unregistered_schema_raises_error(self):
        """Test that getting unregistered schema raises error."""
        with pytest.raises(SchemaNotFoundError, match="No schema registered"):
            SchemaRegistry.get_schema("unknown.type", 1)

    def test_validate_valid_payload(self):
        """Test validating a valid payload."""
        SchemaRegistry.register("test.type", 1, TestSchemaV1)

        payload = {"name": "test", "value": 42}
        validated = SchemaRegistry.validate("test.type", 1, payload)

        assert validated.name == "test"
        assert validated.value == 42

    def test_validate_invalid_payload_raises_error(self):
        """Test that validating invalid payload raises error."""
        SchemaRegistry.register("test.type", 1, TestSchemaV1)

        payload = {"name": "test", "value": -1}  # Negative value not allowed

        with pytest.raises(Exception):  # Pydantic ValidationError
            SchemaRegistry.validate("test.type", 1, payload)

    def test_serialize_payload(self):
        """Test serializing a payload."""
        SchemaRegistry.register("test.type", 1, TestSchemaV1)

        payload = TestSchemaV1(name="test", value=42)
        json_str = SchemaRegistry.serialize("test.type", 1, payload)

        assert '"name":"test"' in json_str
        assert '"value":42' in json_str

    def test_list_registered_schemas(self):
        """Test listing registered schemas."""
        SchemaRegistry.register("test.type", 1, TestSchemaV1)
        SchemaRegistry.register("test.type", 2, TestSchemaV1)
        SchemaRegistry.register("other.type", 1, TestSchemaV1)

        schemas = SchemaRegistry.list_registered_schemas()

        assert len(schemas) == 3
        assert ("test.type", 1) in schemas
        assert ("test.type", 2) in schemas
        assert ("other.type", 1) in schemas

    def test_register_invalid_schema_version_raises_error(self):
        """Test that registering with invalid version raises error."""
        with pytest.raises(ValueError, match="schema_version must be >= 1"):
            SchemaRegistry.register("test.type", 0, TestSchemaV1)

    def test_register_empty_artifact_type_raises_error(self):
        """Test that registering with empty artifact_type raises error."""
        with pytest.raises(ValueError, match="artifact_type cannot be empty"):
            SchemaRegistry.register("", 1, TestSchemaV1)
