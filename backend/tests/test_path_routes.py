"""Test that path routes are properly registered."""

from src.main_api import create_app


def test_path_routes_registered():
    """Test that path routes are registered in the FastAPI app."""
    app = create_app()

    # Get all registered routes
    routes = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            routes.append((route.path, route.methods))

    # Check that our path endpoints are registered
    paths_found = set()
    for path, methods in routes:
        paths_found.add(path)

    expected_paths = {"/v1/paths", "/v1/paths/discover", "/v1/paths/validate"}

    for expected_path in expected_paths:
        assert expected_path in paths_found, f"Route {expected_path} not found"


def test_app_creation():
    """Test that the app can be created without errors."""
    app = create_app()
    assert app is not None
    assert app.title == "Eioku - Semantic Video Search API"
