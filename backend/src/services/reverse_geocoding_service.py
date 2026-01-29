"""Reverse geocoding service for converting GPS coordinates to location names."""

import logging
from functools import lru_cache

from geopy.geocoders import Nominatim

logger = logging.getLogger(__name__)


class ReverseGeocodingService:
    """Service for reverse geocoding GPS coordinates to location names."""

    def __init__(self, timeout: int = 10):
        """Initialize the reverse geocoding service.

        Args:
            timeout: Timeout in seconds for geocoding requests
        """
        self.geocoder = Nominatim(user_agent="eioku_video_metadata", timeout=timeout)

    @lru_cache(maxsize=1000)
    def get_location_info(
        self, latitude: float, longitude: float
    ) -> dict[str, str | None]:
        """Get location information from GPS coordinates.

        Args:
            latitude: GPS latitude coordinate
            longitude: GPS longitude coordinate

        Returns:
            Dictionary with 'country', 'state', and 'city' keys.
            Values are None if not found.
        """
        try:
            logger.info(f"🌍 Reverse geocoding coordinates: {latitude}, {longitude}")
            location = self.geocoder.reverse(f"{latitude}, {longitude}", language="en")
            address = location.raw.get("address", {})

            result = {
                "country": address.get("country"),
                "state": address.get("state"),
                "city": address.get("city")
                or address.get("town")
                or address.get("village"),
            }
            logger.info(f"✓ Geocoding result: {result}")
            return result
        except Exception as e:
            logger.warning(
                f"Failed to reverse geocode coordinates ({latitude}, {longitude}): {e}"
            )
            return {"country": None, "state": None, "city": None}
