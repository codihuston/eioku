"""Find within video service for keyword search."""

import logging

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..repositories.selection_policy_manager import SelectionPolicyManager

logger = logging.getLogger(__name__)


class FindWithinVideoService:
    """Service for keyword search within videos."""

    def __init__(self, session: Session, policy_manager: SelectionPolicyManager):
        """
        Initialize find within video service.

        Args:
            session: Database session
            policy_manager: Manager for selection policies
        """
        self.session = session
        self.policy_manager = policy_manager

    def find_next(
        self,
        asset_id: str,
        query: str,
        from_ms: int,
        source: str = "all",
    ) -> list[dict]:
        """
        Find next occurrence of query text.

        Args:
            asset_id: The asset (video) ID
            query: Search query text
            from_ms: Starting timestamp in milliseconds
            source: Search source - "transcript", "ocr", or "all"

        Returns:
            List of matches with jump_to, artifact_id, snippet, and source
        """
        logger.debug(
            f"find_next: asset_id={asset_id}, query={query}, "
            f"from_ms={from_ms}, source={source}"
        )

        # Handle empty query
        if not query or not query.strip():
            logger.debug("Empty query provided, returning no results")
            return []

        results = []

        if source in ["transcript", "all"]:
            results.extend(
                self._search_transcript_fts(asset_id, query, from_ms, "next")
            )

        if source in ["ocr", "all"]:
            results.extend(self._search_ocr_fts(asset_id, query, from_ms, "next"))

        # Sort by timestamp ascending
        results.sort(key=lambda x: x["jump_to"]["start_ms"])

        logger.info(f"Find next found {len(results)} matches for query '{query}'")
        return results

    def find_prev(
        self,
        asset_id: str,
        query: str,
        from_ms: int,
        source: str = "all",
    ) -> list[dict]:
        """
        Find previous occurrence of query text.

        Args:
            asset_id: The asset (video) ID
            query: Search query text
            from_ms: Starting timestamp in milliseconds
            source: Search source - "transcript", "ocr", or "all"

        Returns:
            List of matches with jump_to, artifact_id, snippet, and source
        """
        logger.debug(
            f"find_prev: asset_id={asset_id}, query={query}, "
            f"from_ms={from_ms}, source={source}"
        )

        # Handle empty query
        if not query or not query.strip():
            logger.debug("Empty query provided, returning no results")
            return []

        results = []

        if source in ["transcript", "all"]:
            results.extend(
                self._search_transcript_fts(asset_id, query, from_ms, "prev")
            )

        if source in ["ocr", "all"]:
            results.extend(self._search_ocr_fts(asset_id, query, from_ms, "prev"))

        # Sort by timestamp descending
        results.sort(key=lambda x: x["jump_to"]["start_ms"], reverse=True)

        logger.info(f"Find prev found {len(results)} matches for query '{query}'")
        return results

    def _search_transcript_fts(
        self, asset_id: str, query: str, from_ms: int, direction: str
    ) -> list[dict]:
        """
        Search transcript FTS table using full-text search.

        Args:
            asset_id: The asset (video) ID
            query: Search query text
            from_ms: Starting timestamp in milliseconds
            direction: Search direction - "next" or "prev"

        Returns:
            List of matches from transcript FTS
        """
        # Determine if we're using PostgreSQL or SQLite
        bind = self.session.bind
        is_postgresql = bind.dialect.name == "postgresql"

        operator = ">" if direction == "next" else "<"
        order = "ASC" if direction == "next" else "DESC"

        if is_postgresql:
            # PostgreSQL: Use tsvector and tsquery
            # First try FTS, but fall back to ILIKE if query contains only stop words
            tsquery = query.replace(" ", " & ")

            sql = text(
                f"""
                SELECT
                    artifact_id,
                    start_ms,
                    end_ms,
                    ts_headline(
                        'english', text, to_tsquery('english', :query)
                    ) as snippet
                FROM transcript_fts
                WHERE text_tsv @@ to_tsquery('english', :query)
                  AND asset_id = :asset_id
                  AND start_ms {operator} :from_ms
                ORDER BY start_ms {order}
                LIMIT 10
                """
            )

            rows = self.session.execute(
                sql,
                {
                    "query": tsquery,
                    "asset_id": asset_id,
                    "from_ms": from_ms,
                },
            ).fetchall()

            # If FTS returned no results, try case-insensitive LIKE search
            if not rows:
                sql_fallback = text(
                    f"""
                    SELECT
                        artifact_id,
                        start_ms,
                        end_ms,
                        text as snippet
                    FROM transcript_fts
                    WHERE text ILIKE :query
                      AND asset_id = :asset_id
                      AND start_ms {operator} :from_ms
                    ORDER BY start_ms {order}
                    LIMIT 10
                    """
                )

                rows = self.session.execute(
                    sql_fallback,
                    {
                        "query": f"%{query}%",
                        "asset_id": asset_id,
                        "from_ms": from_ms,
                    },
                ).fetchall()

        else:
            # SQLite: Use FTS5 MATCH syntax
            # First get artifact_ids from FTS5 table
            fts_sql = text(
                """
                SELECT
                    artifact_id,
                    snippet(transcript_fts, 4, '<b>', '</b>', '...', 32) as snippet
                FROM transcript_fts
                WHERE transcript_fts MATCH :query
                  AND asset_id = :asset_id
                """
            )

            fts_rows = self.session.execute(
                fts_sql,
                {
                    "query": query,
                    "asset_id": asset_id,
                },
            ).fetchall()

            # Get artifact_ids that matched
            artifact_ids = [row.artifact_id for row in fts_rows]
            snippet_map = {row.artifact_id: row.snippet for row in fts_rows}

            if not artifact_ids:
                return []

            # Now get the metadata (timestamps) from metadata table
            # and filter by direction
            placeholders = ",".join([f":id{i}" for i in range(len(artifact_ids))])
            metadata_sql = text(
                f"""
                SELECT
                    artifact_id,
                    start_ms,
                    end_ms
                FROM transcript_fts_metadata
                WHERE artifact_id IN ({placeholders})
                  AND asset_id = :asset_id
                  AND start_ms {operator} :from_ms
                ORDER BY start_ms {order}
                LIMIT 10
                """
            )

            params = {
                f"id{i}": artifact_id for i, artifact_id in enumerate(artifact_ids)
            }
            params["asset_id"] = asset_id
            params["from_ms"] = from_ms

            rows = self.session.execute(metadata_sql, params).fetchall()

            # Combine with snippets
            results = []
            for row in rows:
                results.append(
                    {
                        "jump_to": {"start_ms": row.start_ms, "end_ms": row.end_ms},
                        "artifact_id": row.artifact_id,
                        "snippet": snippet_map.get(row.artifact_id, ""),
                        "source": "transcript",
                    }
                )
            return results

        # PostgreSQL results
        return [
            {
                "jump_to": {"start_ms": row.start_ms, "end_ms": row.end_ms},
                "artifact_id": row.artifact_id,
                "snippet": row.snippet,
                "source": "transcript",
            }
            for row in rows
        ]

    def _search_ocr_fts(
        self, asset_id: str, query: str, from_ms: int, direction: str
    ) -> list[dict]:
        """
        Search OCR FTS table using full-text search.

        Args:
            asset_id: The asset (video) ID
            query: Search query text
            from_ms: Starting timestamp in milliseconds
            direction: Search direction - "next" or "prev"

        Returns:
            List of matches from OCR FTS
        """
        # Determine if we're using PostgreSQL or SQLite
        bind = self.session.bind
        is_postgresql = bind.dialect.name == "postgresql"

        operator = ">" if direction == "next" else "<"
        order = "ASC" if direction == "next" else "DESC"

        if is_postgresql:
            # PostgreSQL: Use tsvector and tsquery
            # First try FTS, but fall back to ILIKE if query contains only stop words
            tsquery = query.replace(" ", " & ")

            sql = text(
                f"""
                SELECT
                    artifact_id,
                    start_ms,
                    end_ms,
                    ts_headline(
                        'english', text, to_tsquery('english', :query)
                    ) as snippet
                FROM ocr_fts
                WHERE text_tsv @@ to_tsquery('english', :query)
                  AND asset_id = :asset_id
                  AND start_ms {operator} :from_ms
                ORDER BY start_ms {order}
                LIMIT 10
                """
            )

            rows = self.session.execute(
                sql,
                {
                    "query": tsquery,
                    "asset_id": asset_id,
                    "from_ms": from_ms,
                },
            ).fetchall()

            # If FTS returned no results, try case-insensitive LIKE search
            if not rows:
                sql_fallback = text(
                    f"""
                    SELECT
                        artifact_id,
                        start_ms,
                        end_ms,
                        text as snippet
                    FROM ocr_fts
                    WHERE text ILIKE :query
                      AND asset_id = :asset_id
                      AND start_ms {operator} :from_ms
                    ORDER BY start_ms {order}
                    LIMIT 10
                    """
                )

                rows = self.session.execute(
                    sql_fallback,
                    {
                        "query": f"%{query}%",
                        "asset_id": asset_id,
                        "from_ms": from_ms,
                    },
                ).fetchall()

        else:
            # SQLite: Use FTS5 MATCH syntax
            # First get artifact_ids from FTS5 table
            fts_sql = text(
                """
                SELECT
                    artifact_id,
                    snippet(ocr_fts, 4, '<b>', '</b>', '...', 32) as snippet
                FROM ocr_fts
                WHERE ocr_fts MATCH :query
                  AND asset_id = :asset_id
                """
            )

            fts_rows = self.session.execute(
                fts_sql,
                {
                    "query": query,
                    "asset_id": asset_id,
                },
            ).fetchall()

            # Get artifact_ids that matched
            artifact_ids = [row.artifact_id for row in fts_rows]
            snippet_map = {row.artifact_id: row.snippet for row in fts_rows}

            if not artifact_ids:
                return []

            # Now get the metadata (timestamps) from metadata table
            # and filter by direction
            placeholders = ",".join([f":id{i}" for i in range(len(artifact_ids))])
            metadata_sql = text(
                f"""
                SELECT
                    artifact_id,
                    start_ms,
                    end_ms
                FROM ocr_fts_metadata
                WHERE artifact_id IN ({placeholders})
                  AND asset_id = :asset_id
                  AND start_ms {operator} :from_ms
                ORDER BY start_ms {order}
                LIMIT 10
                """
            )

            params = {
                f"id{i}": artifact_id for i, artifact_id in enumerate(artifact_ids)
            }
            params["asset_id"] = asset_id
            params["from_ms"] = from_ms

            rows = self.session.execute(metadata_sql, params).fetchall()

            # Combine with snippets
            results = []
            for row in rows:
                results.append(
                    {
                        "jump_to": {"start_ms": row.start_ms, "end_ms": row.end_ms},
                        "artifact_id": row.artifact_id,
                        "snippet": snippet_map.get(row.artifact_id, ""),
                        "source": "ocr",
                    }
                )
            return results

        # PostgreSQL results
        return [
            {
                "jump_to": {"start_ms": row.start_ms, "end_ms": row.end_ms},
                "artifact_id": row.artifact_id,
                "snippet": row.snippet,
                "source": "ocr",
            }
            for row in rows
        ]
