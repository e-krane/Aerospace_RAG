"""
Tests for production monitoring and logging.

Tests:
- Query tracking with spans
- Latency measurement
- Token usage tracking
- Cache hit/miss tracking
- Alert triggering
- Statistics aggregation
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.utils.monitoring import (
    RAGMonitor,
    QueryTrace,
    MonitoringStats,
    Alert,
    AlertLevel,
    ComponentType,
)


class TestQueryTrace:
    """Test QueryTrace dataclass."""

    def test_trace_initialization(self):
        """Test trace initializes correctly."""
        trace = QueryTrace(
            query_id="query_001",
            query_text="What is beam bending?",
        )

        assert trace.query_id == "query_001"
        assert trace.query_text == "What is beam bending?"
        assert len(trace.spans) == 0
        assert trace.tokens_used == 0
        assert trace.error is None

    def test_add_span(self):
        """Test adding execution spans."""
        trace = QueryTrace(query_id="query_001", query_text="Test")

        trace.add_span("embedding", latency_ms=50.0, tokens=10)
        trace.add_span("retrieval", latency_ms=100.0)
        trace.add_span("llm_generation", latency_ms=1500.0, tokens=150)

        assert len(trace.spans) == 3
        assert trace.spans["embedding"]["latency_ms"] == 50.0
        assert trace.spans["llm_generation"]["tokens"] == 150
        assert trace.tokens_used == 160

    def test_calculate_total_latency(self):
        """Test total latency calculation."""
        trace = QueryTrace(query_id="query_001", query_text="Test")

        trace.add_span("embedding", latency_ms=50.0)
        trace.add_span("retrieval", latency_ms=100.0)
        trace.add_span("llm_generation", latency_ms=1500.0)

        trace.calculate_total_latency()

        assert trace.total_latency_ms == 1650.0


class TestRAGMonitor:
    """Test RAG monitoring functionality."""

    def test_monitor_initialization(self):
        """Test monitor initializes correctly."""
        monitor = RAGMonitor(enable_langfuse=False)

        assert monitor.enable_langfuse is False
        assert len(monitor.traces) == 0
        assert len(monitor.alerts) == 0

    def test_track_query_basic(self):
        """Test basic query tracking."""
        monitor = RAGMonitor(enable_langfuse=False)

        with monitor.track_query("What is stress?") as trace:
            time.sleep(0.01)  # Simulate work

        assert len(monitor.traces) == 1

        recorded_trace = monitor.traces[0]
        assert recorded_trace.query_text == "What is stress?"
        assert recorded_trace.total_latency_ms > 0

    def test_track_query_with_spans(self):
        """Test query tracking with component spans."""
        monitor = RAGMonitor(enable_langfuse=False)

        with monitor.track_query("What is beam bending?") as trace:
            # Embedding span
            with trace.span("embedding"):
                time.sleep(0.01)

            # Retrieval span
            with trace.span("retrieval"):
                time.sleep(0.01)

            # LLM generation span
            with trace.span("llm_generation", tokens=150):
                time.sleep(0.01)

        recorded_trace = monitor.traces[0]

        assert "embedding" in recorded_trace.spans
        assert "retrieval" in recorded_trace.spans
        assert "llm_generation" in recorded_trace.spans
        assert recorded_trace.tokens_used == 150

    def test_track_query_with_error(self):
        """Test query tracking with error."""
        monitor = RAGMonitor(enable_langfuse=False)

        with pytest.raises(ValueError):
            with monitor.track_query("Test query") as trace:
                raise ValueError("Test error")

        recorded_trace = monitor.traces[0]

        assert recorded_trace.error == "Test error"

    def test_cache_tracking(self):
        """Test cache hit/miss tracking."""
        monitor = RAGMonitor(enable_langfuse=False)

        # Record cache operations
        monitor.record_cache_hit()
        monitor.record_cache_hit()
        monitor.record_cache_miss()

        assert monitor.cache_hits == 2
        assert monitor.cache_misses == 1

        stats = monitor.get_stats()
        assert abs(stats.cache_hit_rate - 66.67) < 0.1

    def test_quality_score_tracking(self):
        """Test quality metric tracking."""
        monitor = RAGMonitor(enable_langfuse=False)

        # Record quality scores
        monitor.record_quality_score("faithfulness", 0.95)
        monitor.record_quality_score("faithfulness", 0.92)
        monitor.record_quality_score("answer_relevancy", 0.88)

        assert len(monitor.quality_scores["faithfulness"]) == 2
        assert len(monitor.quality_scores["answer_relevancy"]) == 1

    def test_get_stats_empty(self):
        """Test getting stats with no queries."""
        monitor = RAGMonitor(enable_langfuse=False)

        stats = monitor.get_stats()

        assert stats.total_queries == 0
        assert stats.avg_latency_ms == 0.0

    def test_get_stats_with_queries(self):
        """Test getting stats with queries."""
        monitor = RAGMonitor(enable_langfuse=False)

        # Track multiple queries
        for i in range(10):
            with monitor.track_query(f"Query {i}") as trace:
                with trace.span("embedding"):
                    time.sleep(0.001)
                with trace.span("llm_generation", tokens=100):
                    time.sleep(0.001)

        stats = monitor.get_stats()

        assert stats.total_queries == 10
        assert stats.successful_queries == 10
        assert stats.failed_queries == 0
        assert stats.avg_latency_ms > 0
        assert stats.total_tokens == 1000  # 10 queries * 100 tokens
        assert "embedding" in stats.component_latencies
        assert "llm_generation" in stats.component_latencies

    def test_get_stats_last_n(self):
        """Test getting stats for last N queries."""
        monitor = RAGMonitor(enable_langfuse=False)

        # Track 20 queries
        for i in range(20):
            with monitor.track_query(f"Query {i}") as trace:
                time.sleep(0.001)

        # Get stats for last 5 queries
        stats = monitor.get_stats(last_n=5)

        assert stats.total_queries == 5

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        monitor = RAGMonitor(enable_langfuse=False)

        # 7 successful, 3 failed
        for i in range(7):
            with monitor.track_query(f"Query {i}") as trace:
                pass

        for i in range(3):
            with pytest.raises(Exception):
                with monitor.track_query(f"Error query {i}") as trace:
                    raise Exception("Test error")

        stats = monitor.get_stats()

        assert stats.total_queries == 10
        assert stats.successful_queries == 7
        assert stats.failed_queries == 3
        assert stats.error_rate == 30.0

    def test_alert_high_latency(self):
        """Test high latency alert."""
        monitor = RAGMonitor(
            enable_langfuse=False,
            alert_thresholds={"p95_latency_ms": 10.0},  # Low threshold
        )

        # Generate queries with high latency
        for i in range(100):
            with monitor.track_query(f"Query {i}") as trace:
                time.sleep(0.05)  # 50ms - exceeds threshold

        alerts = monitor.get_alerts(level=AlertLevel.WARNING)

        assert len(alerts) > 0
        assert any("latency" in a.message.lower() for a in alerts)

    def test_alert_high_error_rate(self):
        """Test high error rate alert."""
        monitor = RAGMonitor(
            enable_langfuse=False,
            alert_thresholds={"error_rate": 5.0},  # 5% threshold
        )

        # Generate queries with high error rate (20%)
        for i in range(80):
            with monitor.track_query(f"Query {i}") as trace:
                pass

        for i in range(20):
            with pytest.raises(Exception):
                with monitor.track_query(f"Error {i}") as trace:
                    raise Exception("Test error")

        alerts = monitor.get_alerts(level=AlertLevel.ERROR)

        assert len(alerts) > 0
        assert any("error" in a.message.lower() for a in alerts)

    def test_alert_low_cache_hit_rate(self):
        """Test low cache hit rate alert."""
        monitor = RAGMonitor(
            enable_langfuse=False,
            alert_thresholds={"cache_miss_rate": 50.0},  # 50% miss threshold
        )

        # Generate cache operations (20% hit rate = 80% miss rate)
        for _ in range(20):
            monitor.record_cache_hit()

        for _ in range(80):
            monitor.record_cache_miss()

        # Trigger alert check
        with monitor.track_query("Test") as trace:
            pass

        alerts = monitor.get_alerts(level=AlertLevel.INFO)

        assert len(alerts) > 0
        assert any("cache" in a.message.lower() for a in alerts)

    def test_component_latency_tracking(self):
        """Test component-specific latency tracking."""
        monitor = RAGMonitor(enable_langfuse=False)

        # Track queries with different component latencies
        for i in range(10):
            with monitor.track_query(f"Query {i}") as trace:
                with trace.span("embedding"):
                    time.sleep(0.01)
                with trace.span("retrieval"):
                    time.sleep(0.02)
                with trace.span("llm_generation", tokens=100):
                    time.sleep(0.03)

        stats = monitor.get_stats()

        assert "embedding" in stats.component_latencies
        assert "retrieval" in stats.component_latencies
        assert "llm_generation" in stats.component_latencies

        # Retrieval should be slower than embedding
        assert (
            stats.component_latencies["retrieval"]
            > stats.component_latencies["embedding"]
        )

    def test_window_size_limit(self):
        """Test rolling window size limit."""
        monitor = RAGMonitor(enable_langfuse=False, window_size=10)

        # Track 20 queries (should only keep last 10)
        for i in range(20):
            with monitor.track_query(f"Query {i}") as trace:
                pass

        assert len(monitor.traces) == 10

    def test_metadata_tracking(self):
        """Test query metadata tracking."""
        monitor = RAGMonitor(enable_langfuse=False)

        metadata = {"user_id": "user_123", "session_id": "session_456"}

        with monitor.track_query("Test query", metadata=metadata) as trace:
            pass

        recorded_trace = monitor.traces[0]

        assert recorded_trace.metadata["user_id"] == "user_123"
        assert recorded_trace.metadata["session_id"] == "session_456"


class TestMonitoringStats:
    """Test MonitoringStats dataclass."""

    def test_stats_to_dict(self):
        """Test stats conversion to dictionary."""
        stats = MonitoringStats(
            total_queries=100,
            successful_queries=95,
            failed_queries=5,
            avg_latency_ms=1500.5,
        )

        stats_dict = stats.to_dict()

        assert stats_dict["total_queries"] == 100
        assert stats_dict["avg_latency_ms"] == 1500.5


class TestAlert:
    """Test Alert dataclass."""

    def test_alert_creation(self):
        """Test alert creation."""
        alert = Alert(
            level=AlertLevel.WARNING,
            component="retrieval",
            message="High latency detected",
            value=150.0,
            threshold=100.0,
        )

        assert alert.level == AlertLevel.WARNING
        assert alert.component == "retrieval"
        assert alert.value == 150.0

    def test_alert_to_dict(self):
        """Test alert conversion to dictionary."""
        alert = Alert(
            level=AlertLevel.ERROR,
            component="llm",
            message="Generation failed",
        )

        alert_dict = alert.to_dict()

        assert alert_dict["level"] == "error"
        assert alert_dict["component"] == "llm"
        assert "timestamp" in alert_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
