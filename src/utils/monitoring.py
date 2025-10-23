"""
Production monitoring and logging for RAG pipeline.

Tracks:
- Query volume and patterns
- Latency breakdown by component
- Error rates and types
- Token usage and costs
- Quality metrics (RAGAS scores)

Integrations:
- Langfuse: Production monitoring and tracing
- Loguru: Structured logging
- Alerts on performance degradations

Usage:
    monitor = RAGMonitor(enable_langfuse=True)

    # Track query
    with monitor.track_query("What is beam bending?") as trace:
        # Embed query
        with trace.span("embedding"):
            embedding = embed_query(query)

        # Retrieve
        with trace.span("retrieval"):
            chunks = retrieve(embedding)

        # LLM generation
        with trace.span("llm_generation", tokens=150):
            answer = generate_answer(query, chunks)

    # Get stats
    stats = monitor.get_stats()
    print(f"Total queries: {stats['total_queries']}")
    print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
"""

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from datetime import datetime
from contextlib import contextmanager
from enum import Enum

import numpy as np
from loguru import logger

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("Langfuse not installed. Install with: pip install langfuse")


class ComponentType(str, Enum):
    """RAG pipeline component types."""
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    LLM_GENERATION = "llm_generation"
    CACHE_LOOKUP = "cache_lookup"
    END_TO_END = "end_to_end"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QueryTrace:
    """
    Trace for a single query execution.

    Attributes:
        query_id: Unique query identifier
        query_text: Query text
        timestamp: Query timestamp
        spans: Component execution spans
        total_latency_ms: Total query latency
        tokens_used: Total tokens consumed
        error: Error message if query failed
        metadata: Additional metadata
    """

    query_id: str
    query_text: str
    timestamp: datetime = field(default_factory=datetime.now)
    spans: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_span(
        self,
        component: str,
        latency_ms: float,
        tokens: int = 0,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Add a component execution span."""
        self.spans[component] = {
            "latency_ms": latency_ms,
            "tokens": tokens,
            "error": error,
            "metadata": metadata or {},
        }

        self.tokens_used += tokens

    def calculate_total_latency(self):
        """Calculate total latency from spans."""
        self.total_latency_ms = sum(
            span["latency_ms"] for span in self.spans.values()
        )


@dataclass
class MonitoringStats:
    """
    Aggregated monitoring statistics.

    Attributes:
        total_queries: Total queries processed
        successful_queries: Successful queries
        failed_queries: Failed queries
        avg_latency_ms: Average query latency
        p95_latency_ms: 95th percentile latency
        p99_latency_ms: 99th percentile latency
        total_tokens: Total tokens consumed
        total_cost_usd: Total cost
        error_rate: Error rate percentage
        component_latencies: Average latency by component
        cache_hit_rate: Cache hit rate percentage
    """

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    error_rate: float = 0.0
    component_latencies: Dict[str, float] = field(default_factory=dict)
    cache_hit_rate: float = 0.0
    quality_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Alert:
    """
    Monitoring alert.

    Attributes:
        level: Alert severity level
        component: Component that triggered alert
        message: Alert message
        timestamp: Alert timestamp
        value: Metric value that triggered alert
        threshold: Alert threshold
    """

    level: AlertLevel
    component: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    value: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "component": self.component,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "threshold": self.threshold,
        }


class SpanContext:
    """Context manager for tracking component execution."""

    def __init__(
        self,
        trace: QueryTrace,
        component: str,
        tokens: int = 0,
    ):
        """Initialize span context."""
        self.trace = trace
        self.component = component
        self.tokens = tokens
        self.start_time = None
        self.error = None

    def __enter__(self):
        """Start span."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End span."""
        latency_ms = (time.time() - self.start_time) * 1000

        if exc_type is not None:
            self.error = str(exc_val)

        self.trace.add_span(
            self.component,
            latency_ms,
            tokens=self.tokens,
            error=self.error,
        )

        return False  # Don't suppress exceptions


class RAGMonitor:
    """
    Production monitoring for RAG pipeline.

    Usage:
        monitor = RAGMonitor(enable_langfuse=True)

        # Track query
        with monitor.track_query("What is beam bending?") as trace:
            # Embed
            with trace.span("embedding"):
                embedding = embed_query(query)

            # Retrieve
            with trace.span("retrieval"):
                chunks = retrieve(embedding)

            # Generate
            with trace.span("llm_generation", tokens=150):
                answer = generate(query, chunks)

        # Get stats
        stats = monitor.get_stats()
    """

    def __init__(
        self,
        enable_langfuse: bool = False,
        langfuse_public_key: Optional[str] = None,
        langfuse_secret_key: Optional[str] = None,
        langfuse_host: Optional[str] = None,
        alert_thresholds: Optional[Dict[str, float]] = None,
        window_size: int = 1000,
    ):
        """
        Initialize RAG monitor.

        Args:
            enable_langfuse: Enable Langfuse integration
            langfuse_public_key: Langfuse public key
            langfuse_secret_key: Langfuse secret key
            langfuse_host: Langfuse host URL
            alert_thresholds: Alert threshold configuration
            window_size: Rolling window size for stats
        """
        self.enable_langfuse = enable_langfuse and LANGFUSE_AVAILABLE

        # Initialize Langfuse
        self.langfuse = None
        if self.enable_langfuse:
            try:
                self.langfuse = Langfuse(
                    public_key=langfuse_public_key,
                    secret_key=langfuse_secret_key,
                    host=langfuse_host,
                )
                logger.info("Langfuse monitoring enabled")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self.enable_langfuse = False

        # Monitoring state
        self.traces: deque = deque(maxlen=window_size)
        self.alerts: List[Alert] = []

        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "p95_latency_ms": 5000.0,  # 5 seconds
            "error_rate": 5.0,  # 5%
            "cache_miss_rate": 70.0,  # 70%
        }

        # Stats tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.quality_scores: Dict[str, List[float]] = defaultdict(list)

        logger.info("RAG monitoring initialized")

    @contextmanager
    def track_query(self, query_text: str, metadata: Optional[Dict] = None):
        """
        Track a query execution.

        Args:
            query_text: Query text
            metadata: Additional metadata

        Yields:
            QueryTrace object with span() method
        """
        query_id = f"query_{int(time.time() * 1000000)}"

        trace = QueryTrace(
            query_id=query_id,
            query_text=query_text,
            metadata=metadata or {},
        )

        # Add span method to trace
        def span(
            component: str,
            tokens: int = 0,
        ) -> SpanContext:
            return SpanContext(trace, component, tokens)

        trace.span = span

        start_time = time.time()

        try:
            yield trace
        except Exception as e:
            trace.error = str(e)
            raise
        finally:
            # Calculate total latency
            trace.total_latency_ms = (time.time() - start_time) * 1000

            # Store trace
            self.traces.append(trace)

            # Log to Langfuse
            if self.enable_langfuse:
                self._log_to_langfuse(trace)

            # Check alerts
            self._check_alerts()

            # Log summary
            logger.info(
                f"Query {query_id}: {trace.total_latency_ms:.2f}ms, "
                f"{trace.tokens_used} tokens, "
                f"error={trace.error is not None}"
            )

    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record cache miss."""
        self.cache_misses += 1

    def record_quality_score(self, metric: str, score: float):
        """
        Record quality metric score.

        Args:
            metric: Metric name (e.g., "faithfulness", "answer_relevancy")
            score: Score value (0-1)
        """
        self.quality_scores[metric].append(score)

        logger.debug(f"Quality metric '{metric}': {score:.3f}")

    def get_stats(
        self,
        last_n: Optional[int] = None,
    ) -> MonitoringStats:
        """
        Get monitoring statistics.

        Args:
            last_n: Calculate stats for last N queries (default: all)

        Returns:
            Aggregated monitoring statistics
        """
        # Calculate cache stats even if there are no traces
        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            (self.cache_hits / total_cache_ops * 100)
            if total_cache_ops > 0
            else 0.0
        )

        # Quality scores
        quality_scores = {}
        for metric, scores in self.quality_scores.items():
            if scores:
                quality_scores[metric] = np.mean(scores)

        if not self.traces:
            return MonitoringStats(
                cache_hit_rate=cache_hit_rate,
                quality_scores=quality_scores,
            )

        # Select traces
        traces_list = list(self.traces)
        if last_n is not None:
            traces_list = traces_list[-last_n:]

        # Basic stats
        total_queries = len(traces_list)
        successful_queries = sum(1 for t in traces_list if t.error is None)
        failed_queries = total_queries - successful_queries

        # Latency stats
        latencies = [t.total_latency_ms for t in traces_list]
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95) if len(latencies) > 0 else 0.0
        p99_latency = np.percentile(latencies, 99) if len(latencies) > 0 else 0.0

        # Token and cost stats
        total_tokens = sum(t.tokens_used for t in traces_list)
        total_cost = sum(t.cost_usd for t in traces_list)

        # Error rate
        error_rate = (failed_queries / total_queries * 100) if total_queries > 0 else 0.0

        # Component latencies
        component_latencies = {}
        for component in ComponentType:
            component_name = component.value
            component_times = []

            for trace in traces_list:
                if component_name in trace.spans:
                    component_times.append(trace.spans[component_name]["latency_ms"])

            if component_times:
                component_latencies[component_name] = np.mean(component_times)

        return MonitoringStats(
            total_queries=total_queries,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            error_rate=error_rate,
            component_latencies=component_latencies,
            cache_hit_rate=cache_hit_rate,
            quality_scores=quality_scores,
        )

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        last_n: Optional[int] = None,
    ) -> List[Alert]:
        """
        Get monitoring alerts.

        Args:
            level: Filter by alert level
            last_n: Get last N alerts

        Returns:
            List of alerts
        """
        alerts = self.alerts

        if level is not None:
            alerts = [a for a in alerts if a.level == level]

        if last_n is not None:
            alerts = alerts[-last_n:]

        return alerts

    def _log_to_langfuse(self, trace: QueryTrace):
        """Log trace to Langfuse."""
        if not self.langfuse:
            return

        try:
            # Create Langfuse trace
            langfuse_trace = self.langfuse.trace(
                name="rag_query",
                user_id=trace.metadata.get("user_id"),
                metadata={
                    "query": trace.query_text,
                    **trace.metadata,
                },
            )

            # Add spans
            for component, span_data in trace.spans.items():
                langfuse_trace.span(
                    name=component,
                    input={"query": trace.query_text},
                    metadata=span_data.get("metadata", {}),
                    level="ERROR" if span_data.get("error") else "DEFAULT",
                )

            # Add generation span for LLM
            if "llm_generation" in trace.spans:
                llm_span = trace.spans["llm_generation"]
                langfuse_trace.generation(
                    name="llm_generation",
                    model=trace.metadata.get("model", "unknown"),
                    prompt=trace.query_text,
                    usage={
                        "input": llm_span.get("tokens", 0) // 2,
                        "output": llm_span.get("tokens", 0) // 2,
                        "total": llm_span.get("tokens", 0),
                    },
                )

        except Exception as e:
            logger.error(f"Failed to log to Langfuse: {e}")

    def _check_alerts(self):
        """Check for alert conditions."""
        stats = self.get_stats(last_n=100)  # Check recent window

        # High latency alert
        if "p95_latency_ms" in self.alert_thresholds:
            if stats.p95_latency_ms > self.alert_thresholds["p95_latency_ms"]:
                alert = Alert(
                    level=AlertLevel.WARNING,
                    component="end_to_end",
                    message=f"High p95 latency: {stats.p95_latency_ms:.2f}ms",
                    value=stats.p95_latency_ms,
                    threshold=self.alert_thresholds["p95_latency_ms"],
                )
                self.alerts.append(alert)
                logger.warning(alert.message)

        # High error rate alert
        if "error_rate" in self.alert_thresholds:
            if stats.error_rate > self.alert_thresholds["error_rate"]:
                alert = Alert(
                    level=AlertLevel.ERROR,
                    component="pipeline",
                    message=f"High error rate: {stats.error_rate:.1f}%",
                    value=stats.error_rate,
                    threshold=self.alert_thresholds["error_rate"],
                )
                self.alerts.append(alert)
                logger.error(alert.message)

        # Low cache hit rate alert
        if "cache_miss_rate" in self.alert_thresholds:
            cache_miss_rate = 100.0 - stats.cache_hit_rate
            if cache_miss_rate > self.alert_thresholds["cache_miss_rate"]:
                alert = Alert(
                    level=AlertLevel.INFO,
                    component="cache",
                    message=f"High cache miss rate: {cache_miss_rate:.1f}%",
                    value=cache_miss_rate,
                    threshold=self.alert_thresholds["cache_miss_rate"],
                )
                self.alerts.append(alert)
                logger.info(alert.message)

    def print_stats_report(self):
        """Print monitoring stats report."""
        stats = self.get_stats()

        print("\n" + "=" * 70)
        print("MONITORING REPORT")
        print("=" * 70)

        print(f"\nQuery Statistics:")
        print(f"  Total queries: {stats.total_queries}")
        print(f"  Successful: {stats.successful_queries}")
        print(f"  Failed: {stats.failed_queries}")
        print(f"  Error rate: {stats.error_rate:.2f}%")

        print(f"\nLatency Statistics:")
        print(f"  Average: {stats.avg_latency_ms:.2f}ms")
        print(f"  p95: {stats.p95_latency_ms:.2f}ms")
        print(f"  p99: {stats.p99_latency_ms:.2f}ms")

        print(f"\nComponent Latencies:")
        for component, latency in stats.component_latencies.items():
            print(f"  {component}: {latency:.2f}ms")

        print(f"\nResource Usage:")
        print(f"  Total tokens: {stats.total_tokens:,}")
        print(f"  Total cost: ${stats.total_cost_usd:.4f}")

        print(f"\nCache Performance:")
        print(f"  Hit rate: {stats.cache_hit_rate:.1f}%")

        if stats.quality_scores:
            print(f"\nQuality Metrics:")
            for metric, score in stats.quality_scores.items():
                print(f"  {metric}: {score:.3f}")

        # Recent alerts
        recent_alerts = self.get_alerts(last_n=5)
        if recent_alerts:
            print(f"\nRecent Alerts:")
            for alert in recent_alerts:
                print(
                    f"  [{alert.level.value.upper()}] {alert.component}: {alert.message}"
                )

        print("\n" + "=" * 70)

    def save_report(self, output_path: str = "monitoring_report.json"):
        """
        Save monitoring report to JSON.

        Args:
            output_path: Output file path
        """
        stats = self.get_stats()
        alerts = [a.to_dict() for a in self.get_alerts(last_n=100)]

        report = {
            "timestamp": datetime.now().isoformat(),
            "stats": stats.to_dict(),
            "alerts": alerts,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Monitoring report saved to {output_path}")


if __name__ == "__main__":
    logger.add("logs/monitoring.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print("PRODUCTION MONITORING - RAG Pipeline")
    print("=" * 70)
    print("\nFeatures:")
    print("  • Query tracking with component-level spans")
    print("  • Latency breakdown (embedding, retrieval, reranking, LLM)")
    print("  • Token usage and cost tracking")
    print("  • Cache performance monitoring")
    print("  • Quality metrics (RAGAS scores)")
    print("  • Automatic alerts on degradations")
    print("  • Langfuse integration (optional)")
    print("\nUsage:")
    print("  monitor = RAGMonitor(enable_langfuse=True)")
    print("  with monitor.track_query('What is stress?') as trace:")
    print("      with trace.span('embedding'):")
    print("          embedding = embed_query(query)")
    print("      with trace.span('llm_generation', tokens=150):")
    print("          answer = generate(query, chunks)")
    print("  stats = monitor.get_stats()")
    print("=" * 70 + "\n")
