"""
Comprehensive metrics collection system for tracking resource consumption.

Tracks tokens, characters, memory, CPU, GPU, and execution time for each processing step.
Designed for real-time monitoring with per-short metrics output.
"""

import json
import os
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Optional, List, Any
import statistics

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. CPU/Memory tracking disabled.")

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    GPU_COUNT = pynvml.nvmlDeviceGetCount()
except:
    GPU_AVAILABLE = False
    GPU_COUNT = 0


@dataclass
class ResourceSample:
    """Single sample of system resources"""
    timestamp: float
    memory_mb: float
    cpu_percent: float
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None


@dataclass
class StepMetrics:
    """Metrics collected for a single processing step"""
    duration_seconds: float = 0.0

    # Token usage (Gemini)
    tokens: Dict[str, int] = field(default_factory=lambda: {
        "prompt": 0,
        "completion": 0,
        "total": 0,
        "cached": 0
    })

    # Character usage (ElevenLabs)
    characters_used: int = 0

    # System resources
    memory_mb: Dict[str, float] = field(default_factory=lambda: {
        "peak": 0.0,
        "average": 0.0,
        "start": 0.0,
        "end": 0.0
    })

    cpu_percent: Dict[str, float] = field(default_factory=lambda: {
        "peak": 0.0,
        "average": 0.0
    })

    gpu_percent: Optional[Dict[str, float]] = field(default_factory=lambda: {
        "peak": 0.0,
        "average": 0.0
    })

    gpu_memory_mb: Optional[Dict[str, float]] = field(default_factory=lambda: {
        "peak": 0.0,
        "average": 0.0
    })

    timestamp: str = ""


class StepTracker:
    """Context manager for tracking a processing step with background resource sampling"""

    def __init__(self, step_name: str, collector: 'MetricsCollector'):
        self.step_name = step_name
        self.collector = collector
        self.start_time = None
        self.samples: List[ResourceSample] = []
        self.sampling_thread = None
        self.stop_sampling = threading.Event()

        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
        else:
            self.process = None

    def _sample_resources(self):
        """Background thread function to sample resources periodically"""
        while not self.stop_sampling.is_set():
            sample = ResourceSample(timestamp=time.time(), memory_mb=0.0, cpu_percent=0.0)

            try:
                if self.process:
                    # Memory in MB
                    sample.memory_mb = self.process.memory_info().rss / (1024 * 1024)
                    # CPU percent
                    sample.cpu_percent = self.process.cpu_percent()

                # GPU metrics (first GPU only for simplicity)
                if GPU_AVAILABLE and GPU_COUNT > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    sample.gpu_percent = util.gpu
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    sample.gpu_memory_mb = mem_info.used / (1024 * 1024)

                self.samples.append(sample)
            except Exception as e:
                # Silently ignore sampling errors
                pass

            # Sample every 0.5 seconds
            self.stop_sampling.wait(0.5)

    def __enter__(self):
        self.start_time = time.time()

        # Initialize step metrics
        if self.step_name not in self.collector.steps:
            self.collector.steps[self.step_name] = StepMetrics(
                timestamp=datetime.utcnow().isoformat() + 'Z'
            )

        # Take initial sample
        if self.process or GPU_AVAILABLE:
            self.sampling_thread = threading.Thread(
                target=self._sample_resources,
                daemon=True
            )
            self.sampling_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop sampling
        self.stop_sampling.set()
        if self.sampling_thread:
            self.sampling_thread.join(timeout=2.0)

        # Calculate duration
        duration = time.time() - self.start_time
        metrics = self.collector.steps[self.step_name]
        metrics.duration_seconds += duration

        # Process samples if available
        if self.samples:
            memory_values = [s.memory_mb for s in self.samples]
            cpu_values = [s.cpu_percent for s in self.samples if s.cpu_percent > 0]

            if memory_values:
                metrics.memory_mb["peak"] = max(metrics.memory_mb["peak"], max(memory_values))
                metrics.memory_mb["average"] = statistics.mean(memory_values)
                metrics.memory_mb["start"] = memory_values[0]
                metrics.memory_mb["end"] = memory_values[-1]

            if cpu_values:
                metrics.cpu_percent["peak"] = max(metrics.cpu_percent["peak"], max(cpu_values))
                metrics.cpu_percent["average"] = statistics.mean(cpu_values)

            # GPU metrics
            gpu_percent_values = [s.gpu_percent for s in self.samples if s.gpu_percent is not None]
            gpu_memory_values = [s.gpu_memory_mb for s in self.samples if s.gpu_memory_mb is not None]

            if gpu_percent_values and metrics.gpu_percent:
                metrics.gpu_percent["peak"] = max(metrics.gpu_percent["peak"], max(gpu_percent_values))
                metrics.gpu_percent["average"] = statistics.mean(gpu_percent_values)

            if gpu_memory_values and metrics.gpu_memory_mb:
                metrics.gpu_memory_mb["peak"] = max(metrics.gpu_memory_mb["peak"], max(gpu_memory_values))
                metrics.gpu_memory_mb["average"] = statistics.mean(gpu_memory_values)

            # Set to None if no GPU data collected
            if not gpu_percent_values:
                metrics.gpu_percent = None
            if not gpu_memory_values:
                metrics.gpu_memory_mb = None


class MetricsCollector:
    """Singleton collector for all metrics across processing steps"""

    def __init__(self):
        self.steps: Dict[str, StepMetrics] = {}
        self.short_index: Optional[int] = None

    @contextmanager
    def track_step(self, step_name: str):
        """Context manager to track a processing step with automatic resource monitoring"""
        tracker = StepTracker(step_name, self)
        with tracker:
            yield tracker

    def log_tokens(self, step_name: str, usage_metadata: Any):
        """Log token usage from Gemini API response"""
        if step_name not in self.steps:
            self.steps[step_name] = StepMetrics(
                timestamp=datetime.utcnow().isoformat() + 'Z'
            )

        metrics = self.steps[step_name]

        # Extract token counts from usage_metadata
        if hasattr(usage_metadata, 'prompt_token_count'):
            metrics.tokens["prompt"] += usage_metadata.prompt_token_count
        if hasattr(usage_metadata, 'candidates_token_count'):
            metrics.tokens["completion"] += usage_metadata.candidates_token_count
        if hasattr(usage_metadata, 'total_token_count'):
            metrics.tokens["total"] += usage_metadata.total_token_count
        if hasattr(usage_metadata, 'cached_content_token_count'):
            metrics.tokens["cached"] += usage_metadata.cached_content_token_count or 0

    def log_characters(self, step_name: str, char_count: int):
        """Log character usage for TTS services"""
        if step_name not in self.steps:
            self.steps[step_name] = StepMetrics(
                timestamp=datetime.utcnow().isoformat() + 'Z'
            )

        self.steps[step_name].characters_used += char_count

    def calculate_totals(self) -> Dict[str, Any]:
        """Calculate aggregate totals across all steps"""
        totals = {
            "duration_seconds": 0.0,
            "total_tokens": 0,
            "total_characters": 0,
            "peak_memory_mb": 0.0,
            "peak_cpu_percent": 0.0,
            "peak_gpu_percent": None,
            "peak_gpu_memory_mb": None,
        }

        has_gpu_data = False

        for step_metrics in self.steps.values():
            totals["duration_seconds"] += step_metrics.duration_seconds
            totals["total_tokens"] += step_metrics.tokens.get("total", 0)
            totals["total_characters"] += step_metrics.characters_used

            if step_metrics.memory_mb["peak"] > 0:
                totals["peak_memory_mb"] = max(
                    totals["peak_memory_mb"],
                    step_metrics.memory_mb["peak"]
                )

            if step_metrics.cpu_percent["peak"] > 0:
                totals["peak_cpu_percent"] = max(
                    totals["peak_cpu_percent"],
                    step_metrics.cpu_percent["peak"]
                )

            if step_metrics.gpu_percent and step_metrics.gpu_percent["peak"] > 0:
                has_gpu_data = True
                if totals["peak_gpu_percent"] is None:
                    totals["peak_gpu_percent"] = 0.0
                totals["peak_gpu_percent"] = max(
                    totals["peak_gpu_percent"],
                    step_metrics.gpu_percent["peak"]
                )

            if step_metrics.gpu_memory_mb and step_metrics.gpu_memory_mb["peak"] > 0:
                if totals["peak_gpu_memory_mb"] is None:
                    totals["peak_gpu_memory_mb"] = 0.0
                totals["peak_gpu_memory_mb"] = max(
                    totals["peak_gpu_memory_mb"],
                    step_metrics.gpu_memory_mb["peak"]
                )

        return totals

    def write_report(self, filepath: str):
        """Write metrics report to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert to serializable dict
        report = {
            "short_index": self.short_index,
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "steps": {
                name: asdict(metrics)
                for name, metrics in self.steps.items()
            },
            "totals": self.calculate_totals()
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

    def print_report(self):
        """Print formatted metrics report to stdout"""
        totals = self.calculate_totals()

        # Header
        header = f"RESOURCE CONSUMPTION REPORT"
        if self.short_index is not None:
            header += f" - SHORT #{self.short_index}"

        print("\n" + "=" * 80)
        print(header.center(80))
        print("=" * 80 + "\n")

        # Per-step metrics
        for step_name, metrics in self.steps.items():
            print(f"Step: {step_name}")

            # Duration
            mins = int(metrics.duration_seconds // 60)
            secs = metrics.duration_seconds % 60
            duration_str = f"{metrics.duration_seconds:.1f}s"
            if mins > 0:
                duration_str += f" ({mins}m {secs:.0f}s)"
            print(f"  Duration:          {duration_str}")

            # Tokens
            if metrics.tokens["total"] > 0:
                print(f"  Tokens:            {metrics.tokens['total']:,} "
                      f"({metrics.tokens['prompt']:,} prompt + "
                      f"{metrics.tokens['completion']:,} completion)")
                if metrics.tokens["cached"] > 0:
                    print(f"                     ({metrics.tokens['cached']:,} cached)")

            # Characters
            if metrics.characters_used > 0:
                print(f"  Characters:        {metrics.characters_used:,}")

            # Memory
            if metrics.memory_mb["peak"] > 0:
                print(f"  Memory:            {metrics.memory_mb['peak']:.0f} MB peak "
                      f"({metrics.memory_mb['average']:.0f} MB avg)")

            # CPU
            if metrics.cpu_percent["peak"] > 0:
                print(f"  CPU:               {metrics.cpu_percent['peak']:.1f}% peak "
                      f"({metrics.cpu_percent['average']:.1f}% avg)")

            # GPU
            if metrics.gpu_percent and metrics.gpu_percent["peak"] > 0:
                print(f"  GPU:               {metrics.gpu_percent['peak']:.1f}% peak "
                      f"({metrics.gpu_percent['average']:.1f}% avg)")

            if metrics.gpu_memory_mb and metrics.gpu_memory_mb["peak"] > 0:
                print(f"  GPU Memory:        {metrics.gpu_memory_mb['peak']:.0f} MB peak "
                      f"({metrics.gpu_memory_mb['average']:.0f} MB avg)")

            print()  # Blank line between steps

        # Totals
        print("─" * 80)
        totals_header = "TOTALS"
        if self.short_index is not None:
            totals_header += f" FOR SHORT #{self.short_index}"
        print(totals_header + ":")

        mins = int(totals["duration_seconds"] // 60)
        secs = totals["duration_seconds"] % 60
        duration_str = f"{totals['duration_seconds']:.1f}s"
        if mins > 0:
            duration_str += f" ({mins}m {secs:.0f}s)"
        print(f"  Total Duration:    {duration_str}")

        if totals["total_tokens"] > 0:
            print(f"  Total Tokens:      {totals['total_tokens']:,}")

        if totals["total_characters"] > 0:
            print(f"  Total Characters:  {totals['total_characters']:,} (ElevenLabs)")

        if totals["peak_memory_mb"] > 0:
            print(f"  Peak Memory:       {totals['peak_memory_mb']:.0f} MB")

        if totals["peak_cpu_percent"] > 0:
            print(f"  Peak CPU:          {totals['peak_cpu_percent']:.1f}%")

        if totals["peak_gpu_percent"] is not None and totals["peak_gpu_percent"] > 0:
            print(f"  Peak GPU:          {totals['peak_gpu_percent']:.1f}%")

        if totals["peak_gpu_memory_mb"] is not None and totals["peak_gpu_memory_mb"] > 0:
            print(f"  Peak GPU Memory:   {totals['peak_gpu_memory_mb']:.0f} MB")

        print("─" * 80)
        print("=" * 80 + "\n")

    def reset(self):
        """Clear all metrics (call between shorts)"""
        self.steps = {}
        self.short_index = None

    def merge_from_file(self, filepath: str, prefix: str = "subprocess_"):
        """
        Merge metrics from a subprocess metrics file into current collector.

        Args:
            filepath: Path to subprocess metrics JSON file
            prefix: Prefix to add to subprocess step names to avoid conflicts
        """
        if not os.path.exists(filepath):
            print(f"[WARNING] Subprocess metrics file not found: {filepath}")
            return

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            subprocess_steps = data.get("steps", {})

            for step_name, step_data in subprocess_steps.items():
                # Prefix subprocess step names to avoid conflicts
                prefixed_name = f"{prefix}{step_name}"

                # Convert dict back to StepMetrics
                metrics = StepMetrics(
                    duration_seconds=step_data.get("duration_seconds", 0.0),
                    tokens=step_data.get("tokens", {"prompt": 0, "completion": 0, "total": 0, "cached": 0}),
                    characters_used=step_data.get("characters_used", 0),
                    memory_mb=step_data.get("memory_mb", {"peak": 0.0, "average": 0.0, "start": 0.0, "end": 0.0}),
                    cpu_percent=step_data.get("cpu_percent", {"peak": 0.0, "average": 0.0}),
                    gpu_percent=step_data.get("gpu_percent"),
                    gpu_memory_mb=step_data.get("gpu_memory_mb"),
                    timestamp=step_data.get("timestamp", "")
                )

                self.steps[prefixed_name] = metrics

            print(f"[INFO] Merged {len(subprocess_steps)} steps from subprocess metrics")

        except Exception as e:
            print(f"[ERROR] Failed to merge subprocess metrics: {e}")


# Global singleton instance
metrics_collector = MetricsCollector()
