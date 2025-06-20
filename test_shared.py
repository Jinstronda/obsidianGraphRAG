#!/usr/bin/env python3
"""
Shared test classes and configurations
Prevents circular imports between test modules
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TestResult:
    """Container for test results"""
    phase: str
    test_name: str
    baseline_score: float
    enhanced_score: float
    improvement_pct: float
    execution_time: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class TestConfiguration:
    """Test configuration parameters"""
    data_subset_size: int
    validation_split: float
    enable_baseline_comparison: bool
    enable_performance_benchmarks: bool
    gpu_memory_limit: int
    max_test_duration: int  # seconds 