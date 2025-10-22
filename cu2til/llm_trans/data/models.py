from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class RetryRecord:
    retry_index: int
    start_time: str
    end_time: str
    duration_ms: float
    success: bool
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    generation_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class RoundRecord:
    round: int
    retry_limit: int
    retry_records: List[RetryRecord] = field(default_factory=list)
    status: Optional[str] = None

    def add_retry(self, record: RetryRecord) -> None:
        self.retry_records.append(record)

    def to_dict(
        self,
        *,
        include_retry_records: bool = False,
        only_successful_retry: bool = False,
    ) -> Dict[str, Any]:
        data = {
            "round": self.round,
            "retry_limit": self.retry_limit,
            "retry_count": len(self.retry_records),
        }
        if include_retry_records:
            records = self.retry_records
            if only_successful_retry:
                success_records = [record for record in records if record.success]
                if success_records:
                    records = success_records[:1]
            data["retry_records"] = [record.to_dict() for record in records]
        if self.status:
            data["status"] = self.status
        return data


@dataclass
class TestRoundRecord:
    round: int
    success: bool
    execution_mode: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    waiting_ms: Optional[float] = None
    queue_ms: Optional[float] = None
    pending_ms: Optional[float] = None
    total_server_ms: Optional[float] = None
    gpu_id: Optional[str | int] = None
    status: Optional[str] = None
    exit_code: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class AttemptTimingStats:
    case_type: str
    case_name: str
    attempt_number: int
    start_time: str
    llm_rounds: List[RoundRecord] = field(default_factory=list)
    test_rounds: List[TestRoundRecord] = field(default_factory=list)
    total_llm_time_ms: float = 0.0
    llm_retry_time_ms: float = 0.0
    llm_retry_wait_time_ms: float = 0.0
    total_test_time_ms: float = 0.0
    end_time: Optional[str] = None
    wall_clock_duration_ms: Optional[float] = None
    effective_duration_ms: Optional[float] = None
    retry_overhead_ms: Optional[float] = None
    other_overhead_ms: Optional[float] = None
    success: Optional[bool] = None
    final_round: Optional[int] = None
    timestamp: Optional[str] = None
    model_name: Optional[str] = None
    actual_model_used: Optional[str] = None
    testset: Optional[str] = None
    use_nvgpu: Optional[bool] = None
    concurrency: Optional[int] = None
    gpu_server: Optional[str] = None
    max_attempts: Optional[int] = None
    attempt_policy: Optional[str] = None
    ms_format: Optional[str] = None
    error: Optional[str] = None
    work_dir: Optional[str] = None

    def add_llm_round(self, record: RoundRecord) -> None:
        self.llm_rounds.append(record)

    def add_test_round(self, record: TestRoundRecord) -> None:
        self.test_rounds.append(record)

    def add_llm_time(self, value: float) -> None:
        self.total_llm_time_ms += value

    def add_llm_retry_time(self, value: float) -> None:
        self.llm_retry_time_ms += value

    def add_llm_retry_wait(self, value: float) -> None:
        self.llm_retry_wait_time_ms += value

    def add_test_time(self, value: float) -> None:
        self.total_test_time_ms += value

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "case_type": self.case_type,
            "case_name": self.case_name,
            "attempt_number": self.attempt_number,
            "start_time": self.start_time,
            "llm_rounds": [
                round_record.to_dict(
                    include_retry_records=True,
                    only_successful_retry=True,
                )
                for round_record in self.llm_rounds
            ],
            "test_rounds": [round_record.to_dict() for round_record in self.test_rounds],
            "total_llm_time_ms": round(self.total_llm_time_ms, 3),
            "llm_retry_time_ms": round(self.llm_retry_time_ms, 3),
            "llm_retry_wait_time_ms": round(self.llm_retry_wait_time_ms, 3),
            "total_test_time_ms": round(self.total_test_time_ms, 3),
        }
        optional_fields = {
            "end_time": self.end_time,
            "wall_clock_duration_ms": self.wall_clock_duration_ms,
            "effective_duration_ms": self.effective_duration_ms,
            "retry_overhead_ms": self.retry_overhead_ms,
            "other_overhead_ms": self.other_overhead_ms,
            "success": self.success,
            "final_round": self.final_round,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "actual_model_used": self.actual_model_used,
            "testset": self.testset,
            "use_nvgpu": self.use_nvgpu,
            "concurrency": self.concurrency,
            "gpu_server": self.gpu_server,
            "max_attempts": self.max_attempts,
            "attempt_policy": self.attempt_policy,
            "ms_format": self.ms_format,
            "error": self.error,
            "work_dir": self.work_dir,
        }
        for key, value in optional_fields.items():
            if value is not None:
                if isinstance(value, float):
                    data[key] = round(value, 3)
                else:
                    data[key] = value
        return data


@dataclass
class AttemptResult:
    attempt_number: int
    success: bool
    rounds: Optional[int]
    timing_stats: AttemptTimingStats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "success": self.success,
            "rounds": self.rounds,
            "timing_stats": self.timing_stats.to_dict(),
        }


@dataclass
class CaseResult:
    case_type: str
    case_name: str
    success: bool
    rounds: Optional[int]
    attempt_details: List[AttemptResult]
    successful_attempt: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "case_type": self.case_type,
            "case_name": self.case_name,
            "success": self.success,
            "rounds": self.rounds,
            "attempt_details": [attempt.to_dict() for attempt in self.attempt_details],
            "successful_attempt": self.successful_attempt,
        }
        if self.error:
            data["error"] = self.error
        return data


@dataclass
class BatchSummary:
    total_cases: int
    successful_cases: int
    failed_cases: int
    success_rate: float
    case_types_total: int
    case_types_fully_passed: int
    rounds_distribution: Dict[int, int]
    avg_rounds: float
    max_attempts: int
    attempt_policy: str
    ms_format: str
    timestamp: str
    completed_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "RetryRecord",
    "RoundRecord",
    "TestRoundRecord",
    "AttemptTimingStats",
    "AttemptResult",
    "CaseResult",
    "BatchSummary",
]
