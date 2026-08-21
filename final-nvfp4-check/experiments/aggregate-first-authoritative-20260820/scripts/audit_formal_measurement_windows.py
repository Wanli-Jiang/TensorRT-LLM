#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audit formal SA-Bench envelopes using benchmark launch markers.

SA-Bench's result ``date`` is written when the result is saved, not when the
benchmark begins.  The formal launch marker in ``benchmark.out`` is therefore
the authoritative start of the conservative measurement envelope; the result
date is its end.  The JSON duration must fit inside that envelope.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

RESULT_GLOB = "logs/sa-bench_*/results_concurrency_*_gpus_*.json"
SERVER_GLOB = "logs/*_agg_w*.out"
BENCHMARK_LOG = "logs/benchmark.out"
FORMAL_MARKER = re.compile(r"^Running benchmark with concurrency: (\d+)\s*$")
LOCAL_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$")
HEARTBEAT_TIMESTAMP = re.compile(r"^\[([^]]+)\].*state=stopped")
SERVER_TIMESTAMP_PATTERNS = (
    re.compile(r"\[(\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2})\]"),
    re.compile(r"(?:^|\s)(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[,\s])"),
)
FORBIDDEN_PATTERNS = {
    "flashinfer_jit": re.compile(r"flashinfer\.jit|cubin_loader\.py", re.IGNORECASE),
    "deepgemm_jit_or_tune": re.compile(
        r"deepgemm.{0,100}(?:jit|compil|tun)|(?:jit|compil|tun).{0,100}deepgemm",
        re.IGNORECASE,
    ),
    "low_m_jit_or_tune": re.compile(
        r"low[-_ ]?m.{0,100}(?:jit|compil|tun)|(?:jit|compil|tun).{0,100}low[-_ ]?m",
        re.IGNORECASE,
    ),
    "cuda_graph_setup": re.compile(r"cuda graph (?:capture|warmup)", re.IGNORECASE),
    "generic_autotune": re.compile(r"auto[-_ ]?tun(?:e|er|ing)", re.IGNORECASE),
    "torch_or_triton_compile": re.compile(
        r"(?:torchinductor|triton).{0,100}(?:jit|compil)", re.IGNORECASE
    ),
}


@dataclass(frozen=True)
class _FormalWindow:
    concurrency: int
    result: Path
    start: datetime
    end: datetime
    measured_duration_seconds: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _parse_server_timestamp(line: str) -> datetime | None:
    for index, pattern in enumerate(SERVER_TIMESTAMP_PATTERNS):
        match = pattern.search(line)
        if match is None:
            continue
        fmt = "%m/%d/%Y-%H:%M:%S" if index == 0 else "%Y-%m-%d %H:%M:%S"
        return datetime.strptime(match.group(1), fmt)
    return None


def _load_formal_starts(benchmark_log: Path) -> dict[int, list[datetime]]:
    starts: dict[int, list[datetime]] = {}
    lines = benchmark_log.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        marker = FORMAL_MARKER.match(line)
        if marker is None:
            continue
        concurrency = int(marker.group(1))
        timestamp = None
        for following_line in lines[index + 1 : index + 4]:
            match = LOCAL_TIMESTAMP.match(following_line)
            if match is not None:
                timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                break
        if timestamp is not None:
            starts.setdefault(concurrency, []).append(timestamp)
    return starts


def _load_windows(job_dir: Path) -> tuple[list[_FormalWindow], list[str]]:
    failures: list[str] = []
    benchmark_log = job_dir / BENCHMARK_LOG
    if not benchmark_log.is_file():
        return [], [f"missing benchmark log: {benchmark_log}"]
    starts = _load_formal_starts(benchmark_log)
    windows: list[_FormalWindow] = []
    for path in sorted(job_dir.glob(RESULT_GLOB)):
        with path.open(encoding="utf-8") as result_file:
            data = json.load(result_file)
        concurrency = int(data["max_concurrency"])
        result_end = datetime.strptime(data["date"], "%Y%m%d-%H%M%S")
        candidates = [start for start in starts.get(concurrency, []) if start <= result_end]
        if not candidates:
            failures.append(
                f"no formal launch marker precedes result for concurrency {concurrency}"
            )
            continue
        start = max(candidates)
        duration = float(data["duration"])
        envelope_seconds = (result_end - start).total_seconds()
        if envelope_seconds + 2.0 < duration:
            failures.append(
                f"formal envelope for concurrency {concurrency} is {envelope_seconds:.3f}s, "
                f"shorter than measured duration {duration:.3f}s"
            )
        windows.append(
            _FormalWindow(
                concurrency=concurrency,
                result=path,
                start=start,
                end=result_end,
                measured_duration_seconds=duration,
            )
        )
    return windows, failures


def _load_heartbeat_stop(job_dir: Path) -> tuple[datetime | None, list[str]]:
    failures: list[str] = []
    state_path = job_dir / "loading-heartbeat/state"
    control_path = job_dir / "loading-heartbeat/control.log"
    if not state_path.is_file() or state_path.read_text(encoding="utf-8").strip() != "stopped":
        failures.append("loading heartbeat state is not stopped")
    if not control_path.is_file():
        failures.append("loading heartbeat control log is missing")
        return None, failures
    stop = None
    for line in control_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = HEARTBEAT_TIMESTAMP.match(line)
        if match is None:
            continue
        aware = datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
        stop = aware.astimezone().replace(tzinfo=None)
    if stop is None:
        failures.append("loading heartbeat stop timestamp is missing")
    return stop, failures


def _main() -> int:
    args = _parse_args()
    job_dir = args.job_dir.resolve()
    windows, failures = _load_windows(job_dir)
    server_logs = sorted(job_dir.glob(SERVER_GLOB))
    heartbeat_stop, heartbeat_failures = _load_heartbeat_stop(job_dir)
    failures.extend(heartbeat_failures)
    events: list[dict[str, int | str | list[str]]] = []

    if not windows:
        failures.append("no valid result JSON windows found")
    if not server_logs:
        failures.append("no aggregate server logs found")
    if heartbeat_stop is not None and windows:
        earliest_start = min(window.start for window in windows)
        if heartbeat_stop >= earliest_start:
            failures.append(
                "loading heartbeat did not stop before the earliest formal launch"
            )

    for log_path in server_logs:
        with log_path.open(encoding="utf-8", errors="replace") as log_file:
            for line_number, line in enumerate(log_file, start=1):
                matched_kinds = [
                    kind for kind, pattern in FORBIDDEN_PATTERNS.items() if pattern.search(line)
                ]
                if not matched_kinds:
                    continue
                timestamp = _parse_server_timestamp(line)
                if timestamp is None:
                    continue
                for window in windows:
                    if window.start <= timestamp <= window.end:
                        events.append(
                            {
                                "concurrency": window.concurrency,
                                "timestamp": timestamp.isoformat(),
                                "kinds": matched_kinds,
                                "log": str(log_path),
                                "line": line_number,
                                "text": line.strip()[:500],
                            }
                        )
    if events:
        failures.append(f"{len(events)} forbidden startup/JIT events overlap formal windows")

    report = {
        "status": "pass" if not failures else "fail",
        "job_dir": str(job_dir),
        "window_source": "benchmark.out formal launch marker through result save date",
        "heartbeat_stop": heartbeat_stop.isoformat() if heartbeat_stop is not None else None,
        "windows": [
            {
                "concurrency": window.concurrency,
                "result": str(window.result),
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
                "envelope_seconds": (window.end - window.start).total_seconds(),
                "measured_duration_seconds": window.measured_duration_seconds,
            }
            for window in windows
        ],
        "server_logs": [str(path) for path in server_logs],
        "forbidden_events": events,
        "failures": failures,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())
