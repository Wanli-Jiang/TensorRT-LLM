# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed private wrapper for one rank body and its placement report.

REAL product bodies publish only facts that they directly observe through
:func:`publish_product_identity_evidence`. Controller expectations are never
passed to that API and therefore cannot be copied into product-reported
backend or transport evidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import socket
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Callable, Literal, Mapping, cast

from .launchers import (
    ExpectedProductIdentity,
    LaunchKind,
    ObservedProductIdentity,
    RankBinding,
    RankLaunchPlan,
)

RANK_INPUT_SCHEMA_VERSION = 1
RANK_REPORT_SCHEMA_VERSION = 1
PRODUCT_EVIDENCE_PATH_ENVIRONMENT = "STAIRCASE_PRODUCT_EVIDENCE_PATH"

_MAX_INPUT_BYTES = 256 * 1024
_MAX_PRODUCT_EVIDENCE_BYTES = 64 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_GPU_ID = re.compile(r"[0-9]+\Z")
_BODY_KINDS = frozenset({"product", "synthetic"})
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_IDENTITY_VALUE = re.compile(r"[^\x00\r\n]{1,512}\Z")
_SENSITIVE_ENVIRONMENT = re.compile(
    r"(?:AUTH|COOKIE|CREDENTIAL|KEY|PASS|SECRET|TOKEN)", re.IGNORECASE
)
_PROHIBITED_PROGRAMS = frozenset(
    {
        "bash",
        "dash",
        "mpiexec",
        "mpirun",
        "sacct",
        "sbatch",
        "scancel",
        "sh",
        "squeue",
        "srun",
        "ssh",
        "zsh",
    }
)
_INPUT_KEYS = frozenset(
    {
        "schema_version",
        "plan_digest",
        "rank_command_digest",
        "launch_kind",
        "body_kind",
        "argv",
        "environment",
        "cwd",
        "report_directory",
        "world_size",
        "placements",
        "expected_product_identity",
    }
)
_PRODUCT_EVIDENCE_KEYS = frozenset(
    {
        "collective_backend",
        "transport",
    }
)


class RankWorkerError(RuntimeError):
    """Raised when a rank input, runtime placement, or report is unsafe."""


@dataclass(frozen=True, slots=True)
class RankInput:
    """Immutable controller-authored input shared by all planned ranks."""

    schema_version: int
    plan_digest: str
    rank_command_digest: str
    launch_kind: LaunchKind
    body_kind: Literal["product", "synthetic"]
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    cwd: Path
    report_directory: Path
    world_size: int
    placements: tuple[RankBinding, ...]
    expected_product_identity: ExpectedProductIdentity | None = None

    def __post_init__(self) -> None:
        if self.schema_version != RANK_INPUT_SCHEMA_VERSION:
            raise RankWorkerError("unsupported rank input schema version")
        if not _DIGEST.fullmatch(self.plan_digest) or not _DIGEST.fullmatch(
            self.rank_command_digest
        ):
            raise RankWorkerError("rank input digests must be lowercase SHA-256 values")
        if not isinstance(self.launch_kind, LaunchKind):
            raise RankWorkerError("rank input launch_kind must use LaunchKind")
        if self.body_kind not in _BODY_KINDS:
            raise RankWorkerError("rank input body_kind must be product or synthetic")
        expected_body = (
            "synthetic" if self.launch_kind is LaunchKind.SYNTHETIC_MULTI_NODE_RUNNER else "product"
        )
        if self.body_kind != expected_body:
            raise RankWorkerError("rank input body kind is incompatible with launch kind")
        _validate_argv(self.argv)
        _validate_environment(self.environment)
        _validate_existing_directory(self.cwd, "body cwd")
        _validate_existing_directory(self.report_directory, "report directory")
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int):
            raise RankWorkerError("rank input world_size must be an integer")
        if self.world_size < 1 or len(self.placements) != self.world_size:
            raise RankWorkerError("rank input must contain one placement per rank")
        if not all(isinstance(binding, RankBinding) for binding in self.placements):
            raise RankWorkerError("rank input placements must use RankBinding")
        if tuple(binding.rank for binding in self.placements) != tuple(range(self.world_size)):
            raise RankWorkerError("rank input placements must be in complete rank order")
        if self.launch_kind is LaunchKind.SLURM_MULTI_NODE_PRODUCT:
            if not isinstance(self.expected_product_identity, ExpectedProductIdentity):
                raise RankWorkerError(
                    "real multi-node rank input requires expected product identity"
                )
        elif self.expected_product_identity is not None:
            raise RankWorkerError(
                "only real multi-node rank input may carry expected product identity"
            )


BodyExecutor = Callable[[tuple[str, ...], Path, tuple[tuple[str, str], ...], Literal[False]], int]
IdentityProbe = Callable[[Path, RankBinding], Mapping[str, str]]


def publish_rank_input(
    plan: RankLaunchPlan,
    *,
    cwd: Path,
    report_directory: Path,
) -> tuple[RankInput, str]:
    """Create one exclusive rank input and its dedicated report directory.

    Args:
        plan: Frozen launcher plan containing the input pathname.
        cwd: Exact body working directory.
        report_directory: New shared directory for exclusive per-rank reports.

    Returns:
        The normalized input and SHA-256 digest of its exact serialized bytes.
    """
    if not isinstance(plan, RankLaunchPlan):
        raise RankWorkerError("plan must use RankLaunchPlan")
    canonical_cwd = _canonical_existing_directory(cwd, "body cwd")
    canonical_report_directory = _create_report_directory(report_directory)
    rank_input = RankInput(
        schema_version=RANK_INPUT_SCHEMA_VERSION,
        plan_digest=plan.plan_digest,
        rank_command_digest=plan.rank_command_digest,
        launch_kind=plan.kind,
        body_kind="product" if plan.product_rank_body else "synthetic",
        argv=plan.rank_command.argv,
        environment=tuple(sorted(plan.rank_command.environment)),
        cwd=canonical_cwd,
        report_directory=canonical_report_directory,
        world_size=plan.world_size,
        placements=plan.placements,
        expected_product_identity=plan.expected_product_identity,
    )
    payload = _rank_input_payload(rank_input)
    serialized = (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    input_path = plan.rank_input_path
    _validate_new_file_path(input_path, "rank input")
    try:
        with input_path.open("xb") as output:
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())
        input_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        _fsync_directory(input_path.parent)
    except OSError as error:
        raise RankWorkerError(f"could not publish exclusive rank input: {error}") from error
    return rank_input, hashlib.sha256(serialized).hexdigest()


def load_rank_input(path: Path) -> tuple[RankInput, str]:
    """Load one canonical, read-only rank input with strict JSON fields."""
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise RankWorkerError("rank input must be an absolute regular file")
    try:
        canonical = path.resolve(strict=True)
        metadata = path.stat()
    except OSError as error:
        raise RankWorkerError(f"could not inspect rank input: {error}") from error
    if canonical != path or not stat.S_ISREG(metadata.st_mode):
        raise RankWorkerError("rank input must be a canonical regular file")
    if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise RankWorkerError("rank input must be read-only")
    if metadata.st_size > _MAX_INPUT_BYTES:
        raise RankWorkerError("rank input exceeds the fixed size bound")
    try:
        serialized = path.read_bytes()
        value = json.loads(serialized)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RankWorkerError(f"could not parse rank input: {error}") from error
    if not isinstance(value, dict):
        raise RankWorkerError("rank input must be a JSON object")
    payload = cast(dict[str, object], value)
    _require_exact_keys(payload, _INPUT_KEYS, "rank input")
    rank_input = _parse_rank_input(payload)
    return rank_input, hashlib.sha256(serialized).hexdigest()


def publish_product_identity_evidence(
    *,
    collective_backend: str,
    transport: str,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Publish exact product-observed collective identity once and atomically.

    The rank wrapper injects the destination through
    ``STAIRCASE_PRODUCT_EVIDENCE_PATH``. The product body must obtain the
    actual backend and transport from the runtime under test and pass those
    values explicitly. This function never reads controller expectations.

    Args:
        collective_backend: Actual collective backend observed by the body.
        transport: Actual collective transport observed by the body.
        environment: Process environment override used by deterministic tests.

    Returns:
        Canonical path of the published read-only JSON evidence file.

    Raises:
        RankWorkerError: If values or the wrapper-owned destination are unsafe,
            already exist, are symlinks, or cannot be durably published.
    """
    observed = {
        "collective_backend": collective_backend,
        "transport": transport,
    }
    for name, value in observed.items():
        if not isinstance(value, str) or _IDENTITY_VALUE.fullmatch(value) is None:
            raise RankWorkerError(
                f"observed product {name} must be a bounded non-empty single-line string"
            )
    runtime_environment = os.environ if environment is None else environment
    destination_value = runtime_environment.get(PRODUCT_EVIDENCE_PATH_ENVIRONMENT)
    if not isinstance(destination_value, str) or not destination_value:
        raise RankWorkerError(
            f"product body lacks wrapper-owned {PRODUCT_EVIDENCE_PATH_ENVIRONMENT}"
        )
    destination = Path(destination_value)
    pending = destination.with_name(f".{destination.name}.pending")
    _validate_new_file_path(destination, "product identity evidence")
    _validate_new_file_path(pending, "pending product identity evidence")
    serialized = (json.dumps(observed, separators=(",", ":"), sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with pending.open("xb") as output:
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())
        pending.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.link(pending, destination)
        pending.unlink()
        _fsync_directory(destination.parent)
    except OSError as error:
        raise RankWorkerError(
            f"could not exclusively publish product identity evidence: {error}"
        ) from error
    return destination.resolve(strict=True)


def execute_rank(
    input_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
    hostname_provider: Callable[[], str] = socket.gethostname,
    executor: BodyExecutor | None = None,
    identity_probe: IdentityProbe | None = None,
) -> int:
    """Validate one runtime binding, execute its exact body, and report success."""
    rank_input, input_digest = load_rank_input(input_path)
    runtime_environment = dict(os.environ if environment is None else environment)
    binding = _runtime_binding(rank_input, runtime_environment, hostname_provider())
    pending_path = rank_input.report_directory / f"rank-{binding.rank}.pending"
    report_path = rank_input.report_directory / f"rank-{binding.rank}.json"
    if report_path.exists() or report_path.is_symlink():
        raise RankWorkerError("could not reserve exclusive rank report: report already exists")
    try:
        pending = pending_path.open("xb")
    except OSError as error:
        raise RankWorkerError(f"could not reserve exclusive rank report: {error}") from error

    product_evidence_path = (
        rank_input.report_directory / f"rank-{binding.rank}-product-evidence.json"
    )
    product_environment = (
        {PRODUCT_EVIDENCE_PATH_ENVIRONMENT: str(product_evidence_path)}
        if rank_input.expected_product_identity is not None
        else {}
    )
    body_environment = tuple(
        sorted(
            {
                **dict(rank_input.environment),
                **product_environment,
                "CUDA_VISIBLE_DEVICES": str(binding.gpu_id),
                "LOCAL_RANK": str(binding.local_rank),
                "RANK": str(binding.rank),
                "WORLD_SIZE": str(rank_input.world_size),
            }.items()
        )
    )
    body_executor = executor or _execute_body
    try:
        returncode = body_executor(
            rank_input.argv,
            rank_input.cwd,
            body_environment,
            False,
        )
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            raise RankWorkerError("rank body executor returned an invalid status")
        if returncode != 0:
            return returncode if returncode > 0 else 128 - returncode
        product_identity = None
        if rank_input.expected_product_identity is not None:
            product_identity = _observe_product_identity(
                rank_input,
                binding,
                product_evidence_path,
                identity_probe or _probe_runtime_identity,
            )
        report = {
            "schema_version": RANK_REPORT_SCHEMA_VERSION,
            "input_digest": input_digest,
            "plan_digest": rank_input.plan_digest,
            "rank_command_digest": rank_input.rank_command_digest,
            "launch_kind": rank_input.launch_kind.value,
            "body_kind": rank_input.body_kind,
            "rank": binding.rank,
            "hostname": binding.hostname,
            "local_rank": binding.local_rank,
            "gpu_id": binding.gpu_id,
            "product_identity": (
                product_identity.to_dict() if product_identity is not None else None
            ),
        }
        serialized = (json.dumps(report, separators=(",", ":"), sort_keys=True) + "\n").encode(
            "utf-8"
        )
        pending.write(serialized)
        pending.flush()
        os.fsync(pending.fileno())
    except OSError as error:
        raise RankWorkerError(f"rank body or report failed: {error}") from error
    finally:
        pending.close()
    try:
        os.link(pending_path, report_path)
        pending_path.unlink()
        _fsync_directory(rank_input.report_directory)
    except OSError as error:
        raise RankWorkerError(f"could not publish exclusive rank report: {error}") from error
    return 0


def _observe_product_identity(
    rank_input: RankInput,
    binding: RankBinding,
    evidence_path: Path,
    probe: IdentityProbe,
) -> ObservedProductIdentity:
    expected = rank_input.expected_product_identity
    if expected is None:
        raise RankWorkerError("product identity observation requires an expectation")
    observed = dict(probe(rank_input.cwd, binding))
    required_probe_keys = {
        "python_tensorrt_llm_path",
        "compute_capability",
        "cuda_device_uuid",
        "cuda_device_model",
    }
    if set(observed) != required_probe_keys:
        raise RankWorkerError("runtime identity probe keys differ from the fixed schema")
    body, body_digest = _load_product_body_evidence(evidence_path)
    try:
        identity = ObservedProductIdentity(
            repository_commit=expected.repository_commit,
            python_tensorrt_llm_path=observed["python_tensorrt_llm_path"],
            native_build_identity=expected.native_build_identity,
            image_identity=expected.image_identity,
            compute_capability=observed["compute_capability"],
            collective_backend=body["collective_backend"],
            transport=body["transport"],
            cuda_device_uuid=observed["cuda_device_uuid"],
            cuda_device_model=observed["cuda_device_model"],
            body_evidence_digest=body_digest,
        )
    except (KeyError, ValueError) as error:
        raise RankWorkerError(f"invalid observed product identity: {error}") from error
    if identity.expected != expected:
        raise RankWorkerError(
            "independently observed product identity differs from rank input expectation"
        )
    return identity


def _probe_runtime_identity(cwd: Path, binding: RankBinding) -> Mapping[str, str]:
    """Observe import and CUDA identity without trusting product-body output."""
    _ = cwd, binding

    spec = importlib.util.find_spec("tensorrt_llm")
    if spec is None or spec.origin is None:
        raise RankWorkerError("could not resolve the imported tensorrt_llm package")
    python_path = Path(spec.origin).resolve(strict=True)

    try:
        import torch

        if not torch.cuda.is_available():
            raise RankWorkerError("CUDA is unavailable in a product rank")
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        device_uuid = getattr(properties, "uuid", None)
        if device_uuid is None:
            raise RankWorkerError("CUDA runtime did not expose a device UUID")
        device_model = properties.name
        compute_capability = f"{properties.major}.{properties.minor}"
    except (ImportError, OSError, RuntimeError) as error:
        if isinstance(error, RankWorkerError):
            raise
        raise RankWorkerError(f"could not probe CUDA identity: {error}") from error
    if not isinstance(device_model, str) or not isinstance(device_uuid, (str, bytes)):
        raise RankWorkerError("CUDA identity probe returned unsupported values")
    uuid_text = (
        device_uuid.decode("ascii", errors="strict")
        if isinstance(device_uuid, bytes)
        else device_uuid
    )
    return {
        "python_tensorrt_llm_path": str(python_path),
        "compute_capability": compute_capability,
        "cuda_device_uuid": uuid_text,
        "cuda_device_model": device_model,
    }


def _load_product_body_evidence(path: Path) -> tuple[dict[str, str], str]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RankWorkerError(
            "product body did not publish its required regular identity evidence file"
        )
    metadata = path.stat()
    if path.resolve(strict=True) != path:
        raise RankWorkerError("product body identity evidence path must be canonical")
    if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise RankWorkerError("product body identity evidence must be published read-only")
    if metadata.st_size > _MAX_PRODUCT_EVIDENCE_BYTES:
        raise RankWorkerError("product body identity evidence exceeds the fixed size bound")
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RankWorkerError(f"could not parse product body identity evidence: {error}") from error
    if not isinstance(value, dict):
        raise RankWorkerError("product body identity evidence must be a JSON object")
    document = cast(dict[str, object], value)
    _require_exact_keys(document, _PRODUCT_EVIDENCE_KEYS, "product body identity evidence")
    normalized = {key: _required_string(document, key) for key in sorted(_PRODUCT_EVIDENCE_KEYS)}
    return normalized, hashlib.sha256(payload).hexdigest()


def _execute_body(
    argv: tuple[str, ...],
    cwd: Path,
    environment: tuple[tuple[str, str], ...],
    shell: Literal[False],
) -> int:
    if shell is not False:
        raise RankWorkerError("rank body execution must set shell=False")
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=dict(environment),
        shell=False,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode


def _runtime_binding(
    rank_input: RankInput,
    environment: Mapping[str, str],
    hostname: str,
) -> RankBinding:
    if rank_input.launch_kind is LaunchKind.SINGLE_PROCESS:
        rank = _runtime_integer(environment, "RANK")
        local_rank = _runtime_integer(environment, "LOCAL_RANK")
        world_size = _runtime_integer(environment, "WORLD_SIZE")
    else:
        rank = _runtime_integer(environment, "SLURM_PROCID")
        local_rank = _runtime_integer(environment, "SLURM_LOCALID")
        world_size = _runtime_integer(environment, "SLURM_NTASKS")
    if world_size != rank_input.world_size:
        raise RankWorkerError("runtime world size differs from rank input")
    if rank >= rank_input.world_size:
        raise RankWorkerError("runtime rank is outside the planned world")
    gpu_value = environment.get("CUDA_VISIBLE_DEVICES")
    if gpu_value is None or not _GPU_ID.fullmatch(gpu_value):
        raise RankWorkerError("runtime must expose exactly one numeric CUDA device")
    observed = RankBinding(
        rank=rank,
        hostname=hostname,
        local_rank=local_rank,
        gpu_id=int(gpu_value),
    )
    if observed != rank_input.placements[rank]:
        raise RankWorkerError("runtime rank/node/local-rank/GPU differs from planned binding")
    return observed


def _runtime_integer(environment: Mapping[str, str], name: str) -> int:
    value = environment.get(name)
    if value is None or not value.isdigit():
        raise RankWorkerError(f"runtime {name} must be a non-negative integer")
    return int(value)


def _rank_input_payload(rank_input: RankInput) -> dict[str, object]:
    return {
        "schema_version": rank_input.schema_version,
        "plan_digest": rank_input.plan_digest,
        "rank_command_digest": rank_input.rank_command_digest,
        "launch_kind": rank_input.launch_kind.value,
        "body_kind": rank_input.body_kind,
        "argv": list(rank_input.argv),
        "environment": [list(pair) for pair in rank_input.environment],
        "cwd": str(rank_input.cwd),
        "report_directory": str(rank_input.report_directory),
        "world_size": rank_input.world_size,
        "placements": [
            {
                "rank": binding.rank,
                "hostname": binding.hostname,
                "local_rank": binding.local_rank,
                "gpu_id": binding.gpu_id,
            }
            for binding in rank_input.placements
        ],
        "expected_product_identity": (
            rank_input.expected_product_identity.to_dict()
            if rank_input.expected_product_identity is not None
            else None
        ),
    }


def _parse_rank_input(payload: Mapping[str, object]) -> RankInput:
    placements_value = payload["placements"]
    if not isinstance(placements_value, list):
        raise RankWorkerError("rank input placements must be a list")
    placements = tuple(_parse_binding(value) for value in placements_value)
    try:
        launch_kind = LaunchKind(payload["launch_kind"])
    except (TypeError, ValueError) as error:
        raise RankWorkerError("rank input launch_kind is invalid") from error
    body_kind = payload["body_kind"]
    if body_kind not in _BODY_KINDS:
        raise RankWorkerError("rank input body_kind is invalid")
    return RankInput(
        schema_version=_required_integer(payload, "schema_version"),
        plan_digest=_required_string(payload, "plan_digest"),
        rank_command_digest=_required_string(payload, "rank_command_digest"),
        launch_kind=launch_kind,
        body_kind=cast(Literal["product", "synthetic"], body_kind),
        argv=_parse_argv(payload["argv"]),
        environment=_parse_environment(payload["environment"]),
        cwd=Path(_required_string(payload, "cwd")),
        report_directory=Path(_required_string(payload, "report_directory")),
        world_size=_required_integer(payload, "world_size"),
        placements=placements,
        expected_product_identity=_parse_expected_product_identity(
            payload["expected_product_identity"]
        ),
    )


def _parse_expected_product_identity(value: object) -> ExpectedProductIdentity | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RankWorkerError("expected_product_identity must be an object or null")
    identity = cast(dict[str, object], value)
    _require_exact_keys(
        identity,
        frozenset(
            {
                "repository_commit",
                "python_tensorrt_llm_path",
                "native_build_identity",
                "image_identity",
                "compute_capability",
                "collective_backend",
                "transport",
            }
        ),
        "expected product identity",
    )
    try:
        return ExpectedProductIdentity(
            repository_commit=_required_string(identity, "repository_commit"),
            python_tensorrt_llm_path=_required_string(identity, "python_tensorrt_llm_path"),
            native_build_identity=_required_string(identity, "native_build_identity"),
            image_identity=_required_string(identity, "image_identity"),
            compute_capability=_required_string(identity, "compute_capability"),
            collective_backend=_required_string(identity, "collective_backend"),
            transport=_required_string(identity, "transport"),
        )
    except ValueError as error:
        raise RankWorkerError(f"invalid expected product identity: {error}") from error


def _parse_binding(value: object) -> RankBinding:
    if not isinstance(value, dict):
        raise RankWorkerError("rank input placement must be an object")
    binding = cast(dict[str, object], value)
    _require_exact_keys(
        binding, frozenset({"rank", "hostname", "local_rank", "gpu_id"}), "placement"
    )
    return RankBinding(
        rank=_required_integer(binding, "rank"),
        hostname=_required_string(binding, "hostname"),
        local_rank=_required_integer(binding, "local_rank"),
        gpu_id=_required_integer(binding, "gpu_id"),
    )


def _parse_argv(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RankWorkerError("rank input argv must be a list")
    argv = tuple(value)
    _validate_argv(argv)
    return cast(tuple[str, ...], argv)


def _validate_argv(argv: tuple[object, ...]) -> None:
    if not argv or not all(isinstance(argument, str) and argument for argument in argv):
        raise RankWorkerError("rank body argv must contain non-empty strings")
    if any("\x00" in cast(str, argument) or "\n" in cast(str, argument) for argument in argv):
        raise RankWorkerError("rank body argv cannot contain NUL or newline characters")
    program = PurePath(cast(str, argv[0])).name.lower()
    if program in _PROHIBITED_PROGRAMS:
        raise RankWorkerError("rank body cannot invoke a shell, scheduler, SSH, or rank launcher")


def _parse_environment(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise RankWorkerError("rank input environment must be a list")
    pairs: list[tuple[str, str]] = []
    for pair in value:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(item, str) for item in pair)
        ):
            raise RankWorkerError("rank input environment entries must be string pairs")
        pairs.append((pair[0], pair[1]))
    environment = tuple(pairs)
    _validate_environment(environment)
    return environment


def _validate_environment(environment: tuple[tuple[str, str], ...]) -> None:
    names = [name for name, _value in environment]
    if len(set(names)) != len(names):
        raise RankWorkerError("rank body environment contains duplicate names")
    for name, value in environment:
        if not _ENVIRONMENT_NAME.fullmatch(name) or any(
            character in value for character in "\x00\n\r"
        ):
            raise RankWorkerError("rank body environment contains an unsafe name or value")
        if _SENSITIVE_ENVIRONMENT.search(name):
            raise RankWorkerError("rank body environment contains a credential-like name")
    reserved = {"CUDA_VISIBLE_DEVICES", "LOCAL_RANK", "RANK", "WORLD_SIZE"}
    if set(names) & reserved:
        raise RankWorkerError("rank body environment contains wrapper-owned placement variables")
    if dict(environment).get("TRTLLM_MODELING_V2") != "require":
        raise RankWorkerError("rank body must set TRTLLM_MODELING_V2=require")


def _required_integer(payload: Mapping[str, object], name: str) -> int:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RankWorkerError(f"rank input {name} must be an integer")
    return value


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload[name]
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise RankWorkerError(f"rank input {name} must be a non-empty single-line string")
    return value


def _require_exact_keys(
    payload: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise RankWorkerError(
            f"{label} keys differ from schema: missing={missing!r}, unknown={unknown!r}"
        )


def _canonical_existing_directory(path: Path, label: str) -> Path:
    _validate_existing_directory(path, label)
    return path.resolve(strict=True)


def _validate_existing_directory(path: Path, label: str) -> None:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_dir()
    ):
        raise RankWorkerError(f"{label} must be an absolute regular directory")
    if path.resolve(strict=True) != path:
        raise RankWorkerError(f"{label} must be canonical")


def _create_report_directory(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or path.exists() or path.is_symlink():
        raise RankWorkerError("report directory must be a new absolute path")
    parent = path.parent
    _validate_existing_directory(parent, "report directory parent")
    try:
        path.mkdir(mode=stat.S_IRWXU)
        _fsync_directory(parent)
    except OSError as error:
        raise RankWorkerError(f"could not create report directory: {error}") from error
    return path.resolve(strict=True)


def _validate_new_file_path(path: Path, label: str) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or path.exists() or path.is_symlink():
        raise RankWorkerError(f"{label} must be a new absolute path")
    _validate_existing_directory(path.parent, f"{label} parent")
    if path.resolve(strict=False) != path:
        raise RankWorkerError(f"{label} must be canonical")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "BodyExecutor",
    "PRODUCT_EVIDENCE_PATH_ENVIRONMENT",
    "RANK_INPUT_SCHEMA_VERSION",
    "RANK_REPORT_SCHEMA_VERSION",
    "RankInput",
    "RankWorkerError",
    "execute_rank",
    "load_rank_input",
    "publish_product_identity_evidence",
    "publish_rank_input",
]
