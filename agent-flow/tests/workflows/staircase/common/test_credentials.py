# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for write-once agent credential transport and leak prevention."""

from __future__ import annotations

import json
import os
import stat
from hashlib import sha256
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.credentials import (
    CREDENTIAL_MOUNT_PATH,
    MAX_RUN_CREDENTIAL_REVOCATIONS,
    CredentialBinding,
    CredentialDescriptor,
    CredentialError,
    CredentialHandle,
    CredentialProvision,
    CredentialRevocationCause,
    CredentialRevocationReceipt,
    CredentialState,
    agent_credential_environment,
    descriptor_from_public_dict,
    ensure_file_has_no_credential_values,
    ensure_no_credential_values,
    load_worker_credentials,
    locate_credential_provision,
    materialize_credential_provision,
    prepare_credential_provision,
    redact_credential_values,
    revoke_generation_fenced_provisions,
    revoke_run_credential_provisions,
    revoke_terminal_attempt_credentials,
    validate_unique_credential_handles,
)

_SECRET = "sk-test-this-value-must-never-leak"


def _binding(*, backend_kind: str = "codex", attempt_id: str = "attempt-1") -> CredentialBinding:
    return CredentialBinding(
        run_id="run-1",
        item_id="item-1",
        attempt_id=attempt_id,
        task_digest="a" * 64,
        generation=1,
        backend_kind=backend_kind,  # type: ignore[arg-type]
    )


def test_no_credential_provision_is_explicit_and_does_not_copy_ambient(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    provision = prepare_credential_provision(
        workspace=workspace,
        binding=_binding(),
        ambient_environment={
            "ANTHROPIC_API_KEY": "wrong-backend",
            "SLURM_JWT": "scheduler-secret",
            "PATH": "/ambient/bin",
        },
    )

    assert provision.descriptor.state is CredentialState.NONE
    assert provision.descriptor.credential_names == ()
    assert provision.bundle_path is None
    assert provision.mounts == ()
    assert "wrong-backend" not in json.dumps(provision.descriptor.to_public_dict())


def test_bundle_is_exclusive_0600_fixed_mount_and_descriptor_is_secret_free(
    tmp_path: Path,
) -> None:
    workspace = tmp_path.resolve()
    provision = prepare_credential_provision(
        workspace=workspace,
        binding=_binding(),
        ambient_environment={
            "OPENAI_API_KEY": _SECRET,
            "ANTHROPIC_API_KEY": "ignored",
            "SLURM_JWT": "ignored-scheduler-secret",
        },
    )

    assert provision.bundle_path is not None
    assert provision.bundle_path.is_relative_to(workspace / "controller-secrets")
    assert stat.S_IMODE(provision.bundle_path.stat().st_mode) == 0o600
    assert len(provision.mounts) == 1
    assert provision.mounts[0].source == provision.bundle_path
    assert provision.mounts[0].target == CREDENTIAL_MOUNT_PATH
    assert provision.mounts[0].read_only
    public = json.dumps(provision.descriptor.to_public_dict(), sort_keys=True)
    assert _SECRET not in public
    assert provision.descriptor.credential_names == ("OPENAI_API_KEY",)
    assert load_worker_credentials(
        provision.descriptor,
        expected_binding=_binding(),
        bundle_path=provision.bundle_path,
    ) == {"OPENAI_API_KEY": _SECRET}

    adopted = prepare_credential_provision(
        workspace=workspace,
        binding=_binding(),
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    assert adopted == provision
    with pytest.raises(CredentialError, match="does not match"):
        prepare_credential_provision(
            workspace=workspace,
            binding=_binding(),
            ambient_environment={"OPENAI_API_KEY": "changed-secret"},
        )


def test_descriptor_digest_is_public_and_independent_of_secret_value(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    one = prepare_credential_provision(
        workspace=first.resolve(),
        binding=_binding(),
        ambient_environment={"OPENAI_API_KEY": "first-secret"},
    )
    two = prepare_credential_provision(
        workspace=second.resolve(),
        binding=_binding(),
        ambient_environment={"OPENAI_API_KEY": "second-secret"},
    )
    assert one.descriptor == two.descriptor
    assert one.descriptor.descriptor_digest == two.descriptor.descriptor_digest


def test_bundle_rejects_wrong_identity_permissions_extra_keys_and_symlink(
    tmp_path: Path,
) -> None:
    workspace = tmp_path.resolve()
    provision = prepare_credential_provision(
        workspace=workspace,
        binding=_binding(),
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    assert provision.bundle_path is not None

    with pytest.raises(CredentialError, match="identity/backend mismatch"):
        load_worker_credentials(
            provision.descriptor,
            expected_binding=_binding(attempt_id="attempt-2"),
            bundle_path=provision.bundle_path,
        )

    provision.bundle_path.chmod(0o644)
    with pytest.raises(CredentialError, match="mode 0600"):
        load_worker_credentials(
            provision.descriptor,
            expected_binding=_binding(),
            bundle_path=provision.bundle_path,
        )
    provision.bundle_path.chmod(0o600)

    raw = json.loads(provision.bundle_path.read_text(encoding="utf-8"))
    raw["unexpected"] = "field"
    provision.bundle_path.write_text(json.dumps(raw), encoding="utf-8")
    provision.bundle_path.chmod(0o600)
    with pytest.raises(CredentialError, match="keys invalid"):
        load_worker_credentials(
            provision.descriptor,
            expected_binding=_binding(),
            bundle_path=provision.bundle_path,
        )

    link = workspace / "credential-link.json"
    link.symlink_to(provision.bundle_path)
    with pytest.raises(CredentialError, match="regular file"):
        load_worker_credentials(
            provision.descriptor,
            expected_binding=_binding(),
            bundle_path=link,
        )


def test_descriptor_rejects_extra_keys_names_and_backend_mismatch() -> None:
    descriptor = CredentialDescriptor.create(
        _binding(),
        CredentialState.BUNDLE,
        ("OPENAI_API_KEY",),
        handle=CredentialHandle("test-broker", "attempt-1", 2_147_483_647),
    )
    public = descriptor.to_public_dict()
    assert descriptor_from_public_dict(public) == descriptor

    with pytest.raises(CredentialError, match="keys invalid"):
        descriptor_from_public_dict({**public, "secret": _SECRET})
    with pytest.raises(CredentialError, match="non-allowlisted"):
        CredentialDescriptor.create(_binding(), CredentialState.BUNDLE, ("ANTHROPIC_API_KEY",))


def test_broker_handle_cannot_be_reused_across_attempt_bindings() -> None:
    handle = CredentialHandle("test-broker", "one-lease", 2_147_483_647)
    first = CredentialDescriptor.create(
        _binding(attempt_id="attempt-1"),
        CredentialState.BUNDLE,
        ("OPENAI_API_KEY",),
        handle=handle,
    )
    second = CredentialDescriptor.create(
        _binding(attempt_id="attempt-2"),
        CredentialState.BUNDLE,
        ("OPENAI_API_KEY",),
        handle=handle,
    )

    with pytest.raises(CredentialError, match="reused one handle"):
        validate_unique_credential_handles((first, second))


def test_expired_broker_handle_cannot_materialize(tmp_path: Path) -> None:
    descriptor = CredentialDescriptor.create(
        _binding(),
        CredentialState.BUNDLE,
        ("OPENAI_API_KEY",),
        handle=CredentialHandle("test-broker", "expired", 1),
    )

    with pytest.raises(CredentialError, match="has expired"):
        materialize_credential_provision(
            workspace=tmp_path.resolve(),
            descriptor=descriptor,
            credential_values={"OPENAI_API_KEY": _SECRET},
        )


def test_agent_scope_clears_ambient_credentials_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic")

    with agent_credential_environment("codex", {"OPENAI_API_KEY": _SECRET}):
        assert os.environ["OPENAI_API_KEY"] == _SECRET
        assert "ANTHROPIC_API_KEY" not in os.environ

    assert os.environ["OPENAI_API_KEY"] == "ambient-openai"
    assert os.environ["ANTHROPIC_API_KEY"] == "ambient-anthropic"

    with agent_credential_environment("codex", {}):
        assert "OPENAI_API_KEY" not in os.environ
        assert "ANTHROPIC_API_KEY" not in os.environ


def test_response_evidence_and_error_leak_scanning(tmp_path: Path) -> None:
    credentials = {"OPENAI_API_KEY": _SECRET}
    with pytest.raises(CredentialError, match="agent response"):
        ensure_no_credential_values(
            f"response leaked {_SECRET}", credentials, label="agent response"
        )

    evidence = tmp_path / "evidence.bin"
    split = len(_SECRET) // 2
    with evidence.open("wb") as stream:
        stream.write(b"x" * (65_536 - split))
        stream.write(_SECRET.encode("utf-8"))
    with pytest.raises(CredentialError, match="agent evidence"):
        ensure_file_has_no_credential_values(evidence, credentials, label="agent evidence")
    assert _SECRET not in redact_credential_values(f"backend failed with {_SECRET}", credentials)


def test_terminal_revocation_removes_secret_and_is_receipt_idempotent(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace = tmp_path.resolve()
    binding = _binding()
    provision = prepare_credential_provision(
        workspace=workspace,
        binding=binding,
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    assert provision.bundle_path is not None

    receipt = revoke_terminal_attempt_credentials(
        workspace=workspace,
        expected_binding=binding,
        provision=provision,
    )

    assert not provision.bundle_path.exists()
    assert receipt.receipt_path.is_file()
    assert stat.S_IMODE(receipt.receipt_path.stat().st_mode) == 0o400
    receipt_text = receipt.receipt_path.read_text(encoding="utf-8")
    secret_hash = sha256(_SECRET.encode("utf-8")).hexdigest()
    observable = "\n".join(
        (
            receipt_text,
            repr(receipt),
            repr(receipt.to_public_dict()),
            caplog.text,
        )
    )
    assert _SECRET not in observable
    assert secret_hash not in observable
    assert receipt.to_public_dict()["credential_descriptor"] == (
        provision.descriptor.to_public_dict()
    )

    located = locate_credential_provision(
        workspace=workspace,
        descriptor=descriptor_from_public_dict(provision.descriptor.to_public_dict()),
    )
    assert located == provision
    assert not provision.bundle_path.exists()

    repeated = revoke_terminal_attempt_credentials(
        workspace=workspace,
        expected_binding=binding,
        provision=located,
    )
    assert repeated == receipt


def test_missing_bundle_requires_an_exact_prior_revocation_receipt(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    binding = _binding()
    provision = prepare_credential_provision(
        workspace=workspace,
        binding=binding,
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    assert provision.bundle_path is not None
    provision.bundle_path.unlink()

    with pytest.raises(CredentialError, match="missing without an exact") as error:
        revoke_terminal_attempt_credentials(
            workspace=workspace,
            expected_binding=binding,
            provision=provision,
        )
    assert _SECRET not in repr(error.value)
    assert sha256(_SECRET.encode("utf-8")).hexdigest() not in repr(error.value)


def test_revocation_rejects_binding_and_canonical_path_swaps(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    first = prepare_credential_provision(
        workspace=workspace,
        binding=_binding(),
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    second_binding = _binding(attempt_id="attempt-2")
    second = prepare_credential_provision(
        workspace=workspace,
        binding=second_binding,
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    assert first.bundle_path is not None
    assert second.bundle_path is not None

    with pytest.raises(CredentialError, match="binding/descriptor mismatch"):
        revoke_terminal_attempt_credentials(
            workspace=workspace,
            expected_binding=second_binding,
            provision=first,
        )
    swapped = CredentialProvision(first.descriptor, second.bundle_path)
    with pytest.raises(CredentialError, match="not canonical"):
        revoke_terminal_attempt_credentials(
            workspace=workspace,
            expected_binding=_binding(),
            provision=swapped,
        )
    assert first.bundle_path.exists()
    assert second.bundle_path.exists()


def test_revocation_rejects_extra_content_permissions_hardlinks_and_symlinks(
    tmp_path: Path,
) -> None:
    for attempt_id, mutation, match in (
        ("extra", "extra", "keys invalid"),
        ("mode", "mode", "mode 0600"),
        ("hardlink", "hardlink", "exactly one hard link"),
        ("symlink", "symlink", "regular file"),
    ):
        workspace = (tmp_path / attempt_id).resolve()
        workspace.mkdir()
        binding = _binding(attempt_id=attempt_id)
        provision = prepare_credential_provision(
            workspace=workspace,
            binding=binding,
            ambient_environment={"OPENAI_API_KEY": _SECRET},
        )
        assert provision.bundle_path is not None
        if mutation == "extra":
            payload = json.loads(provision.bundle_path.read_text(encoding="utf-8"))
            payload["unexpected"] = "metadata"
            provision.bundle_path.write_text(json.dumps(payload), encoding="utf-8")
            provision.bundle_path.chmod(0o600)
        elif mutation == "mode":
            provision.bundle_path.chmod(0o640)
        elif mutation == "hardlink":
            os.link(provision.bundle_path, workspace / "extra-hardlink")
        else:
            backing = workspace / "bundle-backing"
            provision.bundle_path.rename(backing)
            provision.bundle_path.symlink_to(backing)

        with pytest.raises(CredentialError, match=match) as error:
            revoke_terminal_attempt_credentials(
                workspace=workspace,
                expected_binding=binding,
                provision=provision,
            )
        assert _SECRET not in repr(error.value)


def test_revocation_rejects_non_controller_owned_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path.resolve()
    binding = _binding()
    provision = prepare_credential_provision(
        workspace=workspace,
        binding=binding,
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    effective_user = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: effective_user + 1)

    with pytest.raises(CredentialError, match="owned by the controller user") as error:
        revoke_terminal_attempt_credentials(
            workspace=workspace,
            expected_binding=binding,
            provision=provision,
        )
    assert _SECRET not in repr(error.value)


def test_revocation_receipt_is_strict_private_and_immutable(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    binding = _binding()
    provision = prepare_credential_provision(
        workspace=workspace,
        binding=binding,
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    receipt = revoke_terminal_attempt_credentials(
        workspace=workspace,
        expected_binding=binding,
        provision=provision,
    )

    receipt.receipt_path.chmod(0o600)
    with pytest.raises(CredentialError, match="mode 0400"):
        revoke_terminal_attempt_credentials(
            workspace=workspace,
            expected_binding=binding,
            provision=provision,
        )
    payload = json.loads(receipt.receipt_path.read_text(encoding="utf-8"))
    payload["unexpected"] = "metadata"
    receipt.receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt.receipt_path.chmod(0o400)
    with pytest.raises(CredentialError, match="keys invalid"):
        revoke_terminal_attempt_credentials(
            workspace=workspace,
            expected_binding=binding,
            provision=provision,
        )


def test_revocation_rejects_receipt_hardlink_and_retries_after_unlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path.resolve()
    binding = _binding()
    provision = prepare_credential_provision(
        workspace=workspace,
        binding=binding,
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    assert provision.bundle_path is not None
    real_unlink = os.unlink
    failed = False

    def fail_bundle_unlink_once(
        path: str | bytes | Path,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal failed
        if path == provision.bundle_path.name and dir_fd is not None and not failed:
            failed = True
            raise OSError("injected unlink failure")
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", fail_bundle_unlink_once)
    with pytest.raises(OSError, match="injected unlink failure") as error:
        revoke_terminal_attempt_credentials(
            workspace=workspace,
            expected_binding=binding,
            provision=provision,
        )
    assert _SECRET not in repr(error.value)
    assert provision.bundle_path.exists()

    monkeypatch.setattr(os, "unlink", real_unlink)
    receipt = revoke_terminal_attempt_credentials(
        workspace=workspace,
        expected_binding=binding,
        provision=provision,
    )
    assert not provision.bundle_path.exists()
    hardlink = workspace / "receipt-hardlink"
    os.link(receipt.receipt_path, hardlink)
    with pytest.raises(CredentialError, match="hard-link count"):
        revoke_terminal_attempt_credentials(
            workspace=workspace,
            expected_binding=binding,
            provision=provision,
        )


def test_run_revocation_is_bounded_explicit_and_prevalidates_all_targets(
    tmp_path: Path,
) -> None:
    workspace = tmp_path.resolve()
    first = prepare_credential_provision(
        workspace=workspace,
        binding=_binding(attempt_id="attempt-1"),
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    second = prepare_credential_provision(
        workspace=workspace,
        binding=_binding(attempt_id="attempt-2"),
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    other_binding = CredentialBinding(
        run_id="other-run",
        item_id="item-1",
        attempt_id="attempt-3",
        task_digest="a" * 64,
        generation=1,
        backend_kind="codex",
    )
    other = prepare_credential_provision(
        workspace=workspace,
        binding=other_binding,
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    assert first.bundle_path is not None
    assert second.bundle_path is not None
    unrelated = first.bundle_path.parent / "unrelated-controller-file"
    unrelated.write_text("keep", encoding="utf-8")

    with pytest.raises(CredentialError, match="another run"):
        revoke_run_credential_provisions(
            workspace=workspace,
            run_id="run-1",
            provisions=(first, other),
        )
    assert first.bundle_path.exists()

    receipts = revoke_run_credential_provisions(
        workspace=workspace,
        run_id="run-1",
        provisions=(first, second),
    )
    assert len(receipts) == 2
    assert not first.bundle_path.exists()
    assert not second.bundle_path.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert other.bundle_path is not None and other.bundle_path.exists()

    no_credentials = prepare_credential_provision(
        workspace=workspace,
        binding=_binding(attempt_id="no-credentials"),
        ambient_environment={},
    )
    assert (
        locate_credential_provision(
            workspace=workspace,
            descriptor=no_credentials.descriptor,
        )
        == no_credentials
    )
    with pytest.raises(CredentialError, match="bounded target limit"):
        revoke_run_credential_provisions(
            workspace=workspace,
            run_id="run-1",
            provisions=tuple(no_credentials for _ in range(MAX_RUN_CREDENTIAL_REVOCATIONS + 1)),
        )


def test_generation_fence_revokes_only_exact_non_adopted_handles_idempotently(
    tmp_path: Path,
) -> None:
    workspace = tmp_path.resolve()
    adopted = prepare_credential_provision(
        workspace=workspace,
        binding=_binding(attempt_id="adopted"),
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    fenced = prepare_credential_provision(
        workspace=workspace,
        binding=_binding(attempt_id="fenced"),
        ambient_environment={"OPENAI_API_KEY": _SECRET},
    )
    calls: list[tuple[CredentialBinding, CredentialRevocationCause]] = []

    class Broker:
        def revoke(
            self,
            *,
            workspace: Path,
            expected_binding: CredentialBinding,
            descriptor: CredentialDescriptor,
            cause: CredentialRevocationCause,
        ) -> CredentialRevocationReceipt:
            calls.append((expected_binding, cause))
            return revoke_terminal_attempt_credentials(
                workspace=workspace,
                expected_binding=expected_binding,
                provision=locate_credential_provision(
                    workspace=workspace,
                    descriptor=descriptor,
                ),
                cause=cause,
            )

    receipts = revoke_generation_fenced_provisions(
        broker=Broker(),  # type: ignore[arg-type]
        workspace=workspace,
        current_generation=2,
        descriptors=(adopted.descriptor, fenced.descriptor),
        adopted_bindings=(adopted.descriptor.binding,),
    )

    assert len(receipts) == 1
    assert calls == [(fenced.descriptor.binding, CredentialRevocationCause.GENERATION_FENCE)]
    assert adopted.bundle_path is not None and adopted.bundle_path.exists()
    assert fenced.bundle_path is not None and not fenced.bundle_path.exists()

    repeated = revoke_generation_fenced_provisions(
        broker=Broker(),  # type: ignore[arg-type]
        workspace=workspace,
        current_generation=2,
        descriptors=(adopted.descriptor, fenced.descriptor),
        adopted_bindings=(adopted.descriptor.binding,),
    )
    assert repeated == receipts
