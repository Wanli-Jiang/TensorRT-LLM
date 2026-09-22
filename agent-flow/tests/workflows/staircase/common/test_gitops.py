# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller-owned Staircase Git operations."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common import gitops
from agent_flow.workflows.staircase.common.isolation import digest_metadata_free_tree


def _git(repo: Path, *arguments: str) -> str:
    """Run Git in a temporary test repository."""
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create an isolated repository with a named default branch."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _git(repo, "config", "user.email", "controller@example.com")
    _git(repo, "config", "user.name", "Staircase Controller")
    (repo / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "review.py").write_text("STATE = 'base'\n", encoding="utf-8")
    _git(repo, "add", "--", "model.py", "review.py")
    _git(repo, "commit", "-q", "-s", "-m", "initial")
    return repo


def _controller(repository: Path, tmp_path: Path) -> gitops.ControllerGitOps:
    """Construct a controller whose lock is outside the repository."""
    return gitops.ControllerGitOps(repository, tmp_path / "controller.lock", "run-1:generation-1")


def test_inspect_supports_branch_and_detached_head(repository: Path, tmp_path: Path) -> None:
    controller = _controller(repository, tmp_path)
    inspection = controller.inspect()
    assert inspection.branch == "main"
    assert inspection.clean

    _git(repository, "checkout", "--detach", "HEAD")
    assert controller.inspect().branch is None


def test_transaction_lock_serializes_mutation(repository: Path, tmp_path: Path) -> None:
    controller = _controller(repository, tmp_path)
    contender = gitops.ControllerFileLock(tmp_path / "controller.lock", "other-controller")
    with controller.transaction() as transaction:
        assert transaction.lock.held
        with pytest.raises(gitops.ControllerLockError, match="already held"):
            contender.acquire()
    contender.acquire()
    contender.release()
    with pytest.raises(gitops.ControllerLockError, match="requires the held"):
        transaction.stage_paths(repository, ["model.py"])


def test_create_stage_and_signed_off_commit_are_explicit(repository: Path, tmp_path: Path) -> None:
    controller = _controller(repository, tmp_path)
    base = controller.inspect().head
    worktree = tmp_path / "worktrees" / "item-001"
    with controller.transaction() as transaction:
        created = transaction.create_candidate_worktree(
            worktree,
            branch="staircase/run-1/item-001",
            base_commit=base,
        )
        assert created == worktree.resolve()
        (worktree / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
        assert transaction.stage_paths(worktree, ["model.py"]) == ("model.py",)
        candidate = transaction.commit_signed_off(
            worktree,
            message="staircase: integrate item-001",
            expected_paths=["model.py"],
        )

    message = _git(worktree, "log", "-1", "--format=%B")
    assert "Signed-off-by: Staircase Controller <controller@example.com>" in message
    verified = controller.verify_candidate(
        worktree,
        base_commit=base,
        candidate_commit=candidate,
        allowed_paths=["model.py"],
    )
    assert verified.changed_paths == ("model.py",)
    assert len(verified.patch_sha256) == 64
    assert len(verified.tree_hash) == 40


def test_metadata_free_overlay_is_bound_only_in_controller_worktree(
    repository: Path, tmp_path: Path
) -> None:
    controller = _controller(repository, tmp_path)
    base = controller.inspect().head
    overlay = tmp_path / "workspace" / "candidates" / "item-001" / "attempt-001"
    controller_candidate = (
        tmp_path / "workspace" / "controller-candidates" / "item-001" / "attempt-001"
    )
    with controller.transaction() as transaction:
        transaction.create_candidate_overlay(overlay, base_commit=base)
    assert (overlay / "model.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (overlay / ".git").exists()

    (overlay / "model.py").write_text("VALUE = 7\n", encoding="utf-8")
    with controller.transaction() as transaction:
        binding = transaction.bind_candidate_overlay(
            overlay,
            repository=controller_candidate,
            branch="staircase/overlay/item-001",
            base_commit=base,
            expected_paths=("model.py",),
            expected_overlay_sha256=digest_metadata_free_tree(overlay),
            message="staircase: bind item-001 candidate",
        )

    assert binding.overlay == overlay
    assert binding.repository == controller_candidate
    assert binding.changed_paths == ("model.py",)
    assert _git(controller_candidate, "rev-parse", "HEAD") == binding.candidate_commit
    assert "Signed-off-by:" in _git(controller_candidate, "log", "-1", "--format=%B")
    assert not (overlay / ".git").exists()


def test_overlay_binding_rejects_change_after_worker_scan(repository: Path, tmp_path: Path) -> None:
    controller = _controller(repository, tmp_path)
    base = controller.inspect().head
    overlay = tmp_path / "workspace" / "candidates" / "raced" / "attempt-001"
    destination = tmp_path / "workspace" / "controller-candidates" / "raced"
    with controller.transaction() as transaction:
        transaction.create_candidate_overlay(overlay, base_commit=base)
    (overlay / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
    scanned_digest = digest_metadata_free_tree(overlay)
    (overlay / "model.py").write_text("VALUE = 'post-scan replacement'\n", encoding="utf-8")

    with controller.transaction() as transaction:
        with pytest.raises(gitops.GitOpsError, match="scanned content manifest"):
            transaction.bind_candidate_overlay(
                overlay,
                repository=destination,
                branch="staircase/overlay/raced",
                base_commit=base,
                expected_paths=("model.py",),
                expected_overlay_sha256=scanned_digest,
                message="must not commit",
            )

    assert not destination.exists()
    assert not destination.with_name(f".{destination.name}.overlay-receipt.json").exists()


def test_overlay_binding_rejects_git_metadata_and_undeclared_paths(
    repository: Path, tmp_path: Path
) -> None:
    controller = _controller(repository, tmp_path)
    base = controller.inspect().head
    overlay = tmp_path / "workspace" / "candidates" / "item-002" / "attempt-001"
    with controller.transaction() as transaction:
        transaction.create_candidate_overlay(overlay, base_commit=base)

    (overlay / "unexpected.py").write_text("BAD = True\n", encoding="utf-8")
    with controller.transaction() as transaction:
        with pytest.raises(gitops.GitOpsError, match="strict claim"):
            transaction.bind_candidate_overlay(
                overlay,
                repository=tmp_path / "workspace" / "controller-candidates" / "item-002",
                branch="staircase/overlay/item-002",
                base_commit=base,
                expected_paths=(),
                expected_overlay_sha256=digest_metadata_free_tree(overlay),
                message="unused",
            )

    (overlay / "unexpected.py").unlink()
    (overlay / ".git").write_text("gitdir: /shared/admin\n", encoding="utf-8")
    with controller.transaction() as transaction:
        with pytest.raises(gitops.GitOpsError, match="Git metadata"):
            transaction.bind_candidate_overlay(
                overlay,
                repository=tmp_path / "workspace" / "controller-candidates" / "item-003",
                branch="staircase/overlay/item-003",
                base_commit=base,
                expected_paths=(),
                expected_overlay_sha256="0" * 64,
                message="unused",
            )


def test_overlay_rejects_nested_and_future_git_metadata_links(
    repository: Path, tmp_path: Path
) -> None:
    controller = _controller(repository, tmp_path)
    base = controller.inspect().head
    nested = tmp_path / "workspace" / "candidates" / "nested" / "attempt-001"
    linked = tmp_path / "workspace" / "candidates" / "linked" / "attempt-001"
    with controller.transaction() as transaction:
        transaction.create_candidate_overlay(nested, base_commit=base)
        transaction.create_candidate_overlay(linked, base_commit=base)
    (nested / "subdirectory" / ".git").mkdir(parents=True)
    (nested / "subdirectory" / ".git" / "config").write_text("unsafe\n", encoding="utf-8")
    (linked / "future-git").symlink_to(".git")

    with controller.transaction() as transaction:
        with pytest.raises(gitops.GitOpsError, match="Git metadata"):
            transaction.bind_candidate_overlay(
                nested,
                repository=tmp_path / "workspace" / "controller-candidates" / "nested",
                branch="staircase/overlay/nested",
                base_commit=base,
                expected_paths=(),
                expected_overlay_sha256="0" * 64,
                message="unused",
            )
        with pytest.raises(gitops.GitOpsError, match="would expose Git metadata"):
            transaction.bind_candidate_overlay(
                linked,
                repository=tmp_path / "workspace" / "controller-candidates" / "linked",
                branch="staircase/overlay/linked",
                base_commit=base,
                expected_paths=("future-git",),
                expected_overlay_sha256="0" * 64,
                message="unused",
            )


def test_overlay_binding_detects_ignored_untracked_file(repository: Path, tmp_path: Path) -> None:
    (repository / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
    _git(repository, "add", "--", ".gitignore")
    _git(repository, "commit", "-q", "-s", "-m", "add ignore rule")
    controller = _controller(repository, tmp_path)
    base = controller.inspect().head
    overlay = tmp_path / "workspace" / "candidates" / "ignored" / "attempt-001"
    with controller.transaction() as transaction:
        transaction.create_candidate_overlay(overlay, base_commit=base)
    (overlay / "worker.ignored").write_text("must not disappear\n", encoding="utf-8")

    with controller.transaction() as transaction:
        with pytest.raises(gitops.GitOpsError, match="strict claim"):
            transaction.bind_candidate_overlay(
                overlay,
                repository=tmp_path / "workspace" / "controller-candidates" / "ignored",
                branch="staircase/overlay/ignored",
                base_commit=base,
                expected_paths=(),
                expected_overlay_sha256=digest_metadata_free_tree(overlay),
                message="unused",
            )


@pytest.mark.parametrize("change", ["deletion", "mode"])
def test_overlay_binding_preserves_deletion_and_mode_changes(
    repository: Path, tmp_path: Path, change: str
) -> None:
    controller = _controller(repository, tmp_path)
    base = controller.inspect().head
    overlay = tmp_path / "workspace" / "candidates" / change / "attempt-001"
    candidate = tmp_path / "workspace" / "controller-candidates" / change
    with controller.transaction() as transaction:
        transaction.create_candidate_overlay(overlay, base_commit=base)
    if change == "deletion":
        (overlay / "model.py").unlink()
    else:
        (overlay / "model.py").chmod(0o755)

    with controller.transaction() as transaction:
        binding = transaction.bind_candidate_overlay(
            overlay,
            repository=candidate,
            branch=f"staircase/overlay/{change}",
            base_commit=base,
            expected_paths=("model.py",),
            expected_overlay_sha256=digest_metadata_free_tree(overlay),
            message=f"staircase: bind {change}",
        )

    if change == "deletion":
        assert (
            _git(candidate, "diff", "--name-status", base, binding.candidate_commit)
            == "D\tmodel.py"
        )
    else:
        assert "mode change 100644 => 100755 model.py" in _git(
            candidate, "diff", "--summary", base, binding.candidate_commit
        )


def test_staging_rejects_unclaimed_changes(repository: Path, tmp_path: Path) -> None:
    controller = _controller(repository, tmp_path)
    (repository / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repository / "unexpected.py").write_text("BAD = True\n", encoding="utf-8")
    with controller.transaction() as transaction:
        with pytest.raises(gitops.GitOpsError, match="unexpected changed paths"):
            transaction.stage_paths(repository, ["model.py"])


def test_candidate_verification_rejects_path_and_hash_mismatch(
    repository: Path, tmp_path: Path
) -> None:
    controller = _controller(repository, tmp_path)
    base = controller.inspect().head
    worktree = tmp_path / "candidate"
    with controller.transaction() as transaction:
        transaction.create_candidate_worktree(
            worktree, branch="staircase/candidate", base_commit=base
        )
        (worktree / "review.py").write_text("STATE = 'candidate'\n", encoding="utf-8")
        transaction.stage_paths(worktree, ["review.py"])
        candidate = transaction.commit_signed_off(
            worktree, message="candidate", expected_paths=["review.py"]
        )

    with pytest.raises(gitops.GitOpsError, match="outside its claim"):
        controller.verify_candidate(
            worktree,
            base_commit=base,
            candidate_commit=candidate,
            allowed_paths=["model.py"],
        )
    with pytest.raises(gitops.GitOpsError, match="patch hash mismatch"):
        controller.verify_candidate(
            worktree,
            base_commit=base,
            candidate_commit=candidate,
            allowed_paths=["review.py"],
            expected_patch_sha256="0" * 64,
        )


def test_fast_forward_requires_expected_clean_head(repository: Path, tmp_path: Path) -> None:
    controller = _controller(repository, tmp_path)
    base = controller.inspect().head
    worktree = tmp_path / "ff"
    with controller.transaction() as transaction:
        transaction.create_candidate_worktree(worktree, branch="staircase/ff", base_commit=base)
        (worktree / "model.py").write_text("VALUE = 3\n", encoding="utf-8")
        transaction.stage_paths(worktree, ["model.py"])
        candidate = transaction.commit_signed_off(
            worktree, message="fast-forward candidate", expected_paths=["model.py"]
        )
        integrated = transaction.integrate_fast_forward(candidate_ref=candidate, expected_head=base)
    assert integrated == candidate
    assert (repository / "model.py").read_text(encoding="utf-8") == "VALUE = 3\n"


def test_cherry_pick_like_integration_aborts_on_conflict(repository: Path, tmp_path: Path) -> None:
    controller = _controller(repository, tmp_path)
    base = controller.inspect().head
    worktree = tmp_path / "conflict"
    with controller.transaction() as transaction:
        transaction.create_candidate_worktree(
            worktree, branch="staircase/conflict", base_commit=base
        )
        (worktree / "model.py").write_text("VALUE = 20\n", encoding="utf-8")
        transaction.stage_paths(worktree, ["model.py"])
        candidate = transaction.commit_signed_off(
            worktree, message="candidate conflict", expected_paths=["model.py"]
        )

    (repository / "model.py").write_text("VALUE = 30\n", encoding="utf-8")
    _git(repository, "add", "--", "model.py")
    _git(repository, "commit", "-q", "-s", "-m", "integration divergence")
    expected_head = controller.inspect().head
    with controller.transaction() as transaction:
        with pytest.raises(gitops.GitConflictError, match="without resolution"):
            transaction.integrate_commit(candidate_commit=candidate, expected_head=expected_head)
    assert controller.inspect().clean
    assert controller.inspect().head == expected_head
    assert (repository / "model.py").read_text(encoding="utf-8") == "VALUE = 30\n"


def test_cherry_pick_like_integration_signs_off(repository: Path, tmp_path: Path) -> None:
    controller = _controller(repository, tmp_path)
    base = controller.inspect().head
    worktree = tmp_path / "candidate"
    with controller.transaction() as transaction:
        transaction.create_candidate_worktree(
            worktree, branch="staircase/review-change", base_commit=base
        )
        (worktree / "review.py").write_text("STATE = 'approved'\n", encoding="utf-8")
        transaction.stage_paths(worktree, ["review.py"])
        candidate = transaction.commit_signed_off(
            worktree, message="candidate review change", expected_paths=["review.py"]
        )

    (repository / "model.py").write_text("VALUE = 4\n", encoding="utf-8")
    _git(repository, "add", "--", "model.py")
    _git(repository, "commit", "-q", "-s", "-m", "independent integration change")
    expected_head = controller.inspect().head
    with controller.transaction() as transaction:
        integrated = transaction.integrate_commit(
            candidate_commit=candidate, expected_head=expected_head
        )
    assert integrated != expected_head
    assert (repository / "review.py").read_text(encoding="utf-8") == "STATE = 'approved'\n"
    assert "Signed-off-by:" in _git(repository, "log", "-1", "--format=%B")


def test_linear_patch_sequence_recovery_rejects_extra_commit(
    repository: Path, tmp_path: Path
) -> None:
    controller = _controller(repository, tmp_path)
    base = controller.inspect().head
    candidate_repository = tmp_path / "sequence-candidate"
    with controller.transaction() as transaction:
        transaction.create_candidate_worktree(
            candidate_repository,
            branch="staircase/sequence-candidate",
            base_commit=base,
        )
        (candidate_repository / "review.py").write_text("STATE = 'sequence'\n", encoding="utf-8")
        transaction.stage_paths(candidate_repository, ("review.py",))
        candidate = transaction.commit_signed_off(
            candidate_repository,
            message="sequence candidate",
            expected_paths=("review.py",),
        )
        integrated = transaction.integrate_commit(candidate_commit=candidate, expected_head=base)
    candidate_verification = controller.verify_candidate(
        candidate_repository,
        base_commit=base,
        candidate_commit=candidate,
        allowed_paths=("review.py",),
    )
    expectation = gitops.IntegratedPatchExpectation(
        candidate_verification.patch_sha256,
        candidate_verification.changed_paths,
    )

    verified = controller.verify_linear_patch_sequence(
        previous_head=base,
        observed_head=integrated,
        expectations=(expectation,),
    )
    assert verified.commits == (integrated,)
    assert verified.final_tree_hash == _git(repository, "rev-parse", f"{integrated}^{{tree}}")
    assert len(verified.evidence_sha256) == 64

    (repository / "model.py").write_text("VALUE = 99\n", encoding="utf-8")
    _git(repository, "add", "--", "model.py")
    _git(repository, "commit", "-q", "-s", "-m", "unexpected extra commit")
    with pytest.raises(gitops.GitOpsError, match="missing, extra, reordered, or substituted"):
        controller.verify_linear_patch_sequence(
            previous_head=base,
            observed_head=controller.inspect().head,
            expectations=(expectation,),
        )


def test_every_git_subprocess_uses_argv_explicit_c_and_no_shell(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[list[str], dict[str, object]]] = []
    real_run = subprocess.run

    def recording_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.append((command, kwargs))
        return real_run(command, **kwargs)

    monkeypatch.setattr(gitops.subprocess, "run", recording_run)
    _controller(repository, tmp_path).inspect()
    assert seen
    assert all(command[:2] == ["git", "-C"] for command, _ in seen)
    assert all("shell" not in kwargs for _, kwargs in seen)


@pytest.mark.parametrize("unsafe", ["/absolute.py", "../escape.py", ".git/config"])
def test_explicit_paths_reject_unsafe_values(repository: Path, tmp_path: Path, unsafe: str) -> None:
    controller = _controller(repository, tmp_path)
    with controller.transaction() as transaction:
        with pytest.raises(gitops.GitOpsError, match="unsafe"):
            transaction.stage_paths(repository, [unsafe])


def test_signed_delivery_uses_isolated_branch_without_moving_source_checkout(
    repository: Path, tmp_path: Path
) -> None:
    source_head = _git(repository, "rev-parse", "HEAD")
    workspace = tmp_path / "run"
    workspace.mkdir()
    delivery = gitops.prepare_isolated_delivery_checkout(
        repository,
        workspace=workspace,
        branch="staircase/run-1/delivery",
        base_commit=source_head,
        expected_head=None,
        lock_path=workspace / "git.lock",
        owner="run-1:generation-1",
    )

    assert delivery.repository == workspace / "delivery" / "repository"
    assert delivery.branch == "staircase/run-1/delivery"
    assert delivery.head == source_head
    assert _git(repository, "branch", "--show-current") == "main"
    assert _git(repository, "rev-parse", "HEAD") == source_head

    adopted = gitops.prepare_isolated_delivery_checkout(
        repository,
        workspace=workspace,
        branch="staircase/run-1/delivery",
        base_commit=source_head,
        expected_head=source_head,
        lock_path=workspace / "git.lock",
        owner="run-1:generation-2",
    )
    assert adopted.repository == delivery.repository
    assert adopted.head == delivery.head


def test_diff_delivery_uses_detached_checkout_without_creating_delivery_branch(
    repository: Path, tmp_path: Path
) -> None:
    source_head = _git(repository, "rev-parse", "HEAD")
    branches_before = _git(repository, "for-each-ref", "--format=%(refname)", "refs/heads")
    workspace = tmp_path / "run-diff"
    workspace.mkdir()

    delivery = gitops.prepare_diff_delivery_checkout(
        repository,
        workspace=workspace,
        base_commit=source_head,
        expected_head=None,
        lock_path=workspace / "git.lock",
        owner="run-diff:generation-1",
    )

    assert delivery.repository == workspace / "delivery" / "diff-repository"
    assert delivery.branch is None
    assert _git(delivery.repository, "branch", "--show-current") == ""
    assert _git(repository, "branch", "--show-current") == "main"
    assert _git(repository, "rev-parse", "HEAD") == source_head
    assert _git(repository, "for-each-ref", "--format=%(refname)", "refs/heads") == branches_before

    (delivery.repository / "model.py").write_text("VALUE = 11\n", encoding="utf-8")
    with delivery.git.transaction() as transaction:
        transaction.stage_paths(delivery.repository, ("model.py",))
        integrated_head = transaction.commit_signed_off(
            delivery.repository,
            message="staircase: integrate diff candidate",
            expected_paths=("model.py",),
        )
    adopted = gitops.prepare_diff_delivery_checkout(
        repository,
        workspace=workspace,
        base_commit=source_head,
        expected_head=integrated_head,
        lock_path=workspace / "git.lock",
        owner="run-diff:generation-2",
    )
    artifact = gitops.publish_delivery_patch(
        adopted,
        base_commit=source_head,
        output_path=workspace / "delivery" / "staircase.patch",
    )
    assert artifact.path.read_bytes() == adopted.git.render_patch(
        base_commit=source_head,
        head_commit=integrated_head,
    )
    assert b"VALUE = 11" in artifact.path.read_bytes()
    assert len(artifact.sha256) == 64
    assert (
        gitops.publish_delivery_patch(
            adopted,
            base_commit=source_head,
            output_path=artifact.path,
        )
        == artifact
    )
    parallel_path = workspace / "delivery" / "parallel.patch"
    with ThreadPoolExecutor(max_workers=8) as executor:
        artifacts = tuple(
            executor.map(
                lambda _index: gitops.publish_delivery_patch(
                    adopted,
                    base_commit=source_head,
                    output_path=parallel_path,
                ),
                range(16),
            )
        )
    assert all(published == artifacts[0] for published in artifacts)

    outside = tmp_path / "outside-delivery"
    outside.mkdir()
    (workspace / "delivery" / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(gitops.GitOpsError, match="escapes"):
        gitops.publish_delivery_patch(
            adopted,
            base_commit=source_head,
            output_path=workspace / "delivery" / "escape" / "staircase.patch",
        )
    assert _git(repository, "rev-parse", "HEAD") == source_head
