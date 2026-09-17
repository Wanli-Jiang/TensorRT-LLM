# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Digest-bound launch policy for scheduler-client-free agent workers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping

_IDENTITY = re.compile(r"[^\x00\r\n]{1,512}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_NETWORK_POLICY = "backend_api_only"
_NETWORK_ENFORCEMENT = "certified_worker_image"


class AgentLaunchPolicyError(ValueError):
    """Raised when an agent launch policy is absent, ambiguous, or changed."""


@dataclass(frozen=True, slots=True)
class AgentLaunchPolicy:
    """Exact worker image and trusted network enforcement identity."""

    image: str
    build_identity: str
    network_policy: str
    network_enforcement: str
    digest: str

    def __post_init__(self) -> None:
        image = PurePosixPath(self.image)
        if not image.is_absolute() or ".." in image.parts or image.as_posix() != self.image:
            raise AgentLaunchPolicyError("agent worker image must be a normalized absolute path")
        if _IDENTITY.fullmatch(self.build_identity) is None:
            raise AgentLaunchPolicyError(
                "agent worker build identity must be bounded and single-line"
            )
        if self.network_policy != _NETWORK_POLICY:
            raise AgentLaunchPolicyError("agent worker network policy must be backend_api_only")
        if self.network_enforcement != _NETWORK_ENFORCEMENT:
            raise AgentLaunchPolicyError(
                "agent worker network policy requires certified_worker_image enforcement"
            )
        if _DIGEST.fullmatch(self.digest) is None or self.digest != _policy_digest(self):
            raise AgentLaunchPolicyError("agent worker launch policy digest mismatch")

    def to_public_dict(self) -> dict[str, str]:
        """Return the canonical secret-free policy frozen into worker input."""
        return {
            "image": self.image,
            "build_identity": self.build_identity,
            "network_policy": self.network_policy,
            "network_enforcement": self.network_enforcement,
            "digest": self.digest,
        }

    @classmethod
    def create(
        cls,
        *,
        image: str,
        build_identity: str,
        network_policy: str = _NETWORK_POLICY,
        network_enforcement: str = _NETWORK_ENFORCEMENT,
    ) -> AgentLaunchPolicy:
        """Construct a canonical policy and derive its content digest."""
        return cls(
            image=image,
            build_identity=build_identity,
            network_policy=network_policy,
            network_enforcement=network_enforcement,
            digest=_policy_digest_values(
                image, build_identity, network_policy, network_enforcement
            ),
        )


def agent_launch_policy_from_public_dict(value: object) -> AgentLaunchPolicy:
    """Strictly decode the canonical policy carried by an immutable manifest."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AgentLaunchPolicyError("agent launch policy must be an object")
    expected = {
        "image",
        "build_identity",
        "network_policy",
        "network_enforcement",
        "digest",
    }
    if set(value) != expected:
        raise AgentLaunchPolicyError(
            "agent launch policy keys invalid; "
            f"missing={sorted(expected - set(value))}, unknown={sorted(set(value) - expected)}"
        )
    if not all(isinstance(value[key], str) for key in expected):
        raise AgentLaunchPolicyError("agent launch policy values must be strings")
    return AgentLaunchPolicy(**{key: value[key] for key in expected})  # type: ignore[arg-type]


def _policy_digest(policy: AgentLaunchPolicy) -> str:
    return _policy_digest_values(
        policy.image,
        policy.build_identity,
        policy.network_policy,
        policy.network_enforcement,
    )


def _policy_digest_values(
    image: str,
    build_identity: str,
    network_policy: str,
    network_enforcement: str,
) -> str:
    payload = {
        "image": image,
        "build_identity": build_identity,
        "network_policy": network_policy,
        "network_enforcement": network_enforcement,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AgentLaunchPolicy",
    "AgentLaunchPolicyError",
    "agent_launch_policy_from_public_dict",
]
