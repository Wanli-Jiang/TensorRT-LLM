#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly delivery_root="$(cd "${script_dir}/.." && pwd)"
cd "${delivery_root}"
sha256sum --check SHA256SUMS
echo "DELIVERY_BUNDLE_CHECKSUMS=PASS"
