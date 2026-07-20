# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime property attachment for _KVCache — INTENTIONALLY NOT COMPILED.

mypyc supports read-only @property on non-extension classes but rejects the
@x.setter decorator form, chokes on in-class `x = property(...)` assignments
(compiled class-dict lookup fails under some import orders), and its semantic
analyzer crashes on module-level class-attribute assignment in compiled
modules. This module is excluded from the mypyc build (see setup_mypyc.py),
so the plain setattr below runs interpreted and upgrades each read-only
property to a full read/write property. External writers (the pyexecutor
wrapper and the test suite) keep the attribute-assignment API unchanged;
compiled internal code calls the _set_* methods directly.
"""
from ._kv_cache import _KVCache

for _name in (
    "cuda_stream",
    "beam_width",
    "enable_swa_scratch_reuse",
    "capacity",
    "history_length",
):
    _readonly = getattr(_KVCache, _name)
    setattr(
        _KVCache,
        _name,
        property(_readonly.fget, getattr(_KVCache, f"_set_{_name}")),
    )
del _name, _readonly
