"""Runtime compatibility patch for CUTLASS DSL's Thor NVFP4 MMA guard.

CUTLASS DSL 4.5 has the SM100A/SM103A implementation of the block-scaled
MMA operation, but its Python-side architecture allow-list omits SM110A.  The
hardware operation is used unchanged; this module only extends that guard for
the CuTE DSL baselines bundled with this repository.
"""

from __future__ import annotations

import os
import sys


def apply_thor_mma_compat_patch() -> tuple[str, ...]:
    """Allow the CUTLASS DSL block-scaled MMA wrappers to target SM110A.

    The patch is idempotent and intentionally applies only in this Python
    process.  It neither edits TensorRT-LLM nor any installed package.
    """
    # The DSL runtime otherwise leaves its launch target unknown on Thor.
    # Keep an explicit user override intact.
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_110a")

    try:
        from cutlass.base_dsl.arch import Arch
        from cutlass.cute.nvgpu.tcgen05 import copy, mma
    except Exception as exc:  # pragma: no cover - depends on optional CuTE DSL
        raise RuntimeError(
            "Unable to load CUTLASS DSL for the SM110A CuTE compatibility patch"
        ) from exc

    thor_archs = (Arch.sm_110a,)
    patched = []
    # CUTLASS versions name the wrapper differently.  NVFP4 uses
    # MmaMXF4NVF4Op in the currently pinned version; BlockScaledMmaOp keeps
    # this compatible with adjacent releases.
    for module, names in (
        (mma, ("MmaMXF4NVF4Op", "BlockScaledMmaOp")),
    ):
        for name in names:
            op = getattr(module, name, None)
            if op is None:
                continue
            admissible_archs = list(op.admissible_archs)
            missing_archs = [arch for arch in thor_archs if arch not in admissible_archs]
            if missing_archs:
                op.admissible_archs = admissible_archs + missing_archs
                patched.append(name)

    # tcgen05 load/copy instructions have several concrete wrappers.  Their
    # allow-lists differ by instruction, so extend every copy wrapper that
    # exposes one instead of discovering them one at a time during JIT.
    for name, op in vars(copy).items():
        if not isinstance(op, type) or not hasattr(op, "admissible_archs"):
            continue
        admissible_archs = list(op.admissible_archs)
        missing_archs = [arch for arch in thor_archs if arch not in admissible_archs]
        if missing_archs:
            op.admissible_archs = admissible_archs + missing_archs
            patched.append(name)

    # S2T copy ops use an ``Arch.is_family_of(sm_100f)`` guard instead of an
    # ``admissible_archs`` class attribute.  Retain its CTA-group validation
    # and bypass only that guard on Thor.  Cp4x32x128bOp is used to transfer
    # NVFP4 scale factors in the persistent GEMM main loop.
    s2t_copy_base = getattr(copy, "_S2TCopyBase", None)
    if (
        s2t_copy_base is not None
        and not hasattr(s2t_copy_base, "_sm110a_original_post_init")
    ):
        original_post_init = s2t_copy_base.__post_init__

        def thor_compatible_post_init(self):
            arch = copy.BaseDSL._get_dsl().get_arch_enum()
            if arch not in thor_archs:
                return original_post_init(self)
            if not isinstance(self.cta_group, copy.CtaGroup):
                raise copy.OpError(
                    self,
                    "expects the 'cta_group' Op parameter to be a tcgen05.CtaGroup instance",
                )

        s2t_copy_base._sm110a_original_post_init = original_post_init
        s2t_copy_base.__post_init__ = thor_compatible_post_init
        patched.append("_S2TCopyBase")

    if os.environ.get("CUTE_THOR_PATCH_VERBOSE") == "1":
        names = ", ".join(patched) if patched else "already applied"
        print(f"[cute-thor-patch] SM110A enabled for: {names}", file=sys.stderr)
    return tuple(patched)
