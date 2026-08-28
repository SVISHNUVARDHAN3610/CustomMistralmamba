"""Verify the model/ package imports cleanly and exposes the public API."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Submodules that must import without error (implementation layout).
REQUIRED_MODULES = [
    "model",
    "model.core",
    "model.core.builders",
    "model.core.config",
    "model.core.constants",
    "model.core.optim",
    "model.layers",
    "model.layers.attention",
    "model.layers.moe",
    "model.layers.fusion",
    "model.mixtral",
    "model.hybrid",
    "model.hybrid.losses",
    "model.hybrid.memory",
    "model.hybrid.mamba",
    "model.hybrid.layer",
    "model.hybrid.model",
]

# Symbols re-exported from model.__init__ (public API).
PUBLIC_SYMBOLS = [
    "MEMORY_NAN_FIX_ID",
    "CompressiveMemoryBank",
    "DroplessMoELayer",
    "HybridForCausalLM",
    "HybridMambaMoEConfig",
    "MixtralConfig",
    "MixtralForCausalLM",
    "build_test3_null_baseline_config",
    "count_trainable_params",
    "fused_mamba_scan_available",
    "log_mamba_backend",
]


def main() -> int:
    errors: list[str] = []

    for mod_name in REQUIRED_MODULES:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"failed to import {mod_name}: {exc}")

    try:
        import model as model_pkg
    except Exception as exc:  # noqa: BLE001
        errors.append(f"failed to import model package: {exc}")
        model_pkg = None

    if model_pkg is not None:
        for symbol in PUBLIC_SYMBOLS:
            if not hasattr(model_pkg, symbol):
                errors.append(f"model package missing public symbol: {symbol}")

        declared = set(getattr(model_pkg, "__all__", []))
        missing_from_all = [s for s in PUBLIC_SYMBOLS if s not in declared]
        if missing_from_all:
            errors.append(
                "model.__all__ missing: " + ", ".join(sorted(missing_from_all))
            )

    print("=" * 72)
    print("MODEL PACKAGE VERIFICATION")
    print("=" * 72)
    print(f"  Modules checked: {len(REQUIRED_MODULES)}")
    print(f"  Public symbols:  {len(PUBLIC_SYMBOLS)}")
    print()

    if errors:
        print("## VERDICT: FAIL")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("## VERDICT: PASS")
    print("  All submodules import and public API symbols are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
