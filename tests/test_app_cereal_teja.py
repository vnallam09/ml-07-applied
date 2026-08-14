# ============================================================
# tests/test_app_cereal_teja.py
# ============================================================
# WHY: Smoke test the custom Phase 5 module, matching the example's test.
#
# Run:
#   uv run python -m pytest

import pytest

from mlstudio import app_cereal_teja


def test_app_cereal_teja_has_main() -> None:
    """Verify the custom module exposes a main function."""
    assert callable(app_cereal_teja.main)


def test_predict_refuses_incomplete_payload() -> None:
    """An incomplete payload must be refused, not answered.

    This mirrors the HTTP 400 the deployed penguin API returns when a
    required feature is missing.
    """
    payload = dict.fromkeys(app_cereal_teja.FEATURE_COLS[:-1], 0.0)

    with pytest.raises(KeyError):
        app_cereal_teja.predict(model=None, payload=payload)  # type: ignore[arg-type]
