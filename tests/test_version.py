"""Package-level smoke tests."""

import sonoloc


def test_version_is_string() -> None:
    assert isinstance(sonoloc.__version__, str)
    assert sonoloc.__version__.count(".") >= 2
