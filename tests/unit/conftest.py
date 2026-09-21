import pytest


@pytest.fixture(autouse=True)
def _disable_exploration(monkeypatch):
    """Unit tests should only cover deterministic baseline behaviour.

    Wrap ``observa.config.get_settings`` so any attribute access falls through
    to the real Settings object, except ``agents_exploration_enabled`` which is
    pinned to ``False``. Tests that explicitly want exploration wired should
    override this fixture locally.
    """
    try:
        from observa.config import get_settings as _real_get_settings
    except Exception:
        _real_get_settings = None
    try:
        real = _real_get_settings() if _real_get_settings else None
    except Exception:
        real = None

    class _ExplorationDisabledProxy:
        def __getattr__(self, name: str):
            if name == "agents_exploration_enabled":
                return False
            if real is not None:
                return getattr(real, name)
            raise AttributeError(name)

    proxy = _ExplorationDisabledProxy()

    def _patched(*args, **kwargs):
        if args or kwargs:
            # Test is loading a specific config path — defer to the real loader
            # so behaviour (including error cases) is preserved.
            if _real_get_settings is None:
                raise RuntimeError("real get_settings unavailable")
            return _real_get_settings(*args, **kwargs)
        return proxy

    monkeypatch.setattr("observa.config.get_settings", _patched, raising=False)
    yield
