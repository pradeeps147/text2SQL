from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_renders_professional_workspace_when_backend_is_offline():
    app_path = Path(__file__).resolve().parent.parent / "src" / "app" / "app.py"
    app = AppTest.from_file(app_path, default_timeout=15).run()

    assert not app.exception
    assert any(title.value == "Logistics intelligence" for title in app.title)
    assert len(app.tabs) == 4
    assert any(error.value == "Backend offline" for error in app.error)
