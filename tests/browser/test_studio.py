import pytest

pytestmark = pytest.mark.browser


def test_studio_in_real_browser(tmp_path):
    from scripts.browser_verify import run
    rep = run(port=7872, shots=tmp_path)
    assert rep["failures"] == [], rep["failures"]
    assert len(rep["screenshots"]) >= 12
