import mgsl


def test_version():
    assert isinstance(mgsl.__version__, str)
