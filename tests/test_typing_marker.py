from importlib.resources import files


def test_package_contains_pep561_marker() -> None:
    marker = files("causal4d").joinpath("py.typed")
    assert marker.is_file()
    assert marker.read_bytes() == b""
