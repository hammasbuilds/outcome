"""The downloader: ranged, resumable, and never leaves a truncated parquet
under the real name. No network: urlopen is replaced by an in-memory server."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_data.py"
spec = importlib.util.spec_from_file_location("fetch_data", SCRIPT)
fetch_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_data)

PAYLOAD = bytes(range(256)) * 40  # 10,240 bytes


class _Server:
    """Serves PAYLOAD by Range; can be told to fail after N ranged requests."""

    def __init__(self, fail_after: int | None = None):
        self.fail_after = fail_after
        self.ranges: list[str] = []

    def __call__(self, request, timeout=None):
        if request.get_method() == "HEAD":
            response = io.BytesIO(b"")
            response.headers = {"Content-Length": str(len(PAYLOAD))}
        else:
            if self.fail_after is not None and len(self.ranges) >= self.fail_after:
                raise TimeoutError("simulated stall")
            spec = request.get_header("Range")
            self.ranges.append(spec)
            start, end = (int(x) for x in spec.split("=")[1].split("-"))
            response = io.BytesIO(PAYLOAD[start : end + 1])
        return _Ctx(response)


class _Ctx:
    def __init__(self, inner):
        self.inner = inner

    def __enter__(self):
        return self.inner

    def __exit__(self, *a):
        return False


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(fetch_data, "CHUNK", 1000)


def test_interrupted_fetch_leaves_only_a_part_file_then_resumes(
    tmp_path, monkeypatch, small_chunks
):
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", _Server(fail_after=3))
    with pytest.raises(TimeoutError):
        fetch_data.fetch("ecthr_a", "test", tmp_path)
    final = tmp_path / "ecthr_a_test.parquet"
    part = tmp_path / "ecthr_a_test.parquet.part"
    assert not final.exists()
    assert part.stat().st_size == 3000

    server = _Server()
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", server)
    fetch_data.fetch("ecthr_a", "test", tmp_path)
    assert server.ranges[0] == "bytes=3000-3999"  # resumed, not restarted
    assert final.read_bytes() == PAYLOAD
    assert not part.exists()


def test_wrong_size_leftover_under_the_real_name_is_replaced(tmp_path, monkeypatch, small_chunks):
    final = tmp_path / "ecthr_b_train.parquet"
    final.write_bytes(PAYLOAD[:500])  # what the old, non-atomic fetch left behind
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", _Server())
    fetch_data.fetch("ecthr_b", "train", tmp_path)
    assert final.read_bytes() == PAYLOAD


def test_complete_file_is_not_downloaded_again(tmp_path, monkeypatch):
    (tmp_path / "ecthr_a_validation.parquet").write_bytes(PAYLOAD)
    server = _Server()
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", server)
    fetch_data.fetch("ecthr_a", "validation", tmp_path)
    assert server.ranges == []


def test_outcome_data_redirects_the_download(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOME_DATA", str(tmp_path))
    assert fetch_data.data_dir() == tmp_path
