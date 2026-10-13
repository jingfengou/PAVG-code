"""按需从 AOSS 下载 H5，并提供给 DataLoader 使用的本地缓存。

设计目标：
1. AOSS 只作为长期对象存储；h5py 始终读取本地完整文件。
2. 每个 DataLoader worker 懒加载自己的 AOSS Client，避免跨进程共享连接。
3. 下载写入 .part 文件，完成后原子改名；同一文件使用 flock 防止重复下载。
4. 缓存超过 max_cache_bytes 时，按访问时间淘汰旧文件。

依赖：aoss-python-sdk >= 2.2.6、h5py、torch。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import h5py
from torch.utils.data import Dataset, get_worker_info


class AossH5Cache:
    """把 AOSS 对象按需下载到本地缓存目录。"""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        config_path: str | os.PathLike[str] = "~/aoss.conf",
        max_cache_bytes: int = 200 * 1024**3,
        max_cache_files: int | None = None,
        retries: int = 5,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be positive")
        if max_cache_files is not None and int(max_cache_files) <= 0:
            raise ValueError("max_cache_files must be positive when provided")
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = str(Path(config_path).expanduser())
        self.max_cache_bytes = int(max_cache_bytes)
        self.max_cache_files = (
            None if max_cache_files is None else int(max_cache_files)
        )
        self.retries = max(1, int(retries))
        self._client_factory = client_factory
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory(self.config_path)
            else:
                from aoss_client.client import Client

                self._client = Client(self.config_path)
        return self._client

    def _paths(self, uri: str) -> tuple[Path, Path, Path]:
        digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
        suffix = Path(uri.split("?", 1)[0]).suffix or ".h5"
        data_path = self.cache_dir / f"{digest}{suffix}"
        part_path = data_path.with_name(data_path.name + ".part")
        lock_path = data_path.with_name(data_path.name + ".lock")
        return data_path, part_path, lock_path

    @staticmethod
    def _touch(path: Path) -> None:
        now = time.time()
        os.utime(path, (now, now))

    def _download_once(self, uri: str, target: Path) -> None:
        client = self._get_client()
        client.download_file(uri, str(target))

    def _data_files(self) -> list[Path]:
        return [
            path
            for path in self.cache_dir.iterdir()
            if path.is_file()
            and not path.name.endswith((".part", ".lock", ".json"))
        ]

    @contextmanager
    def _exclusive_lock(self, path: Path):
        """Lock one URI while downloading and updating the shared cache.

        Linux training uses ``fcntl``.  A small ``msvcrt`` fallback keeps the
        cache unit-testable on Windows; the production target remains Linux.
        """
        with path.open("a+b") as lock:
            try:
                import fcntl
            except ImportError:
                import msvcrt

                lock.seek(0)
                lock.write(b"0")
                lock.flush()
                locked = False
                while not locked:
                    try:
                        msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                        locked = True
                    except OSError:
                        time.sleep(0.05)
                try:
                    yield
                finally:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _evict(self, protected: Path | None = None) -> None:
        files = self._data_files()
        total = sum(path.stat().st_size for path in files)

        def over_limit() -> bool:
            return total > self.max_cache_bytes or (
                self.max_cache_files is not None
                and len(files) > self.max_cache_files
            )

        if not over_limit():
            return
        for path in sorted(files, key=lambda item: item.stat().st_atime):
            if protected is not None and path == protected:
                continue
            try:
                total -= path.stat().st_size
                path.unlink()
                files.remove(path)
            except FileNotFoundError:
                continue
            if not over_limit():
                break

    def ensure_local(self, uri: str) -> Path:
        data_path, part_path, lock_path = self._paths(uri)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._exclusive_lock(lock_path):
            if data_path.is_file() and data_path.stat().st_size > 0:
                self._touch(data_path)
                return data_path
            last_error: Exception | None = None
            for attempt in range(self.retries):
                try:
                    part_path.unlink(missing_ok=True)
                    self._download_once(uri, part_path)
                    if not part_path.is_file() or part_path.stat().st_size == 0:
                        raise IOError(f"empty AOSS download: {uri}")
                    os.replace(part_path, data_path)
                    self._touch(data_path)
                    self._evict(protected=data_path)
                    return data_path
                except Exception as exc:  # noqa: BLE001 - retry remote IO errors
                    last_error = exc
                    part_path.unlink(missing_ok=True)
                    if attempt + 1 < self.retries:
                        time.sleep(min(2**attempt, 16))
            raise RuntimeError(f"AOSS download failed after {self.retries} attempts: {uri}") from last_error

    def read_h5(self, uri: str, reader: Callable[[h5py.File], Any]) -> Any:
        """下载/命中缓存后打开 H5；reader 返回的数据会在文件关闭后保留。"""
        path = self.ensure_local(uri)
        with h5py.File(path, "r") as handle:
            result = reader(handle)
        self._touch(path)
        return result

    def stats(self) -> dict[str, int]:
        """Return cache usage for monitoring and integration tests."""
        files = self._data_files()
        return {
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        }


class AossH5Dataset(Dataset):
    """把 AOSS URI 列表接入 PyTorch DataLoader 的最小 Dataset。"""

    def __init__(
        self,
        uris: Iterable[str],
        reader: Callable[[h5py.File], Any],
        cache_dir: str | os.PathLike[str],
        config_path: str | os.PathLike[str] = "~/aoss.conf",
        max_cache_bytes: int = 200 * 1024**3,
        max_cache_files: int | None = None,
        retries: int = 5,
    ) -> None:
        self.uris = list(uris)
        self.reader = reader
        self.cache_kwargs = {
            "cache_dir": cache_dir,
            "config_path": config_path,
            "max_cache_bytes": max_cache_bytes,
            "max_cache_files": max_cache_files,
            "retries": retries,
        }
        self._cache_by_worker: dict[int, AossH5Cache] = {}

    def _cache(self) -> AossH5Cache:
        info = get_worker_info()
        worker_id = -1 if info is None else info.id
        if worker_id not in self._cache_by_worker:
            self._cache_by_worker[worker_id] = AossH5Cache(**self.cache_kwargs)
        return self._cache_by_worker[worker_id]

    def __len__(self) -> int:
        return len(self.uris)

    def __getitem__(self, index: int) -> Any:
        return self._cache().read_h5(self.uris[index], self.reader)


def load_uri_manifest(path: str | os.PathLike[str]) -> list[str]:
    """读取每行一个 URI 的 manifest；空行和 # 注释会被忽略。"""
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            result.append(line)
    return result


def write_cache_metadata(path: str | os.PathLike[str], **metadata: Any) -> None:
    """记录缓存配置，便于实验复现。"""
    Path(path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
