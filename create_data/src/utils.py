"""I/O helpers: JSONL, downloads, and reading single members of a remote zip via HTTP Range."""
import io
import json
import shutil
import time
import zipfile
from pathlib import Path

import requests


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def download(url, dest, chunk=1 << 20, tries=6):
    """Download url to dest unless it already exists."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(tries):  # slow servers (VG at cs.stanford.edu) time out now and then
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
            break
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(min(60, 2 ** attempt))
    tmp.rename(dest)
    return dest


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests, enough for zipfile."""

    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for attempt in range(8):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 7:
                    raise
                time.sleep(2 ** attempt)
        data = r.content
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


class RemoteZip:
    """Extract individual members of a large remote zip without downloading all of it."""

    def __init__(self, url):
        self.zf = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url), buffer_size=1 << 20))

    def names(self):
        return self.zf.namelist()

    def extract(self, member, dest):
        dest = Path(dest)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with self.zf.open(member) as src, open(tmp, "wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
        tmp.rename(dest)
        return dest
