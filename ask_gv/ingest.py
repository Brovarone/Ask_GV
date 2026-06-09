from __future__ import annotations
import fnmatch
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List
from .models import SourceDocument
from .utils import normalize_markdown, read_text, compute_sha256, first_heading_or_filename, extract_headings

def _is_ignored(rel_path: str, ignore_patterns: List[str]) -> bool:
    rel_path = rel_path.replace("\\", "/")
    name = Path(rel_path).name
    for patt in ignore_patterns:
        patt = patt.replace("\\", "/")
        if fnmatch.fnmatch(rel_path, patt) or fnmatch.fnmatch(name, patt):
            return True
    return False

def iter_files_recursive(base: Path, ignore_patterns: List[str]) -> Iterable[Path]:
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(base).as_posix()
        if _is_ignored(rel, ignore_patterns):
            continue
        if p.suffix.lower() == ".md":
            yield p

def clone_repo(repo_url: str, workdir: Path) -> Path:
    repo_dir = workdir / "repo"
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return repo_dir

def expand_local_patterns(patterns: List[str], ignore_patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    cwd = Path(".").resolve()

    for patt in patterns:
        matches = list(Path(".").glob(patt)) if any(ch in patt for ch in "*?[]") else [Path(patt)]
        for p in matches:
            if p.is_file() and p.suffix.lower() == ".md":
                rp = p.resolve()
                rel = rp.relative_to(cwd).as_posix() if rp.is_relative_to(cwd) else rp.as_posix()
                if not _is_ignored(rel, ignore_patterns):
                    out.append(rp)
            elif p.is_dir():
                base_dir = p.resolve()
                for x in p.rglob("*.md"):
                    rx = x.resolve()

                    rel_to_base = rx.relative_to(base_dir).as_posix()
                    rel_to_cwd = rx.relative_to(cwd).as_posix() if rx.is_relative_to(cwd) else rx.as_posix()

                    if _is_ignored(rel_to_base, ignore_patterns):
                        continue
                    if _is_ignored(rel_to_cwd, ignore_patterns):
                        continue

                    out.append(rx)
    uniq, seen = [], set()
    for p in out:
        s = str(p)
        if s not in seen:
            uniq.append(p)
            seen.add(s)
    return uniq

def load_documents_from_repo(repo_url: str, ignore_patterns: List[str]) -> List[SourceDocument]:
    with tempfile.TemporaryDirectory(prefix="rpg_repo_mod_") as tmp:
        repo_dir = clone_repo(repo_url, Path(tmp))
        docs: List[SourceDocument] = []
        for f in iter_files_recursive(repo_dir, ignore_patterns):
            rel = f.relative_to(repo_dir).as_posix()
            content = normalize_markdown(read_text(f))
            if not content:
                continue
            docs.append(SourceDocument(rel, first_heading_or_filename(rel, content), content, compute_sha256(content), len(content), extract_headings(content)))
        return docs

def load_documents_from_files(files: List[str], ignore_patterns: List[str]) -> List[SourceDocument]:
    docs: List[SourceDocument] = []
    for f in expand_local_patterns(files, ignore_patterns):
        content = normalize_markdown(read_text(f))
        if not content:
            continue
        docs.append(
            SourceDocument(
                str(f),
                first_heading_or_filename(str(f), content),
                content,
                compute_sha256(content),
                len(content),
                extract_headings(content),
            )
        )
    return docs
