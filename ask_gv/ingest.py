from __future__ import annotations
import fnmatch
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List
from .models import SourceDocument
from .utils import normalize_markdown, read_text, compute_sha256, first_heading_or_filename, extract_headings

def iter_files_recursive(base: Path, ignore_patterns: List[str]) -> Iterable[Path]:
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(base).as_posix()
        if any(fnmatch.fnmatch(rel, patt) for patt in ignore_patterns):
            continue
        if p.suffix.lower() == ".md":
            yield p

def clone_repo(repo_url: str, workdir: Path) -> Path:
    repo_dir = workdir / "repo"
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return repo_dir

def expand_local_patterns(patterns: List[str]) -> List[Path]:
    out = []
    for patt in patterns:
        matches = list(Path(".").glob(patt)) if any(ch in patt for ch in "*?[]") else [Path(patt)]
        for p in matches:
            if p.is_file() and p.suffix.lower() == ".md":
                out.append(p.resolve())
            elif p.is_dir():
                out.extend(x.resolve() for x in p.rglob("*.md"))
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

def load_documents_from_files(files: List[str]) -> List[SourceDocument]:
    docs: List[SourceDocument] = []
    for f in expand_local_patterns(files):
        content = normalize_markdown(read_text(f))
        if not content:
            continue
        docs.append(SourceDocument(str(f), first_heading_or_filename(str(f), content), content, compute_sha256(content), len(content), extract_headings(content)))
    return docs
