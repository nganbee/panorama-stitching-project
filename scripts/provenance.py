"""Fingerprints for the native program and the reproducible report procedure.

Authors: Group 08 (names/IDs pending), with OpenAI Codex assistance.
External sources: Python standard-library APIs; no copied implementation.

``source_sha256`` identifies native sources/headers in both src/ and tools/.
``implementation_sha256`` additionally identifies Python report/build helpers,
CMake configuration, and the notebook's ordered cell types and source text.
Notebook results, execution counts and metadata are deliberately excluded:
saving newly measured outputs must not change the procedure being measured.
Paths are relative to the project and records are length-framed, so identical
archive contents have the same fingerprint after extraction anywhere.
"""
import hashlib
import json
from pathlib import Path


NATIVE_SUFFIXES = frozenset((".h", ".hh", ".hpp", ".hxx", ".cuh", ".inl",
                             ".c", ".cc", ".cpp", ".cxx", ".cu"))
REPORT_NAME = "Group08_Midterm.ipynb"


def _native_inputs(root):
    records = []
    for directory in ("src", "tools"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and path.suffix.lower() in NATIVE_SUFFIXES:
                records.append((path.relative_to(root).as_posix(), path.read_bytes()))
    return records


def _digest(domain, records):
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii") + b"\0")
    for relative_path, content in sorted(records, key=lambda record: record[0]):
        name = relative_path.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def notebook_source_bytes(path):
    """Canonical notebook procedure, independent of JSON formatting/results.

    Jupyter permits cell ``source`` to be either a string or a list of strings;
    both representations are normalized to the same joined text. Cell order
    and every character of source text remain significant.
    """
    notebook = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Notebook cells must be a list")
    normalized = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or not isinstance(cell.get("cell_type"), str):
            raise ValueError(f"Invalid notebook cell type at index {index}")
        source = cell.get("source", "")
        if isinstance(source, list) and all(isinstance(part, str) for part in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise ValueError(f"Invalid notebook cell source at index {index}")
        normalized.append({"cell_type": cell["cell_type"], "source": source})
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def source_sha256(root):
    """Hash all native program/test sources, excluding binaries and documents."""
    root = Path(root).resolve()
    return _digest("panorama-native-source-v2", _native_inputs(root))


def implementation_sha256(root, notebook_path=None):
    """Hash the program plus the report procedure that generates its evidence.

    ``notebook_path`` can select the actual notebook exported from Colab. Its
    location/name does not affect the hash; the report has one canonical path
    in the inventory. A missing explicit override is an error. A default report
    that is absent is omitted, allowing source-only builds while ensuring that
    adding/removing a report changes the implementation fingerprint.
    """
    root = Path(root).resolve()
    records = _native_inputs(root)
    for path in (root / "scripts").rglob("*.py"):
        if path.is_file():
            records.append((path.relative_to(root).as_posix(), path.read_bytes()))
    cmake = root / "CMakeLists.txt"
    if cmake.is_file():
        records.append(("CMakeLists.txt", cmake.read_bytes()))
    notebook = root / REPORT_NAME if notebook_path is None else Path(notebook_path).resolve()
    if notebook_path is not None or notebook.is_file():
        records.append((REPORT_NAME, notebook_source_bytes(notebook)))
    return _digest("panorama-report-implementation-v1", records)
