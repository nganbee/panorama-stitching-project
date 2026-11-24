#!/usr/bin/env python3
"""Export current Colab outputs, or validate an explicitly saved notebook.
Authors: Team 8; implementation assisted by OpenAI Codex.
External API: googlecolab/colabtools google/colab/_message.py and files.py
(https://github.com/googlecolab/colabtools); no source implementation copied.
Colab's get_ipynb request is internal; failure requires explicit saved-file input.
"""
import argparse
import json
from pathlib import Path


def frontend_notebook():
    from google.colab import _message
    response = _message.blocking_request("get_ipynb", request="", timeout_sec=15)
    if isinstance(response, str):
        response = json.loads(response)
    notebook = response.get("ipynb", response) if isinstance(response, dict) else None
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4 or not notebook.get("cells"):
        raise ValueError("Colab frontend did not return an nbformat-4 notebook")
    return notebook


def sync_source_notebook(destination):
    """Sync edited cell sources before build; clear outputs in this source copy."""
    notebook = frontend_notebook()
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook.setdefault("metadata", {})["execution_status"] = "source_copy"
    Path(destination).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def validate_notebook(notebook):
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4 or not notebook.get("cells"):
        raise ValueError("Export did not contain a valid nbformat-4 notebook")
    code = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if not code or any(cell.get("execution_count") is None for cell in code):
        raise ValueError("Export contains unexecuted cells; finish Runtime → Run all before exporting")
    if any(output.get("output_type") == "error" for cell in code for output in cell.get("outputs", [])):
        raise ValueError("Export contains execution errors; fix and rerun before packaging")
    return notebook


def export_current_notebook(destination, saved_notebook=None):
    """Never use an old workspace copy as a fallback for current UI outputs."""
    destination = Path(destination).resolve()
    if saved_notebook is not None:
        notebook = json.loads(Path(saved_notebook).read_text(encoding="utf-8"))
    else:
        try:
            notebook = frontend_notebook()
        except Exception as error:
            raise RuntimeError("Colab export unavailable. File → Download → .ipynb with outputs, upload that file, "
                               "then export_current_notebook(destination, saved_notebook=uploaded_path).") from error
    validate_notebook(notebook)
    # Colab does not update our local executor's custom status field. All code
    # counts/error outputs were validated above; preserve every actual output.
    notebook.setdefault("metadata", {})["execution_status"] = "complete"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-notebook", type=Path, required=True, help="Explicit notebook downloaded after Run all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_current_notebook(args.output, args.saved_notebook))


if __name__ == "__main__":
    main()
