#!/usr/bin/env python3
"""Execute notebook cells in a real Jupyter kernel and save output.
Authors: Group 08 (names/IDs pending), AI-assisted; uses jupyter_client public APIs.
"""
import argparse
import json
import queue
import sys
import time
from pathlib import Path

from jupyter_client import KernelManager


def execute(path, timeout=1200):
    path = Path(path).resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    # Persist incomplete state before starting so a failed rerun cannot retain old results.
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook.setdefault("metadata", {})["execution_status"] = "running"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manager = KernelManager(kernel_name=notebook.get("metadata", {}).get("kernelspec", {}).get("name", "python3"))
    manager.start_kernel(cwd=str(path.parent))
    client = manager.client()
    client.start_channels()
    errors = []
    try:
        client.wait_for_ready(timeout=60)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            cell["outputs"] = []
            print(f"Executing cell {index}: {source.splitlines()[0] if source else '(empty)'}", flush=True)
            identifier = client.execute(source, store_history=True)
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Cell {index} timed out")
                try:
                    message = client.get_iopub_msg(timeout=min(5, remaining))
                except queue.Empty:
                    continue
                if message.get("parent_header", {}).get("msg_id") != identifier:
                    continue
                kind = message["header"]["msg_type"]
                content = message["content"]
                if kind == "status" and content["execution_state"] == "idle":
                    break
                if kind == "execute_input":
                    cell["execution_count"] = content["execution_count"]
                elif kind == "stream":
                    cell["outputs"].append({"output_type": "stream", "name": content["name"], "text": content["text"]})
                elif kind in ("display_data", "execute_result"):
                    output = {"output_type": kind, "data": content["data"], "metadata": content.get("metadata", {})}
                    if kind == "execute_result":
                        output["execution_count"] = content["execution_count"]
                    cell["outputs"].append(output)
                elif kind == "error":
                    cell["outputs"].append({"output_type": "error", **content})
                    errors.append(f"Cell {index}: {content['ename']}: {content['evalue']}")
                elif kind == "clear_output":
                    cell["outputs"] = []
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            if errors:
                raise RuntimeError(errors[-1])
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
    notebook["metadata"]["execution_status"] = "complete"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Saved actual execution outputs: {path}")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, nargs="?", default=Path("Group08_Midterm.ipynb"))
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    execute(args.notebook, args.timeout)


if __name__ == "__main__":
    main()
