"""Execute the showcase notebook with the caller's Python environment."""

from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def execute(notebook, root: Path, output: Path, *, html: bool = False):
    client = NotebookClient(notebook, timeout=180, kernel_name="python3", allow_errors=False,
                            resources={"metadata": {"path": str(root)}})
    client.create_kernel_manager()
    client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client.execute()
    nbformat.validate(notebook)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    if html:
        rendered, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
        output.with_suffix(".html").write_text(rendered)
    return notebook


def run_notebook(source: Path, output: Path, data_source: str = "processed"):
    root = source.resolve().parent.parent
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code" and 'DATA_SOURCE = "processed"' in cell.source:
            cell.source = cell.source.replace('DATA_SOURCE = "processed"', f'DATA_SOURCE = "{data_source}"')
    execute(notebook, root, output, html=True)
    print(f"Executed notebook: {output}")
