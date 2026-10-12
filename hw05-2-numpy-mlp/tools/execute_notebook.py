"""Execute all notebook cells using a standard local Jupyter kernel."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / "assignment.ipynb", as_version=4)
NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
nbformat.write(notebook, root / "assignment.ipynb")
print("Notebook executed successfully")
