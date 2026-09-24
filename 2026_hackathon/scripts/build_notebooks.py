"""Convert the percent-format sources in notebooks/ into .ipynb.

The sources are kept as .py so they diff cleanly in git; the .ipynb files are
what participants open. Run after editing any notebook source.
"""

import pathlib

import nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent / "notebooks"


def split_cells(text):
    cell, kind = [], "code"
    for line in text.splitlines():
        if line.startswith("# %%"):
            if any(entry.strip() for entry in cell):
                yield kind, "\n".join(cell).strip("\n")
            cell, kind = [], "markdown" if "[markdown]" in line else "code"
            continue
        cell.append(line[2:] if kind == "markdown" and line.startswith("# ") else
                    "" if kind == "markdown" and line.strip() == "#" else line)
    if any(entry.strip() for entry in cell):
        yield kind, "\n".join(cell).strip("\n")


def main():
    for source in sorted(ROOT.glob("*.py")):
        notebook = nbformat.v4.new_notebook()
        notebook.cells = [
            nbformat.v4.new_markdown_cell(body)
            if kind == "markdown"
            else nbformat.v4.new_code_cell(body)
            for kind, body in split_cells(source.read_text())
        ]
        notebook.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        }
        target = source.with_suffix(".ipynb")
        nbformat.write(notebook, target)
        print(f"  {source.name} -> {target.name} ({len(notebook.cells)} cells)")


if __name__ == "__main__":
    main()
