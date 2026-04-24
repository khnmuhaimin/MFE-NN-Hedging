from pathlib import Path

def project_path(relative_path: str, marker: str = "pyproject.toml") -> Path:
    """
    Convert a project-relative path string into an absolute Path.

    Example:
        project_path("results/figures/plot.png")
    """
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / marker).exists():
            return (parent / relative_path).resolve()

    raise FileNotFoundError(
        f"Could not find project root containing '{marker}'"
    )