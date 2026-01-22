# /// script
# requires-python = ">=3.12"
# dependencies = ["ruamel.yaml"]
# ///
"""Generate API reference documentation for op_engine."""

from pathlib import Path

from ruamel.yaml import YAML  # type: ignore[import-not-found]


def get_module_name(file_path: Path, src_root: Path) -> str:
    """Convert a file path to a Python module name.

    Args:
        file_path: Path to the Python file.
        src_root: Root source directory.

    Returns:
        The fully qualified module name.
    """
    parts = list(file_path.relative_to(src_root).parts)

    # Remove .py extension from the last part
    parts[-1] = parts[-1].removesuffix(".py")

    # Remove __init__ from module names
    if parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts)


def is_public_module(module_path: Path) -> bool:
    """Check if a module is public (not prefixed with _).

    Args:
        module_path: Path to the module file.

    Returns:
        `True` if the module is public, `False` otherwise.
    """
    return not any(
        (part.startswith("_") or part == "__init__.py") for part in module_path.parts
    )


def get_python_files(package_dir: Path) -> list[Path]:
    """Get all Python module files in a package directory.

    Args:
        package_dir: Path to the package directory.

    Returns:
        List of Python file paths, excluding '__init__.py'.
    """
    return [file for file in sorted(package_dir.glob("*.py")) if is_public_module(file)]


def get_subpackages_with_init(package_dir: Path) -> list[Path]:
    """Return direct subpackages that contain an __init__.py."""
    return [
        child
        for child in sorted(package_dir.iterdir())
        if child.is_dir()
        and (child / "__init__.py").exists()
        and is_public_module(child)
    ]


def has_init_in_tree(package_dir: Path) -> bool:
    """Return True if the directory contains any `__init__.py` beneath it."""
    return any(
        pkg_init.name == "__init__.py" for pkg_init in package_dir.rglob("__init__.py")
    )


def generate_api_doc(item_path: Path, src_root: Path, output_dir: Path) -> str:
    """Generate API documentation for a package or module.

    Args:
        item_path: Path to the package directory or module file.
        src_root: Root source directory.
        output_dir: Directory to write the documentation file.

    Returns:
        The name of the generated documentation file (without .md extension).
    """
    # Get the module name from the path
    module_name = get_module_name(item_path, src_root)

    # Create output file name (last part of module name)
    doc_name = module_name.split(".")[-1]
    output_file = output_dir / f"{doc_name}.md"

    # Start building the content
    lines = [f"# {doc_name.replace('_', ' ').title()}", ""]

    # Only include package-level directive if this path has its own __init__.py
    if item_path.is_file() or (
        item_path.is_dir() and (item_path / "__init__.py").exists()
    ):
        lines.extend((f"## ::: {module_name}", ""))

    # If it's a package directory, add references to each submodule
    if item_path.is_dir():
        for py_file in get_python_files(item_path):
            submodule_name = get_module_name(py_file, src_root)
            lines.extend((f"### ::: {submodule_name}", ""))
        for subpkg in get_subpackages_with_init(item_path):
            submodule_name = get_module_name(subpkg, src_root)
            lines.extend((f"### ::: {submodule_name}", ""))

    # Write the file
    output_file.write_text("\n".join(lines))
    print(f"Generated: {output_file}")

    return doc_name


def update_mkdocs_nav(mkdocs_path: Path, api_docs: list[str]) -> None:
    """Update the mkdocs.yml navigation with generated API docs.

    Uses ruamel.yaml to preserve comments and formatting.

    Args:
        mkdocs_path: Path to the mkdocs.yml file.
        api_docs: List of API document names (without .md extension).
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.explicit_start = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    with mkdocs_path.open("r", encoding="utf-8") as f:
        config = yaml.load(f)

    # Find the Reference -> API section in navigation
    nav = config.get("nav", [])
    api_reference_idx = None
    for i, item in enumerate(nav):
        if isinstance(item, dict) and "API Reference" in item:
            api_reference_idx = i
            break

    # If not found, exit
    if api_reference_idx is None:
        print("Warning: Could not find 'API Reference' section in mkdocs.yml")
        return

    # Build the new API navigation structure
    nav[api_reference_idx]["API Reference"] = [
        {doc.replace("_", " ").title(): f"api-reference/{doc}.md"}
        for doc in sorted(api_docs)
    ]
    with mkdocs_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)
    print(f"Updated mkdocs.yml navigation with {len(api_docs)} API docs")


def main() -> None:
    """Generate API reference documentation for all packages."""
    # Define paths
    repo_root = Path(__file__).parent.parent
    src_root = repo_root / "src"
    op_engine_root = src_root / "op_engine"
    output_dir = repo_root / "docs" / "api-reference"
    mkdocs_path = repo_root / "mkdocs.yml"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up existing markdown files in the API reference directory
    for md_file in output_dir.glob("*.md"):
        md_file.unlink()
        print(f"Removed: {md_file}")

    # Find all public packages and modules
    generated_docs = []
    for item in sorted(op_engine_root.iterdir()):
        if item.name == "py.typed":
            continue
        if not is_public_module(item):
            continue
        if item.is_dir():
            if not has_init_in_tree(item):
                continue
        elif item.suffix != ".py":
            continue
        doc_name = generate_api_doc(item, src_root, output_dir)
        generated_docs.append(doc_name)

    # Update mkdocs.yml navigation
    update_mkdocs_nav(mkdocs_path, generated_docs)
    print("API reference generation complete")


if __name__ == "__main__":
    main()
