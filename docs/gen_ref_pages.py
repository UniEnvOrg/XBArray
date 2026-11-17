"""Build the site nav (SUMMARY.md) and one API page per importable xbarray module."""
from pathlib import Path

import mkdocs_gen_files

PACKAGE_DIR = Path("xbarray")
DOCS_DIR = Path("docs")
SECTION_ORDER = ["index.md", "getting-started.md", "concepts", "guides"]


def is_private(path: Path) -> bool:
    return any(part.startswith("_") and part != "__init__.py" for part in path.parts)


def humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").capitalize()


def doc_sort_key(rel: Path):
    top = rel.parts[0]
    section = SECTION_ORDER.index(top) if top in SECTION_ORDER else len(SECTION_ORDER)
    # section index first, development guide last, the rest alphabetical
    return (section, rel.name != "index.md", rel.stem == "development", rel.as_posix())


def nav_title(rel: Path) -> tuple[str, ...]:
    if rel == Path("index.md"):
        return ("Home",)
    parents = tuple(humanize(p) for p in rel.parts[:-1])
    if rel.name == "index.md":
        return parents
    with open(DOCS_DIR / rel, encoding="utf-8") as fd:
        first = fd.readline().strip()
    title = first.lstrip("# ").strip() if first.startswith("# ") else humanize(rel.stem)
    return parents + (title,)


site_nav = mkdocs_gen_files.Nav()
docs = [p.relative_to(DOCS_DIR) for p in DOCS_DIR.rglob("*.md")]
for rel in sorted((p for p in docs if p.parts[0] != "api" and p.name != "SUMMARY.md"), key=doc_sort_key):
    site_nav[nav_title(rel)] = rel.as_posix()

api_nav = mkdocs_gen_files.Nav()
for source in sorted(PACKAGE_DIR.rglob("*.py")):
    rel = source.relative_to(PACKAGE_DIR.parent)
    if is_private(rel) or source.name == "__main__.py":
        continue
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        page = Path("api", *parts, "index.md")
    else:
        page = Path("api", *parts).with_suffix(".md")
    if not (source.parent / "__init__.py").exists():
        continue
    api_nav[tuple(parts)] = page.relative_to("api").as_posix()
    with mkdocs_gen_files.open(page, "w") as fd:
        fd.write(f"# `{'.'.join(parts)}`\n\n::: {'.'.join(parts)}\n")
    mkdocs_gen_files.set_edit_path(page, source)

with mkdocs_gen_files.open("api/index.md", "w") as fd:
    fd.write("# API reference\n\nGenerated from the docstrings of every public `xbarray` module.\n\n")
    fd.writelines(api_nav.build_literate_nav())

with mkdocs_gen_files.open("SUMMARY.md", "w") as fd:
    fd.writelines(site_nav.build_literate_nav())
    fd.write("* [API reference](api/index.md)\n")
    for line in api_nav.build_literate_nav():
        # 4 spaces: Python-Markdown nests a list item only at 4+ spaces of indent.
        fd.write("    " + line.replace("](", "](api/", 1))
