"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

project = "tapeagents"
nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
src = root / project

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = (project,) + module_path.parts

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    if not doc_path:
        continue

    nav_aliases = {
        "io": "IO",
        "rl": "RL",
        "llms": "LLMs",
        "llm_function": "LLM Function",
    }
    nav_key = list(parts[1:])
    if nav_key:
        for i in range(len(nav_key)):
            if nav_key[i] in nav_aliases:
                nav_key[i] = nav_aliases[nav_key[i]]
            else:
                nav_key[i] = nav_key[i].title().replace("_", " ").strip()
        nav_key = tuple(nav_key)
        nav[nav_key] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        if identifier:
            print(f"::: {identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
