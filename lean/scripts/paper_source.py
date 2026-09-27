"""Read the multi-file manuscript for optional structural (not semantic) audits."""
from pathlib import Path
import re

_LOCATIONS: list[str] = []


def read_paper_source(main: Path) -> str:
    root = main.resolve().parent
    locations: list[str] = []
    output: list[str] = []

    def expand(path: Path, stack: tuple[Path, ...]) -> None:
        path = path.resolve()
        if path in stack:
            raise ValueError(f"Cyclic LaTeX input: {path}")
        if not path.is_relative_to(root):
            raise ValueError(f"LaTeX input outside paper directory: {path}")
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = re.split(r"(?<!\\)%", raw, maxsplit=1)[0]
            start = 0
            for match in re.finditer(r"\\input\{([^}]+)\}", line):
                output.append(line[start:match.start()])
                locations.append(f"paper/{path.relative_to(root)}:{number}")
                child = root / match.group(1)
                if child.suffix != ".tex":
                    child = child.with_suffix(".tex")
                expand(child, (*stack, path))
                start = match.end()
            output.append(line[start:])
            locations.append(f"paper/{path.relative_to(root)}:{number}")

    expand(main, ())
    _LOCATIONS[:] = locations
    return "\n".join(output)


def paper_source_location(line: int) -> str:
    if 1 <= line <= len(_LOCATIONS):
        return _LOCATIONS[line - 1]
    return f"paper/ (expanded source line {line})"
