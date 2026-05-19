from pathlib import Path
import subprocess
import re


def read_file(file_path: str) -> str:

    return Path(file_path).read_text(
        encoding="utf-8"
    )


def write_file(
    file_path: str,
    content: str,
    overwrite: bool = False,
) -> str:

    path = Path(file_path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {file_path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content,
        encoding="utf-8",
    )

    return f"Wrote file: {file_path}"


def edit_file(
    file_path: str,
    find: str,
    replace: str,
) -> str:

    path = Path(file_path)

    content = path.read_text(
        encoding="utf-8"
    )

    if content.count(find) != 1:
        raise ValueError(
            f"Target text not unique in {file_path}"
        )

    updated = content.replace(
        find,
        replace,
        1,
    )

    path.write_text(
        updated,
        encoding="utf-8",
    )

    return f"Edited file: {file_path}"


def bash(
    command: list[str],
    timeout: int = 30,
    cwd: str | None = None,
) -> str:

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )

    output = result.stdout.strip()
    error = result.stderr.strip()

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{output}\n"
            f"stderr:\n{error}"
        )

    return output or error


def grep_file(
    file_path: str,
    pattern: str,
) -> str:

    content = read_file(file_path)

    matches = re.findall(
        pattern,
        content,
        flags=re.MULTILINE,
    )

    if not matches:
        return "No matches found"

    return "\n".join(
        map(str, matches)
    )