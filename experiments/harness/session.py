import json
from pathlib import Path
from typing import Protocol, Iterable, Dict, Any


class SessionManager(Protocol):
    def append(self, session_id: str, event: Dict[str, Any]) -> None:
        ...

    def replay(self, session_id: str) -> Iterable[Dict[str, Any]]:
        ...


class JSONLSessionManager:
    """
    Simple append-only session manager using JSONL files.
    Each session_id maps to a separate .jsonl file.
    """

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _file_path(self, session_id: str) -> Path:
        return self.base_path / f"{session_id}.jsonl"

    def append(self, session_id: str, event: Dict[str, Any]) -> None:
        path = self._file_path(session_id)

        line = json.dumps(event, ensure_ascii=False, default=str)

        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def replay(self, session_id: str) -> Iterable[Dict[str, Any]]:
        path = self._file_path(session_id)

        if not path.exists():
            return

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)