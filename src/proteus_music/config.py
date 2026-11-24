"""Local application configuration."""

import os
from pathlib import Path

from platformdirs import user_data_path

DATABASE_ENV = "PROTEUS_DB"


def default_database_path() -> Path:
    """Database location: ``$PROTEUS_DB`` if set, otherwise the per-user data directory."""
    override = os.environ.get(DATABASE_ENV)
    if override:
        return Path(override).expanduser()
    return user_data_path("proteus", appauthor=False) / "proteus.db"
