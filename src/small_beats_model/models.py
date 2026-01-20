from pydantic import BaseModel, Field


class BeatSaverDiff(BaseModel):
    """Represents a single difficulty of a map."""

    njs: float
    notes: int
    nps: float
    characteristic: str
    difficulty: str
    seconds: float


class VeatSaverVersion(BaseModel):
    """Represents a specific version of a map."""

    diffs: list[BeatSaverDiff]
    downloadURL: str


class BeatSaverMetadata(BaseModel):
    """Inner metadata about the song itself."""

    bpm: float
    duration: int
    songName: str
    songAuthorName: str
    levelAuthorName: str


class BeatSaverMap(BaseModel):
    """The top-level Map object from the API."""

    id: str
    name: str
    description: str
    metadata: BeatSaverMetadata
    versions: list[VeatSaverVersion]


class BeatSaverResponse(BaseModel):
    """The wrapper for search/latest results."""

    docs: list[BeatSaverMap]

