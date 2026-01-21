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


# ---


class MapDiffNote(BaseModel):
    time: float = Field(alias="_time")
    lineIndex: int = Field(
        alias="_lineIndex",
        description="Horizontal width of the notes 0=left 1=mid_left 2=mid_right 3=right",
    )
    lineLayer: int = Field(
        alias="_lineLayer",
        description="Vertical height of the notes 0=bottom 1=middle 2=top",
    )
    type: int = Field(alias="_type", description="Color 0=red 1=blue")
    cutDirection: int = Field(
        alias="_cutDirection",
        description="0=up 1=down 2=left 3=right 4=up-left 5=up-right 6=down-left 7=down-right 8=dot",
    )


class MapDiffFile(BaseModel):
    version: str = Field(alias="_version")
    notes: list[MapDiffNote] = Field(alias="_notes")
    events: list = Field(alias="_events")
    obstacles: list = Field(alias="_obstacles")
    bookmarks: list = Field(alias="_bookmarks")


class MapInfoDiff(BaseModel):
    difficulty: str = Field(alias="_difficulty")
    difficultyRank: int = Field(alias="_difficultyRank")
    beatmapFilename: str = Field(alias="_beatmapFilename")
    noteJumpMovementSpeed: float = Field(alias="_noteJumpMovementSpeed")
    noteJumpStartBeatOffset: float = Field(alias="_noteJumpStartBeatOffset")


class MapInfoDiffSet(BaseModel):
    beatmapCharacteristicName: str = Field(alias="_beatmapCharacteristicName")
    difficultyBeatmaps: list[MapInfoDiff] = Field(alias="_difficultyBeatmaps")


class MapInfoFile(BaseModel):
    version: str = Field(alias="_version")
    songName: str = Field(alias="_songName")
    songSubName: str = Field(alias="_songSubName")
    songAuthorName: str = Field(alias="_songAuthorName")
    levelAuthorName: str = Field(alias="_levelAuthorName")
    beatsPerMinute: float = Field(alias="_beatsPerMinute")
    songTimeOffset: float = Field(alias="_songTimeOffset")
    shuffle: float = Field(alias="_shuffle")
    shufflePeriod: float = Field(alias="_shufflePeriod")
    previewStartTime: float = Field(alias="_previewStartTime")
    previewDuration: float = Field(alias="_previewDuration")
    songFilename: str = Field(alias="_songFilename")
    coverImageFilename: str = Field(alias="_coverImageFilename")
    environmentName: str = Field(alias="_environmentName")
    difficultyBeatmapSets: list[MapInfoDiffSet] = Field(alias="_difficultyBeatmapSets")


# ---


class VocabKey(BaseModel):
    color: int
    direction: int
    col: int
    row: int

    class Config:
        frozen = True


# ---


class DatasetMeta(BaseModel):
    bpm: float
    njs: float
    njOffset: float
    difficulty: str
    song_duration_s: float
    total_beats: int
