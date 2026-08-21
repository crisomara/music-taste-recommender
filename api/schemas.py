"""Pydantic response models for the recommendation API."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    known_users: int
    known_tracks: int


class SampleUser(BaseModel):
    user_id: str


class SampleTrack(BaseModel):
    track_id: str
    track_name: str
    artist: str


class Recommendation(BaseModel):
    join_key: str
    track_name: str
    artist: str


class RecommendResponse(BaseModel):
    user_id: str
    seed_track_id: str | None
    recommendations: list[Recommendation]


class SimilarTrack(BaseModel):
    track_id: str
    track_name: str
    artist: str
    similarity: float


class SimilarResponse(BaseModel):
    seed_track_id: str
    seed_track_name: str
    seed_artist: str
    similar: list[SimilarTrack]
