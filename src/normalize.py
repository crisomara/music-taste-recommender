"""Track/artist name normalization for joining Spotify Tracks Dataset (no user data)
against LastFM-1K (user listening events, no Spotify track IDs).

The join is name-based and therefore lossy by construction — remixes, live versions,
"feat." credits, and typos will not match. That loss rate is itself a reported finding
(see notebooks/01_data_audit.ipynb), not something to hide by over-fuzzing the match.
"""
import re
import unicodedata

_PAREN_RE = re.compile(r"[\(\[][^\)\]]*[\)\]]")
_FEAT_RE = re.compile(r"\b(feat\.?|featuring|ft\.?)\b.*$", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation, drop parenthetical and feat. clauses."""
    if not isinstance(name, str) or not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = _PAREN_RE.sub(" ", s)
    s = _FEAT_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def primary_artist(artists_field: str) -> str:
    """Spotify's `artists` column is ';'-separated for collabs; take the first credited."""
    if not isinstance(artists_field, str) or not artists_field:
        return ""
    return artists_field.split(";")[0].strip()


def join_key(artist: str, track: str) -> str:
    return f"{normalize_name(artist)}|||{normalize_name(track)}"
