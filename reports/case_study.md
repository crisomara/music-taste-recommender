# Can Audio Features Tell You What Someone Wants to Hear?

*A hybrid recommender case study: content-based similarity vs. collaborative filtering,
tested against real listening data.*

## The question

Netflix and Spotify's actual recommenders are built on collaborative filtering — "people
who listened to what you listened to also played this." That needs a record of what
people actually did. Audio-feature-based recommenders — "this sounds like that" — need
no listening history at all. Which one actually works, and does combining them help?

This project builds both, evaluates them against a real held-out test of listening
behavior, and reports what happened — including the parts that didn't go as expected.

## The data

- **Spotify Tracks Dataset**: 114,000 tracks (77,134 after deduplication), with audio
  features (danceability, energy, valence, tempo, acousticness, etc.), genre, and
  popularity. A curated, genre-queried sample — not an exhaustive catalog.
- **LastFM-1K**: 13.4 million real listening events from ~792 users, 2005–2013 (bulk of
  activity 2008–2009), covering 740,932 distinct tracks.

The two datasets share no common ID. Attaching audio features to listening events
required a name-based join (artist + track name, normalized). That join is lossy by
construction, and the loss turned out to be the first real finding.

### The join coverage number is low — and here's why

Only **8.87% of listening events** (1,185,414 of 13,365,803) and **1.29% of unique
tracks** (9,544 of 740,932) matched a Spotify Tracks Dataset entry.

Before treating that as a real finding, I ruled out a join bug: artist-only overlap
(ignoring track name) was **7.40%** — six times higher than the track-level match rate.
That gap confirms the low coverage isn't a broken match, it's a real data-integration
limitation: the Spotify Tracks Dataset is a curated ~114k-track sample, not an exhaustive
catalog, so even when an artist is present, the *specific* tracks a LastFM-1K user played
in 2008–2009 rarely happen to be among the tracks that dataset sampled for that artist.
Temporal mismatch (2008–2009 listening vs. a modern snapshot) compounds it further.

**Why this matters for everything downstream:** the content-based layer and the hybrid's
cold-start fallback can only ever operate on that matched ~1.3% of the catalog. That
constraint shows up directly in the results below.

## Two lenses, compared

Both approaches were evaluated with a temporal per-user split (train on each user's
earlier plays, hold out their most recent ones — a random split would leak future
listens into training) and standard precision@10 / recall@10, against a trivial
popularity baseline (recommend the same top-10 overall to everyone) as the floor.

| Method | Precision@10 | Recall@10 | Catalog coverage | Diversity |
|---|---|---|---|---|
| Popularity baseline | **0.0965** | 0.0011 | ~0% | 1.01 |
| Content-based only | 0.0023 | ~0.0000 | 0.93% | 0.08 |
| Collaborative (ALS) | 0.0880 | **0.0013** | 0.52% | 0.61 |
| Hybrid | 0.0880 | **0.0013** | 0.52% | 0.61 |

*(Full-catalog evaluation: 657,551 items, 792 held-out users, k=10.)*

**Popularity narrowly wins precision; ALS wins recall.** Recommending the same globally
popular tracks to everyone gets a slightly higher hit rate per recommendation slot — but
personalization (ALS) finds meaningfully more of what each user *actually* held out
(19% higher recall), because it isn't stuck recommending the same 10 items to a catalog
of 657,551. That's the real tradeoff textbooks describe: precision vs. breadth of
relevant discovery, not a simple "collaborative wins."

**Content-based alone performed far worse than expected** — but this is a coverage
artifact, not a fundamental weakness. With only ~1.3% track coverage, the vast majority
of the 792 held-out users have no audio-feature-matched track to seed a recommendation
from at all, so they contribute zero precision by construction. The 0.23% precision this
method shows is really "content-based similarity, evaluated on a population it can barely
reach" — not a fair test of the method itself.

## The hybrid — and the twist

The hybrid was designed to cascade: use collaborative filtering when a user has enough
history (≥5 interactions), fall back to content-based similarity otherwise.

**Result: hybrid and pure-ALS numbers are identical.** This wasn't a bug — it's a real
methodological finding. The held-out evaluation population, by construction, only
includes users with ≥5 interactions (the temporal split's minimum threshold). That's
*exactly* the population for which the hybrid never triggers its content-based fallback.
The cold-start path exists in the code and works (verified separately in notebook 04),
but this particular evaluation slice can't demonstrate its value, because it excludes the
very users — genuine cold-start cases — the fallback exists for.

**What this means:** to actually prove the hybrid earns its complexity, the right next
evaluation is a cold-start-specific slice (new/low-activity users deliberately included,
not filtered out) — see Limitations & Next Steps.

## The taste map

A UMAP projection of the standardized audio-feature space (15,000-track sample, top 12
genres) — see `figures/02_taste_map.png`. Genres form visually distinct regions,
confirming the audio features carry real discriminative signal, not noise.

**What actually drives that separation surprised me.** Going in, the hypothesis was that
danceability/energy/valence would dominate genre clustering. They didn't. Ranked by
variance across genre means:

| Feature | Variance across genre means |
|---|---|
| **tempo** | **77.24** |
| **loudness** | **11.81** |
| key | 0.063 |
| acousticness | 0.054 |
| instrumentalness | 0.045 |
| energy | 0.029 |
| valence | 0.021 |
| danceability | 0.013 |
| liveness | 0.006 |

Tempo and loudness dominate by more than an order of magnitude over everything else,
including the features I expected to matter most. In hindsight this makes sense — tempo
and loudness are genre-defining almost by convention (a genre's characteristic BPM range
and production loudness are often more consistent within-genre than "how danceable" or
"how positive" a track feels) — but it's a real, testable claim the data confirmed and
the intuition didn't predict correctly upfront.

**Popularity doesn't have a strong audio-feature signature either.** Correlation between
popularity and any single audio feature stayed under |0.2| across the board —
instrumentalness was the strongest single predictor, and it's negative (r = -0.198).
Loudness and danceability show the strongest positive correlation (~0.09–0.095), but
that's weak. Whatever makes a track popular, it isn't well captured by these audio
features alone — a legitimate limit on how far "this sounds like a hit" reasoning can go.

## Limitations & next steps

- **Join coverage (1.29%)** is the binding constraint on the content-based and hybrid
  layers. A production version would need a real shared ID (ISRC, Spotify track ID
  natively in the interaction log) rather than name-based matching.
- **The hybrid's cold-start value is unproven by this evaluation**, precisely because the
  held-out population excludes the users it's meant to help. Next step: a
  cold-start-specific eval slice (include low-activity users, measure the hybrid against
  content-only and a "no personalization" baseline specifically for that group).
- **LastFM-1K is 2008–2009 data**; the Spotify catalog is a modern snapshot. Temporal
  mismatch is folded into the coverage number above but not isolated from the "curated
  sample" effect — a dataset with continuous listening history to the present would
  separate the two causes more cleanly.
- **ALS factors/regularization/iterations were not tuned** — a grid search over these
  (and over the min-interactions-for-CF threshold) is the natural next step before
  reading too much into the exact precision/recall gap.
