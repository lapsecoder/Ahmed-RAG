---
title: MovieMind
category: project
sources:
  - D:/MovieMind/README.md
  - D:/MovieMind/docs/architecture.md
---

# MovieMind

## Summary

MovieMind is a content-based movie recommendation engine built by Ahmed from a
local TMDb snapshot. A user types a film title, reads its metadata, and sees the
other films the engine considers similar, **with the evidence behind every claim
shown on screen**.

It has no trained model, no database, no accounts, no cloud services and no
per-user data. Similarity is computed at request time over a sparse TF-IDF matrix
that is built once at startup.

- **Role:** Design and build
- **Source repository:** not publicly linked

## What it does

- Search by title across the film snapshot.
- Read details for a selected film: year, runtime, genres and feature counts.
- Get recommendations ranked by cosine similarity, each with the shared terms that
  justify it, plus a tally of candidates examined but rejected.
- Show the honest numbers. Similarity is a raw score and is never presented as a
  percentage. Returned-versus-examined counts are always visible.
- Keep the TMDb attribution notice on screen even when the API is unreachable.

## Selected retrieval configuration

The configuration was chosen by measuring four representations and 31 ablations
end to end.

| Setting | Value |
| --- | --- |
| Representation | D — per-field TF-IDF blocks, L2-normalised, stacked with weights |
| Fields | overview, genres, keywords, cast, director |
| Similarity | cosine, as dot product over L2-normalised vectors |
| Genre weight | 0.44 |
| Evidence floor | minimum shared terms of 3 |
| Catalogue | 5,000 films |
| Feature dimensions | 23,404 |
| Non-zero entries | 226,352 |
| Tie-breaking | ascending TMDb id, so results are bit-reproducible |

## Why the evidence floor exists

The minimum of three shared terms leaves a measured thin-evidence rate of 49.3
percent at the top 10. This is a deliberate trade: fewer results, but each one has
at least three shared terms behind it. A result list shorter than expected means
the catalogue held too few candidates above the evidence floor, not that the
engine stopped searching.

## Architecture

Three structural boundaries are enforced rather than conventional.

- **The engine knows nothing about HTTP.** The recommendation module has no
  FastAPI import. It was complete, measured and tested before an API existed.
- **The service layer is the only caller.** No route reaches into the engine
  directly. The service translates HTTP onto engine calls and imports nothing
  from FastAPI.
- **The frontend never computes a recommendation.** Every network call is built
  in one typed client, and the result order is rendered exactly as received. There
  is no client-side scoring, sorting, filtering, thresholding or percentage
  conversion.

### Backend layers

| Module | Responsibility |
| --- | --- |
| `text.py` | tokenisation, stopwords, English function words, unicode folding |
| `representations.py` | feature blocks for representations A to D, L2 normalisation |
| `recommend.py` | sparse top-k, ranking, evidence and filtering |
| `experiments.py` | the experiment grid and the two selected constants |
| `evaluate.py` | metric definitions |
| `labels.py` | structural proxy labels |
| `pipeline.py` | snapshot to features to processed artefacts, batch and not served |
| `api/` | the FastAPI surface: routes, service, schemas, errors, settings |

### API

Six read-only `GET` endpoints under `/api/v1`. There are no writes and no
database. The API never reads a TMDb credential and makes no outbound network
calls.

### Frontend

React 19 with TypeScript, Vite and Tailwind v4. Components are a search panel,
movie details, recommendations and an about panel. Data fetching uses per-endpoint
hooks with abort handling. Tests use Vitest and Testing Library with fixtures
captured from real backend payloads.

### Configuration integrity

The two constants that decide the output are defined once in the experiments
module and imported elsewhere, so a literal cannot drift from the measured
configuration. A SHA-256 configuration fingerprint covers the representation,
field set, weights, geometry and preprocessing, and is served by the meta endpoint
so a running server can be checked against the reported one.

## Testing

331 backend tests via pytest, and 77 frontend tests. The backend suite is
hermetic apart from the snapshot integration test, which skips when no snapshot is
present. Ruff and strict mypy are both clean.

## Stated limitations

1. The catalogue is 5,000 films, not all of TMDb, so many well-known films return
   no results.
2. There is no relevance ground truth. Scores measure agreement with a structural
   proxy label, not human judgement or user satisfaction.
3. Roughly 49.3 percent of top-10 results sit at the minimum evidence floor.
4. Cast-based search is poorly served, with a recall at 10 of 0.157, so searching
   by an actor's name often fails.
5. There is no user-facing filtering or sorting.
6. It is not a real-time system. The corpus is a static snapshot with no
   incremental update path.
7. No authentication, no database and no persistence. Every request is stateless.
8. It was not tested on physical assistive hardware.
9. The TMDb snapshot is not redistributable. TMDb data is free for non-commercial
   use with mandatory attribution; commercial use requires a separate written
   agreement.

## Data licensing and attribution

MovieMind uses TMDb and the TMDb APIs but is not endorsed, certified or otherwise
approved by TMDb. The attribution notice is served by the API, rendered by the
frontend, and backed by a built-in fallback, so it appears even when the API is
unreachable.

## Performance

Startup loads and vectorises the 5,000-film corpus in about 0.27 seconds. The
processed corpus and matrices are generated at build time and are not committed
to the repository.
