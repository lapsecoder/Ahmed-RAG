---
title: ResumeForge
category: project
sources:
  - D:/ResumeForge/README.md
  - D:/ResumeForge/docs/architecture.md
  - D:/ResumeForge/docs/phase-7a-copilot-foundation.md
  - D:/ResumeForge/docs/phase-7d-resume-builder.md
  - D:/portfolio/src/content/projects/resumeforge.mdx
---

# ResumeForge

## Summary

ResumeForge is an AI-powered resume and career platform that Ahmed designed and
built, engineered specifically to run at **zero cost**. It uses only free,
open-source and locally hosted tooling.

It lets a user upload a resume, parse it into structured data, analyse it for
quality and applicant tracking system readiness, match it against a job
description, identify skill gaps, and receive AI-powered improvement suggestions
and career recommendations.

- **Live URL:** https://resume-forge-one-zeta.vercel.app
- **Role:** Design and build
- **Source repository:** not publicly linked

## What it does

- Resume upload and parsing for PDF and DOCX into structured data.
- Deterministic job-description parsing.
- Deterministic resume-to-job matching as a baseline.
- A local semantic matching layer using sentence embeddings.
- A hybrid matching engine blending the deterministic and semantic signals.
- Skill gap analysis.
- ATS and quality readiness analysis of the resume.
- A Resume Copilot that proposes controlled, reversible edits.
- A resume builder with interchangeable templates, live preview and PDF export.
- Undo, redo and reset-to-parsed history.

## Technology stack

The ResumeForge tech stack is Next.js with TypeScript and Tailwind on the
frontend, Python with FastAPI on the backend, PostgreSQL with pgvector, and
sentence-transformers with Ollama for the AI and NLP layer.

| Layer | Technology |
| --- | --- |
| Frontend | Next.js (App Router), TypeScript, Tailwind CSS |
| Backend | Python, FastAPI, Pydantic v2, SQLAlchemy 2.0 |
| Database | PostgreSQL with pgvector |
| AI and NLP | sentence-transformers, scikit-learn, Ollama (local LLM) |
| Infrastructure | Docker, Docker Compose, Nginx, Redis, MinIO |
| Tests | pytest, Vitest, Playwright |

## Architecture

Two decoupled services communicating over versioned REST under `/api/v1`, plus
server-sent events for job progress.

- The frontend is a Next.js, TypeScript and Tailwind application.
- The backend is an async FastAPI service.
- Long-running work such as parsing, embedding, matching and LLM calls runs on
  background workers and never blocks the API.
- An Nginx gateway fronts both services for TLS, rate limiting and static
  serving.

## Key technical decisions

- **Deterministic-first, LLM-where-needed.** Parsing and segmentation use
  libraries, rules and embeddings. The LLM is used only for generation and
  summarisation.
- **Provider-agnostic LLM abstraction.** Swappable providers, versioned prompts,
  structured-output validation, retries, token budgets and rule-based fallbacks.
- **Local embeddings.** sentence-transformers with pgvector, so resume personal
  data never leaves his own infrastructure for similarity search.
- **Hybrid semantic matching.** Cosine similarity of resume and job description
  vectors combined with weighted skill and keyword overlap, producing
  explainable section-level breakdowns.
- **Symmetric parsing pipelines.** The resume and job description share one
  parser core.
- **Resume version history.** Each re-parse creates a new version, enabling
  rollback and diffing.
- **UUID primary keys, soft deletes and timestamps** across entities.
- **S3-compatible storage with presigned URLs.** Private buckets, opaque keys,
  server-side validation and short-TTL download links.

## Accounts and privacy

- **Account-free by default.** There is no login, registration, password, email
  verification, JWT or refresh-token flow in the current version.
- **Local-first and privacy-first.** Data is processed on the user's own machine
  or a local self-hosted backend wherever practical, with minimal retention.
- **Optional accounts are deferred**, to be added only if a concrete need for
  cloud sync or persistent server-side user data appears.
- **PostgreSQL and pgvector are retained** as the persistence foundation for
  future core application data and semantic search.

## Matching endpoints

- `POST /api/v1/matching/score` — deterministic baseline, no embeddings.
- `POST /api/v1/matching/semantic` — local sentence-embedding layer, transient.
- `POST /api/v1/matching/hybrid` — blends both at 70 percent deterministic and
  30 percent semantic when both signals are meaningful, with deterministic
  evidence staying authoritative for explicit requirements.
- `POST /api/v1/jobs/parse` — deterministic job-description parsing.

## Implementation status

Implemented: the foundation, the database foundation, resume ingestion,
deterministic resume parsing, the frontend upload-parse-results flow, deterministic
job-description parsing, the deterministic matching baseline, the local semantic
matching layer, the hybrid matching engine, ATS readiness analysis, the Copilot
foundation through its resume improvement and resume builder phases, and PDF
export.

Not yet implemented: accounts and authentication, application database models,
application tracking, and career analytics.

## Known caveats

- The project is proprietary and licensed as undecided, so it is not open source.
- Most features are transient by design, which means work is not persisted across
  sessions in the current version.
- The portfolio case study previously described ResumeForge as storing nothing
  permanently while also listing a database and an application tracker. The
  authoritative statement is the account-free, transient behaviour documented
  above.
