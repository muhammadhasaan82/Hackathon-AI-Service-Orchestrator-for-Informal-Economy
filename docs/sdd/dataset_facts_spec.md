# Dataset Facts Spec

## Responsibilities

`DatasetFacts` is the shared read-optimized service for facts derived from provider data. It answers coverage, count, availability, category, city, area, and future analytics questions without hardcoded business values.

## Ownership Boundaries

- `DatasetFacts` owns startup aggregation, canonical lookup maps, fuzzy matching, confidence scores, and coverage snapshots.
- Provider CSV, Weaviate metadata, PostgreSQL, Redis, and future onboarding feeds own raw facts.
- FAQ and orchestration layers consume facts but do not compute or hardcode them.
- Rust CAG should eventually own cached aggregation and hot-reload helpers, with Python retaining a thin adapter.

## Data Contracts

Fact methods return canonical values and confidence metadata:

```json
{
  "matched": true,
  "value": "Plumber",
  "score": 0.94,
  "source": "exact | fuzzy | semantic | none",
  "candidates": [
    {"value": "Plumber", "score": 0.94}
  ],
  "ambiguous": false
}
```

Aggregate methods return plain deterministic values:

- `get_available_cities() -> list[str]`
- `get_available_categories() -> list[str]`
- `get_areas_by_city(city) -> list[str]`
- `get_total_provider_count() -> int`
- `get_category_city_coverage(category) -> dict[str, int]`
- `get_availability_summary(city=None, category=None, area=None) -> dict[str, int]`

## Config Contracts

Thresholds are read from YAML config or caller-provided values:

- category fuzzy threshold
- city fuzzy threshold
- area fuzzy threshold
- ambiguity threshold
- ambiguity margin
- hot reload enabled flag
- optional Redis cache namespace

## Response Semantics

Dataset facts do not produce final user prose except through template consumers. They return canonical facts and confidence metadata so FAQ, discovery, and routing can explain why a route was selected.

## Fallback Behavior

- Exact canonical match is preferred.
- Fuzzy match is allowed only above threshold.
- Ambiguous close matches are returned as ambiguity metadata, not guessed.
- If the backing dataset is empty or unavailable, callers receive empty lists/counts and observable errors.

## Latency Expectations

All common fact reads are O(1) dictionary lookups after startup. Fuzzy matching over service and city catalogs should remain below 50 ms because catalogs are small.

## CPU Constraints

Aggregations are computed once during startup. Request-time work must avoid dataframe scans unless a hot reload explicitly refreshes the snapshot.

## Extensibility Expectations

The service must support replacing the CSV source with PostgreSQL, Redis, Weaviate metadata, or onboarding events without changing FAQ or orchestrator contracts.

## Failure Handling

- Missing columns fail startup with clear logs.
- Unknown facts return empty deterministic responses, not fabricated values.
- Hot reload failures keep the last good snapshot.

## Observability Requirements

Each high-level consumer trace should record:

- `DatasetFacts_used`
- fact method name
- match source
- confidence score
- ambiguity status
- snapshot version or loaded timestamp

