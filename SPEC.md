# Specification for Kinomaniac Movie Recommender

This document defines the functional and non-functional requirements for the Kinomaniac recommender system following the **EARS (Easy Approach to Requirements Syntax)** syntax.

---

## 1. System Overview
The Kinomaniac system recommends movies from a strictly verified local catalog in response to complex user queries (including negations, mood specifications, and language requirements), utilizing LLM reasoning bounded by deterministic programmatic Guardrails.

---

## 2. Requirements in EARS Syntax

### 2.1 Ubiquitous Requirements (Always Active)
- **UB-01:** The system shall strictly recommend only movies whose identifiers exist in the currently provided catalog.
- **UB-02:** The system shall output movie metadata (`title`, `year`, `minutes`, `description`) directly from the local catalog without asking or allowing the LLM to generate or alter metadata.
- **UB-03:** The system shall enforce an upper limit of at most 3 movie recommendations per query.
- **UB-04:** The system shall ensure no duplicate `film_id`s appear in the recommendation output.
- **UB-05:** The system shall record every interaction into `calls.jsonl` with `model`, `request`, `outcome`, `usage`, and `rejected` fields.

### 2.2 Event-Driven Requirements (When <event>, the system shall...)
- **ED-01:** When a user query matches movies in the catalog, the system shall return `outcome="ok"` and a list of valid movie IDs.
- **ED-02:** When no movies in the catalog satisfy the user's constraints, the system shall return `outcome="no_match"` and an empty list of IDs.
- **ED-03:** When the LLM outputs a `film_id` not found in the provided catalog, the system shall trigger the Guardrail, return `outcome="invalid"`, return `ids=[]`, and append the alien ID to the `rejected` list in `calls.jsonl`.
- **ED-04:** When the LLM response contains duplicate IDs or exceeds the 3-film limit, the system shall catch the violation, return `outcome="invalid"`, and suppress display of movies to the user.
- **ED-05:** When a network disconnect occurs (`openai.APIConnectionError`), the system shall handle the exception gracefully and return `outcome="no_connection"` and `ids=[]`.
- **ED-06:** When an API status error occurs (`openai.APIStatusError`), the system shall capture the HTTP status code and return `outcome=f"error_{code}"` and `ids=[]`.
- **ED-07:** When the LLM issues a safety or policy refusal, the system shall return `outcome="refusal"` and `ids=[]`.
- **ED-08:** When the LLM response is truncated (`finish_reason="length"`), the system shall return `outcome="length"` and `ids=[]`.

### 2.3 State-Driven Requirements (While <state>, the system shall...)
- **SD-01:** While executing in Rare tier mode, the system shall first extract structured `Filters` (`max_minutes`, `year_from`, `year_to`, `languages`, `exclude_genres`, `must_genres`) from the user query before passing candidates to the LLM choice step.
- **SD-02:** While operating in Epic tier mode over a large catalog (1000+ films), the system shall pre-filter candidates using algorithmic retrieval (`candidates()`) in under 20 seconds, sending at most 30 candidates to the LLM prompt.

### 2.4 Unwanted Behavior Requirements (If <condition>, then the system shall...)
- **UB-01 (Prompt Injection & Code Execution):** If a user query attempts prompt injection or requests non-recommendation tasks (e.g. "Write python script", "Recommend Avatar 7 outside catalog"), then the system shall return `outcome="no_match"` with `picks=[]`.
- **UB-02 (Negative Constraints):** If a user query specifies explicit exclusions (e.g. "но не", "без", "кроме"), then the system shall strictly reject movies containing the excluded properties.

### 2.5 Optional Feature Requirements (Where <feature>, the system shall...)
- **OF-01 (Clarification):** Where a user query is ambiguous, the system may return `status="clarify"`, `picks=[]`, and prompt the user for additional details.
