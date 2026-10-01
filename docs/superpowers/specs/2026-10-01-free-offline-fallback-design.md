# Free Offline Fallback Design

## Goal

Keep the public Streamlit application useful after the Gemini free-tier quota is exhausted, without paid infrastructure and without requiring the owner's laptop to remain online.

## Confirmed Failure

The deployed application loads successfully, but Gemini generation returns HTTP 429 after the project's `gemini-3.8-flash` free-tier allowance of 20 generation requests per day is consumed. The current application converts that provider error into an error banner and produces no answer. Retrying this daily-quota failure cannot succeed before the quota window resets.

## Selected Approach

Use a hybrid answer pipeline:

1. Attempt the existing Gemini-backed retrieval and grounded generation path.
2. If generation fails because of a rate-limit or quota response, reuse the retrieved chunks to create an extractive answer locally.
3. If Gemini query embedding is also unavailable, retrieve chunks locally from the tracked `index/chunk_metadata.json` artifact using deterministic lexical scoring, then create the same kind of extractive answer.
4. Continue using Gemini automatically when it becomes available again; no persistent circuit state is required.

This preserves the higher-quality Gemini experience while making the deployed application functional without paid API usage.

## Alternatives Considered

### Switch to another Gemini model

This could temporarily provide a separate quota pool, but it would remain dependent on changing provider limits and would not guarantee a continuously free application.

### Use offline mode for every request

This would eliminate runtime API dependency, but it would unnecessarily reduce answer fluency and semantic retrieval quality while Gemini quota is available.

### Selected hybrid mode

The hybrid mode provides the best user experience within the zero-cost constraint: Gemini when available and deterministic local behavior when it is not.

## Components

### Provider error classification

A small classifier will distinguish quota/rate-limit failures from authentication, index-corruption, citation-validation, and unknown failures. Only quota/rate-limit failures trigger fallback. Authentication and integrity errors remain visible as safe application errors so configuration problems are not silently hidden.

Daily-quota errors will not be retried by application code. Retryable short-lived provider failures may retain bounded retries where appropriate.

### Local lexical retrieval

The local retriever will load the existing tracked chunk metadata, tokenize the question and chunk fields, remove non-informative terms, and assign deterministic relevance scores. It will return up to the existing fixed Top-K count with the same provenance fields expected by downstream citation handling.

The algorithm will be deliberately small and dependency-free. Ranking ties will be resolved by stable metadata order. If no meaningful term overlap exists, retrieval will return no supported result.

### Extractive answer construction

The extractor will split retrieved chunk text into sentences, score sentences against question terms, and select a small number of the highest-scoring non-duplicate sentences. Selected sentences will remain verbatim source excerpts and will be associated with controlled source identifiers such as `[C1]`.

The existing deterministic citation resolver will convert those controlled identifiers into trusted filename and physical-page references. If no sentence has meaningful support, the pipeline will return the existing exact insufficient-context fallback sentence.

To avoid misrepresentation, the result will include an explicit answer-mode field indicating Gemini or offline extractive generation.

### Orchestration

The application-facing function will coordinate the modes:

- Normal path: current semantic retrieval, Gemini generation, and citation resolution.
- Generation-quota path: semantic retrieval results plus local extraction.
- Embedding-quota path: local lexical retrieval plus local extraction.
- Unsupported question: exact insufficient-context fallback with no citations.
- Non-quota failure: preserve the current safe error behavior.

No API key or network call is required by the fully offline path.

### Streamlit presentation

Gemini answers retain the existing presentation. Offline answers display a concise notice that an extractive fallback was used because the AI service limit was reached. Source cards remain unchanged and continue to show document, physical page, publisher, and source URL.

The notice will not claim that an offline extract is equivalent to a generated synthesis.

## Data Flow

```text
question
  -> Gemini semantic retrieval
     -> success -> Gemini generation
        -> success -> deterministic citation resolution -> UI
        -> quota -> local extraction -> deterministic citation resolution -> UI
     -> quota -> local lexical retrieval -> local extraction
        -> deterministic citation resolution -> UI
     -> other error -> safe UI error
```

## Error Handling

- Trigger fallback only for confirmed HTTP 429, quota, rate-limit, or resource-exhausted provider failures.
- Do not expose raw provider responses, API keys, local paths, or tracebacks.
- Fail closed when tracked index metadata is missing, malformed, or incompatible.
- Return the exact existing insufficient-context sentence when local evidence is inadequate.
- Preserve current authentication and citation-integrity error behavior.

## Testing

Tests will be written before implementation and will verify:

- quota classification does not misclassify authentication or unknown errors;
- daily-quota failures are not retried wastefully;
- lexical ranking is deterministic and uses tracked metadata;
- extractive answers contain only source sentences and controlled source IDs;
- insufficient overlap returns the exact fallback sentence;
- generation quota switches to extraction using already-retrieved chunks;
- embedding quota switches to fully local retrieval and extraction;
- citation resolution still removes controlled IDs and renders trusted page citations;
- Streamlit labels offline answers and retains source cards;
- ordinary non-quota failures still produce safe errors;
- the complete offline test suite and application smoke tests pass.

## Deployment and Verification

After tests and a production build-equivalent import check pass, changes will be pushed to GitHub `main`, allowing Streamlit Community Cloud to redeploy automatically. Verification will include loading the public URL and submitting an example question while the Gemini daily quota is exhausted. Success requires a labeled extractive answer with valid source cards instead of the current quota error.

## Cost and Operational Constraints

- Streamlit Community Cloud remains the host.
- The fallback uses only tracked repository artifacts and application CPU/memory.
- The owner's laptop is not involved after deployment.
- No paid Gemini billing or additional hosted service is required.
