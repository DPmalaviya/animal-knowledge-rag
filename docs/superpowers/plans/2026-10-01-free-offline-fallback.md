# Free Offline Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the deployed Streamlit app answering from its tracked corpus when Gemini generation or embedding quota is exhausted, without paid services or a running laptop.

**Architecture:** Add a provider-error classifier and a focused offline fallback module. A hybrid orchestration function will attempt semantic retrieval and Gemini generation first, reuse semantic results for extractive generation after generation quota exhaustion, and fall back to deterministic lexical retrieval if embedding quota is unavailable. Existing citation resolution remains the trust boundary, and Streamlit only adds a visible mode notice.

**Tech Stack:** Python 3.13, standard library (`json`, `math`, `pathlib`, `re`), existing Google Gen AI SDK, FAISS metadata artifacts, `unittest`, Streamlit `AppTest`.

---

## File Structure

- Create `src/provider_errors.py`: classify API failures without importing UI or provider SDK types.
- Create `src/offline_fallback.py`: deterministic tokenization, lexical retrieval, sentence extraction, and hybrid orchestration.
- Create `tests/test_provider_errors.py`: quota classification and retry-policy tests.
- Create `tests/test_offline_fallback.py`: lexical retrieval, extraction, orchestration, and citation integration tests.
- Modify `src/retrieval.py`: stop retrying confirmed quota exhaustion while preserving transient retries.
- Modify `src/generation.py`: stop retrying confirmed daily quota exhaustion while preserving transient retries.
- Modify `app.py`: use hybrid orchestration and label offline extractive results.
- Modify `tests/test_retrieval.py`, `tests/test_generation.py`, and `tests/test_app.py`: regression tests for retry and UI behavior.
- Modify `README.md`: document hybrid behavior, free-tier limitation, and runtime modes.

### Deterministic Defaults

- Token regex: lowercase alphanumeric terms matching `[a-z0-9]+`.
- Stop words: a fixed local set of common English function words; terms shorter than three characters are ignored unless numeric.
- Lexical score: sum of query-term frequencies in `title` weighted by 3 plus frequencies in `text`, normalized by `sqrt(total candidate tokens)`.
- Minimum retrieval overlap: at least one non-stop query term; otherwise return no records.
- Tie break: descending score, then ascending tracked `faiss_id`, which is the canonical metadata-list order in the stored artifact.
- Sentence split: punctuation (`.`, `!`, `?`) followed by whitespace, while preserving original sentence text.
- Sentence selection: up to three unique sentences, maximum 600 rendered source characters total, ordered by score then retrieval rank and sentence position. Skip an individual sentence that exceeds the remaining budget; never truncate source text.
- Deduplication: compare normalized lowercase whitespace-collapsed sentence text.
- Exact fallback: `I don't have enough information in the provided sources to answer that question.`
- Top-K: 4.

### Task 1: Provider Error Classification and Retry Policy

**Files:**
- Create: `src/provider_errors.py`
- Create: `tests/test_provider_errors.py`
- Modify: `src/retrieval.py:147-175`
- Modify: `src/generation.py:275-308`
- Test: `tests/test_retrieval.py`
- Test: `tests/test_generation.py`

- [ ] **Step 1: Write failing classifier tests**

Cover status-code 429, `RESOURCE_EXHAUSTED`, `quota exceeded`, structured daily quota ID `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, ordinary 503, 401, and unrelated `ValueError`. Require `is_quota_error()` to identify quota/rate-limit failures and `is_daily_quota_error()` to identify only non-recoverable daily quota responses.

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python3 -m unittest tests.test_provider_errors -v`

Expected: import failure because `src.provider_errors` does not exist.

- [ ] **Step 3: Implement minimal classifiers**

Implement pure helpers that inspect exception chains, numeric `code`/`status_code`, and lowercase text. Daily classification must require a daily/per-day quota marker or the known quota ID; a bare 429 remains quota-triggering for fallback but may retain bounded retry behavior.

- [ ] **Step 4: Run classifier tests and verify GREEN**

Run: `python3 -m unittest tests.test_provider_errors -v`

Expected: all classifier tests pass.

- [ ] **Step 5: Add failing retry regression tests**

Add tests proving a daily-quota exception causes exactly one SDK call and zero sleeps in both `embed_query()` and `generate_grounded_answer()`. Preserve existing tests proving 429 followed by success retries once.

- [ ] **Step 6: Run retry tests and verify RED**

Run: `python3 -m unittest tests.test_retrieval tests.test_generation -v`

Expected: the new daily-quota tests fail because current code retries.

- [ ] **Step 7: Apply the classifier to both retry loops**

Before generic transient retry handling, re-raise a wrapped `ValueError` immediately when `is_daily_quota_error(exc)` is true. Do not change authentication or ordinary transient behavior.

- [ ] **Step 8: Run focused tests and commit**

Run: `python3 -m unittest tests.test_provider_errors tests.test_retrieval tests.test_generation -v`

Expected: all tests pass.

Commit: `git commit -m "fix: stop retrying exhausted daily Gemini quota"`

### Task 2: Deterministic Local Retrieval

**Files:**
- Create: `src/offline_fallback.py`
- Create: `tests/test_offline_fallback.py`

- [ ] **Step 1: Write failing tokenization and lexical-ranking tests**

Use synthetic metadata to verify stop-word removal, title weighting, stable tie ordering, Top-K truncation, required overlap, preservation of the 12-field retrieval contract, non-mutation, missing/malformed JSON rejection, and a real-artifact smoke test against `index/chunk_metadata.json`.

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m unittest tests.test_offline_fallback.TestLocalLexicalRetrieval -v`

Expected: import or missing-function failures.

- [ ] **Step 3: Implement local retrieval minimally**

Implement `tokenize_meaningful(text)`, `rank_metadata(question, metadata, top_k)`, and `retrieve_locally(question, index_dir, top_k)`. Return ranked copies containing the fields required by `build_source_map()` with finite float `similarity_score` values.

- [ ] **Step 4: Run retrieval tests and verify GREEN**

Run: `python3 -m unittest tests.test_offline_fallback.TestLocalLexicalRetrieval -v`

Expected: all local retrieval tests pass.

- [ ] **Step 5: Commit**

Commit: `git commit -m "feat: add deterministic offline corpus retrieval"`

### Task 3: Extractive Answer Construction

**Files:**
- Modify: `src/offline_fallback.py`
- Modify: `tests/test_offline_fallback.py`

- [ ] **Step 1: Write failing extraction tests**

Verify sentence scoring, stable tie order, normalized deduplication, three-sentence and 600-character bounds, verbatim source preservation, controlled IDs, exact insufficient-context fallback, and no mutation of retrieval records.

- [ ] **Step 2: Run extraction tests and verify RED**

Run: `python3 -m unittest tests.test_offline_fallback.TestExtractiveAnswer -v`

Expected: missing-function failures.

- [ ] **Step 3: Implement minimal extraction**

Implement `build_extractive_rag_result(question, retrieval_records)`. For supported records, build the source map with the existing validator, select supported sentences, append each selected sentence's controlled ID, and return the existing Stage 9 result shape plus `answer_mode="offline_extractive"`. Retain the citation resolver's expected generation-model field and use `answer_mode` as the mode discriminator. If no record or sentence meets the support threshold, do not send an empty source map through `resolve_rag_citations()`; instead return a dedicated final-result helper containing the exact insufficient-context text, empty citation collections, zero counts, `is_fallback=True`, and `answer_mode="offline_extractive"`.

- [ ] **Step 4: Add citation integration test**

Pass supported extractive results through `resolve_rag_citations()` and assert controlled markers disappear, filenames/pages render, and citations are structured. Assert the dedicated unsupported final result has the exact fallback text and no sources without invoking the resolver on an empty source map.

- [ ] **Step 5: Run extraction and citation tests**

Run: `python3 -m unittest tests.test_offline_fallback tests.test_citations -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

Commit: `git commit -m "feat: add source-grounded extractive answers"`

### Task 4: Hybrid Orchestration

**Files:**
- Modify: `src/offline_fallback.py`
- Modify: `tests/test_offline_fallback.py`

- [ ] **Step 1: Write failing orchestration tests**

Inject retrieval and generation callables to prove four paths: normal Gemini success, generation quota reuses already-retrieved records, embedding/retrieval quota invokes local retrieval, and non-quota errors propagate. Assert the returned `answer_mode` is `gemini` or `offline_extractive` and no second Gemini call occurs on fallback paths.

- [ ] **Step 2: Run orchestration tests and verify RED**

Run: `python3 -m unittest tests.test_offline_fallback.TestHybridOrchestration -v`

Expected: missing-function failures.

- [ ] **Step 3: Implement `answer_with_free_fallback()`**

Validate the question and tracked artifacts first. Attempt semantic retrieval; on quota use local retrieval. After successful semantic retrieval, attempt Gemini generation; on quota build the extractive result from those same records. Resolve citations exactly once for supported answers, then explicitly attach `answer_mode` to the final Stage 10 dictionary because `resolve_rag_citations()` intentionally returns a fixed field set. Return the dedicated final fallback dictionary directly when local retrieval/extraction has no support. Propagate authentication, integrity, and unknown errors.

- [ ] **Step 4: Run orchestration tests and verify GREEN**

Run: `python3 -m unittest tests.test_offline_fallback -v`

Expected: all offline fallback tests pass.

- [ ] **Step 5: Commit**

Commit: `git commit -m "feat: orchestrate Gemini and free offline fallback"`

### Task 5: Streamlit Mode Disclosure

**Files:**
- Modify: `app.py:12-16, 164-166, 236-276`
- Modify: `tests/test_app.py:31-88, 224-256`

- [ ] **Step 1: Write failing UI tests**

Add `answer_mode` to synthetic fixtures. Verify Gemini answers show no fallback notice; offline extractive answers show a clear informational notice and retain answer/source cards; insufficient-context fallback still shows no source cards.

- [ ] **Step 2: Run UI tests and verify RED**

Run: `python3 -m unittest tests.test_app -v`

Expected: offline notice assertion fails.

- [ ] **Step 3: Switch the default backend and render the notice**

Import `answer_with_free_fallback` as the default `render_app()` backend. When `answer_mode == "offline_extractive"`, render: `Gemini's free limit is currently reached, so this answer was extracted directly from the indexed sources.` Do not display raw errors or imply generative synthesis.

- [ ] **Step 4: Run UI tests and verify GREEN**

Run: `python3 -m unittest tests.test_app -v`

Expected: all AppTest tests pass.

- [ ] **Step 5: Commit**

Commit: `git commit -m "feat: show offline fallback mode in Streamlit"`

### Task 6: Documentation and Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update documentation**

Document the 20-request daily generation allowance observed for the configured model as a changeable provider limit, the hybrid runtime flow, the offline label, the fact that fallback runs on Streamlit Cloud, and that the laptop may be off. Update the test-count badge only after the final verified count is known.

- [ ] **Step 2: Run static and import checks**

Run: `python3 -m compileall -q app.py src tests`

Run: `git diff --check`

Expected: both exit successfully.

- [ ] **Step 3: Run the complete offline suite**

Run: `python3 -m unittest discover tests -v`

Expected: zero failures and zero errors.

- [ ] **Step 4: Run a forced-quota application smoke test**

Use an injected quota-raising fake client or backend; do not consume live Gemini quota. Verify an example bald-eagle question produces an offline extractive answer with citations.

- [ ] **Step 5: Review the diff and commit docs**

Run: `git status --short` and `git diff --stat HEAD~5..HEAD`

Commit: `git commit -m "docs: explain free offline fallback"`

### Task 7: Publish and Verify Production

**Files:** No additional source changes expected.

- [ ] **Step 1: Re-run completion verification immediately before publishing**

Run: `python3 -m unittest discover tests -v && python3 -m compileall -q app.py src tests && git diff --check`

Expected: zero failures/errors and successful exits.

- [ ] **Step 2: Push `main` to GitHub**

Run: `git push origin main`

Expected: remote advances to the verified local commit.

- [ ] **Step 3: Wait for Streamlit redeployment and inspect the public URL**

Open the existing Streamlit URL and wait for the app to wake and redeploy.

- [ ] **Step 4: Submit one example question while quota remains exhausted**

Expected: no quota error banner; a labeled extractive answer appears with at least one trusted source card and physical-page citation.

- [ ] **Step 5: Report operational caveats**

Report the live URL, verified mode, test count, commits pushed, and the distinction between free hosting sleep/wake behavior and Gemini quota behavior.
