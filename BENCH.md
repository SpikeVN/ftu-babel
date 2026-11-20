# FTU500 Benchmark Log

Comprehensive evaluation log for **Babel** (`genie-01-alpha`) vs. baseline (`ftu-assistant`) on the **FTU500** benchmark (`bench/ftu500.json`).

---

## Run #1 — Baseline (`2026-11-20`)

### 1. Run Metadata & Configuration

| Item | Value |
| :--- | :--- |
| **Date** | `2026-11-20` (`09:10:02 +07` – `10:00:20 +07` batch judge) |
| **Dataset** | [`bench/ftu500.json`](bench/ftu500.json) (`500` items across 6 question types and 11 FTU categories) |
| **OpenWebUI Host & Base Model** | `https://swarm.ftds.online` (`Qwen3.6-35B-A3B`) |
| **Evaluated Models** | **`genie-01-alpha`** (native function calling via `tool_ids=["ftu_babel_query"]` against `https://babel.falcolabs.org`; `499` scored, `1` context-overflow collection error `F2-099`)<br>**`ftu-assistant`** (OpenWebUI server-side `ragflow_ftu_context` filter; `500` scored, `0` errors) |
| **Knowledge Base Snapshot** | `10,303` documents (`9,420` embedded), `65,551` parent chunks (avg `2,609` chars), `439,313` child vectors in USearch HNSW (avg `448` chars), `10,054` FTS5 documents |
| **Judge Model & Batch Job** | `deepseek-ai/DeepSeek-V4-Flash-0731` via DeepInfra Batch API (`batch_OvTybUMabNqIVHuqoNG5XRUa`)<br>`999 / 999` completed (`0` failed), `1,258,272` input tokens (`772,864` cached) + `158,510` output tokens = `1,416,782` total tokens |
| **Scoring Rubric** | `final_score = round(0.45 * factual_accuracy + 0.25 * groundedness + 0.15 * completeness + 0.15 * adherence, 2)`<br>`CORRECT`: `>= 0.80` · `PARTIAL`: `0.40 – 0.79` · `INCORRECT`: `< 0.40` |
| **Archived Artifacts** | `bench/archive/2026-11-20_run1_baseline/` (collected answers, batch judge input/output, per-model results, and summaries) |

---

### 2. Overall Scoreboard

| Metric | `genie-01-alpha` (`N=499`) | `ftu-assistant` (`N=500`) | Delta (`Genie - FTU`) |
| :--- | :---: | :---: | :---: |
| **Overall Final Score** | **0.545** *(0.544 over all 500)* | **0.713** | `-0.168` |
| **Factual Accuracy** (`0.45`) | 0.516 | 0.686 | `-0.170` |
| **Groundedness** (`0.25`) | 0.594 | 0.758 | `-0.164` |
| **Completeness** (`0.15`) | 0.502 | 0.667 | `-0.165` |
| **Adherence** (`0.15`) | 0.595 | 0.765 | `-0.170` |
| **CORRECT (`>= 0.80`)** | 230 (`46.1%`) | 315 (`63.0%`) | `-16.9 pp` |
| **PARTIAL (`0.40 – 0.79`)** | 55 (`11.0%`) | 60 (`12.0%`) | `-1.0 pp` |
| **INCORRECT (`< 0.40`)** | 214 (`42.9%`) | 125 (`25.0%`) | `+17.9 pp` |
| **Collection / Runtime Errors** | 1 (`F2-099`, skipped in batch) | 0 | `+1` |

---

### 3. Head-to-Head Comparison (`N=499` common scored items)

- **Score comparison**:
  - `genie-01-alpha` wins: **122** (`24.4%`) — including **46** large-margin wins (`delta >= 0.50`)
  - `ftu-assistant` wins: **242** (`48.5%`) — including **130** large-margin wins (`delta >= 0.50`)
  - Ties: **135** (`27.1%`)
- **Verdict agreement matrix (`CORRECT` vs non-`CORRECT`)**:

| | `ftu-assistant` CORRECT | `ftu-assistant` PARTIAL/INCORRECT | Total |
| :--- | :---: | :---: | :---: |
| **`genie-01-alpha` CORRECT** | 173 (`34.7%`) | 57 (`11.4%`) | 230 |
| **`genie-01-alpha` PARTIAL/INCORRECT** | 142 (`28.5%`) | 127 (`25.5%`) | 269 |
| **Total** | 315 | 184 | 499 |

---

### 4. Latency, Token & Efficiency Telemetry

| Metric | `genie-01-alpha` (`N=499`) | `ftu-assistant` (`N=500`) | Ratio (`Genie / FTU`) |
| :--- | :---: | :---: | :---: |
| **Latency — Avg** | 62.01s | 24.01s | `2.58×` |
| **Latency — Min / p50** | 1.35s / 52.10s | 6.35s / 21.29s | `2.45×` (p50) |
| **Latency — p90 / p95** | 108.04s / 138.04s | 37.91s / 49.96s | `2.85×` (p90) |
| **Latency — p99 / Max** | 262.88s / 342.89s | 69.82s / 87.34s | `3.77×` (p99) |
| **Prompt Tokens — Avg** | 43,438 | 13,297 | `3.27×` |
| **Prompt Tokens — Total** | 21,675,362 | 6,648,279 | `3.26×` |
| **Prompt Tokens — p50 / p90 / Max** | 32,221 / 83,981 / 431,145 | — | — |
| **Prompt Tokens — `>100k` items** | 37 items (`p95=121,113`, `p99=211,829`) | — | — |
| **Completion Tokens — Avg / Total** | 410 / 204,360 | 141 / 70,650 | `2.91×` (avg) |
| **Total Tool Calls** | 1,858 (`3.72` / question) | 0 (inlet filter) | — |

---

### 5. Accuracy Breakdowns

#### 5.1 By Question Type

| Type | Count (`G / F`) | Genie Score | Genie Fact / Grnd | Genie C/P/I | FTU Score | FTU Fact / Grnd | FTU C/P/I | Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `factual` | 249 / 250 | 0.596 | 0.567 / 0.646 | 132 / 19 / 98 | 0.738 | 0.712 / 0.790 | 168 / 22 / 60 | `-0.142` |
| `unanswerable` | 60 / 60 | 0.601 | 0.588 / 0.663 | 31 / 9 / 20 | 0.815 | 0.807 / 0.835 | 44 / 10 / 6 | `-0.214` |
| `paraphrase` | 50 / 50 | 0.547 | 0.534 / 0.532 | 24 / 5 / 21 | 0.759 | 0.754 / 0.728 | 35 / 6 / 9 | `-0.212` |
| `time_sensitive` | 40 / 40 | 0.478 | 0.410 / 0.645 | 16 / 2 / 22 | 0.645 | 0.603 / 0.708 | 22 / 4 / 14 | `-0.167` |
| `multi_hop` | 75 / 75 | 0.463 | 0.431 / 0.481 | 24 / 15 / 36 | 0.568 | 0.521 / 0.636 | 31 / 17 / 27 | `-0.105` |
| `multi_turn` | 25 / 25 | **0.260** | 0.212 / 0.300 | 3 / 5 / 17 | 0.669 | 0.628 / 0.756 | 15 / 1 / 9 | **`-0.409`** |

#### 5.2 By Difficulty

| Difficulty | Count (`G / F`) | Genie Score | Genie Fact / Grnd | Genie C/P/I | FTU Score | FTU Fact / Grnd | FTU C/P/I | Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Dễ` | 50 / 50 | 0.666 | 0.664 / 0.656 | 30 / 6 / 14 | 0.900 | 0.890 / 0.890 | 42 / 6 / 2 | `-0.234` |
| `Trung bình` | 100 / 100 | 0.592 | 0.556 / 0.649 | 47 / 15 / 38 | 0.754 | 0.728 / 0.784 | 64 / 19 / 17 | `-0.162` |
| `Khó` | 349 / 350 | 0.514 | 0.483 / 0.570 | 153 / 34 / 162 | 0.675 | 0.645 / 0.732 | 209 / 35 / 106 | `-0.161` |

#### 5.3 By Expected Behavior

| Expected Behavior | Count (`G / F`) | Genie Score | Genie Fact / Grnd | Genie C/P/I | FTU Score | FTU Fact / Grnd | FTU C/P/I | Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `refuse_no_info` | 30 / 30 | **0.808** | 0.807 / 0.833 | 22 / 5 / 3 | 0.812 | 0.800 / 0.830 | 21 / 7 / 2 | **`-0.004`** |
| `answer` | 439 / 440 | 0.538 | 0.506 / 0.585 | 199 / 46 / 194 | 0.699 | 0.670 / 0.747 | 271 / 50 / 119 | `-0.161` |
| `out_of_scope` | 10 / 10 | 0.398 | 0.380 / 0.500 | 3 / 2 / 5 | 0.783 | 0.780 / 0.790 | 8 / 0 / 2 | `-0.385` |
| `correct_premise` | 20 / 20 | **0.391** | 0.365 / 0.490 | 6 / 2 / 12 | 0.836 | 0.830 / 0.865 | 15 / 3 / 2 | **`-0.445`** |

#### 5.4 By Category

| Category | Count (`G / F`) | Genie Score | Genie Fact / Grnd | Genie C/P/I | FTU Score | FTU Fact / Grnd | FTU C/P/I | Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Lịch sử - Tổ chức - Lãnh đạo` | 61 / 61 | 0.668 | 0.639 / 0.715 | 38 / 4 / 19 | 0.793 | 0.761 / 0.857 | 43 / 7 / 11 | `-0.125` |
| `Chương trình đào tạo` | 43 / 43 | 0.667 | 0.660 / 0.663 | 26 / 5 / 12 | 0.766 | 0.753 / 0.774 | 31 / 3 / 9 | `-0.099` |
| `Học bổng - Học phí` | 28 / 28 | 0.623 | 0.611 / 0.636 | 16 / 4 / 8 | 0.680 | 0.661 / 0.679 | 17 / 3 / 8 | `-0.057` |
| `Hợp tác quốc tế - Kiểm định` | 44 / 44 | 0.580 | 0.552 / 0.605 | 23 / 4 / 17 | 0.807 | 0.789 / 0.841 | 34 / 2 / 8 | `-0.227` |
| `Tin tức - Xếp hạng - NCKH` | 39 / 39 | 0.556 | 0.510 / 0.692 | 16 / 7 / 16 | 0.809 | 0.797 / 0.821 | 29 / 5 / 5 | `-0.253` |
| `Công tác sinh viên - Đời sống` | 30 / 30 | 0.536 | 0.493 / 0.580 | 12 / 5 / 13 | 0.724 | 0.700 / 0.770 | 19 / 5 / 6 | `-0.188` |
| `Tuyển sinh` | 73 / 73 | 0.499 | 0.477 / 0.515 | 31 / 7 / 35 | 0.607 | 0.562 / 0.703 | 38 / 8 / 27 | `-0.108` |
| `Phân hiệu TP.HCM - Quảng Ninh` | 41 / 41 | 0.485 | 0.407 / 0.646 | 13 / 7 / 21 | 0.743 | 0.707 / 0.785 | 25 / 8 / 8 | `-0.258` |
| `Học tập - Quy chế` | 67 / 67 | 0.482 | 0.473 / 0.478 | 28 / 4 / 35 | 0.705 | 0.691 / 0.713 | 42 / 9 / 16 | `-0.223` |
| `Thư viện - CNTT - Cơ sở vật chất` | 43 / 44 | 0.476 | 0.426 / 0.595 | 16 / 5 / 22 | 0.505 | 0.443 / 0.648 | 15 / 8 / 21 | `-0.029` |
| `Điểm rèn luyện` | 30 / 30 | **0.426** | 0.410 / 0.467 | 11 / 3 / 16 | 0.775 | 0.773 / 0.750 | 22 / 2 / 6 | **`-0.349`** |

---

### 6. Genie Tool-Calling Breakdown & Score Correlation

- **Total tool invocations (`1,858` across 499 items, `3.72` avg/item)**:
  - `web_search`: `790` (`42.5%`)
  - `babel_query`: `719` (`38.7%`)
  - `babel_fetch`: `349` (`18.8%`)
- **Calls-per-question distribution**: `0` calls (`32`), `1` (`36`), `2` (`108`), `3` (`86`), `4` (`94`), `5` (`58`), `6+` (`85`, max `16`).
- **Score by tool combination**:

| Tool Combination Used | Count (`N=499`) | Share | Avg Final Score |
| :--- | :---: | :---: | :---: |
| `babel_query` only | 52 | 10.4% | **0.838** |
| `babel_query + babel_fetch` | 142 | 28.5% | **0.686** |
| `babel_query + web_search` | 60 | 12.0% | **0.628** |
| `babel_query + babel_fetch + web_search` | 121 | 24.2% | **0.515** |
| `none` (0 tool calls) | 32 | 6.4% | **0.298** |
| `web_search` only | 92 | 18.4% | **0.233** |

- **Impact of querying Babel first**:
  - `babel_query` called (`n=375`, `75.2%`): **avg `0.643`**, `209` CORRECT (`55.7%`)
  - `babel_query` skipped (`n=124`, `24.8%`): **avg `0.250`**, `21` CORRECT (`16.9%`)

---

### 7. Key Failure Modes & Root-Cause Analysis

1. **Tool-routing collapse — `babel_query` skipped on `24.8%` (`124/499`) of questions**
   - **`web_search`-only (`92` items, avg `0.233`)**: Despite `SYSTEM_PROMPT.md` requiring `babel_query` first, the model skipped Babel completely on `48` `factual`, `13` `time_sensitive`, `11` `multi_hop`, `11` `unanswerable`, `7` `paraphrase`, and `2` `multi_turn` items (`81` `answer`, `6` `correct_premise`, `3` `out_of_scope`, `2` `refuse_no_info`), relying solely on SearXNG web snippets and missing internal FTU rules/numbers.
   - **Zero-tool answers (`32` items, avg `0.298`)**: Heavily concentrated in **`multi_turn` (`16/25` = `64%` called 0 tools; only `7/25` called `babel_query`)**, where the model ignored retrieval and answered from memory or fabricated details/citations (e.g., `MT-004`, `MT-018`, `MT-020`), crashing `multi_turn` accuracy to **`0.260`** vs `0.669`.

2. **Whole-document `babel_fetch` context overflow & token/latency bloat**
   - **Hard context-length crash (`F2-099`)**: Question `"trụ sở 91 chùa láng của ftu diện tích đat bn"` ran `babel_fetch(doc_ids=[8422, 8449, 8445, 8451])`, which expanded prompt context to `131,073+` tokens against Qwen's `131,072`-token ceiling (`OpenWebUI HTTP 400`).
   - Because `stash.db` contains `1,039` documents `>24k` chars (`193` `>100k`, max `1.38M` chars), fetching full documents drove Genie to **3.27× prompt tokens** (`43,438` avg, `37` items `>100k` cumulative prompt tokens, max `431,145`) and **2.58× latency** (`62.01s` vs `24.01s`).

3. **Reranker & search snippet head-truncation**
   - In [`server/services/embedding.py`](server/services/embedding.py) and [`server/services/rerank.py`](server/services/rerank.py), `build_rerank_document(..., max_chars=900)` prepends `Title` + auto-generated `Summary` (~300–450 chars) before slicing at `900` chars, while `search_hybrid` returns only the first `900` chars of the parent chunk. Evidence located in the middle/tail of a 2,600-char parent chunk can be truncated before reaching either `zerank-2` or the LLM.

4. **Premise correction (`correct_premise`) and `out_of_scope` gaps**
   - `genie-01-alpha` scored **`0.391`** on `correct_premise` (vs `0.836` for `ftu-assistant`) and **`0.398`** on `out_of_scope` (vs `0.783`), frequently failing to search/verify false premises (`6/20` skipped `babel_query`). In contrast, `refuse_no_info` is already on par (`0.808` vs `0.812`).

5. **Weakest subject domains & output hygiene**
   - Largest category deficits: `Điểm rèn luyện` (`0.426` vs `0.775`, `-0.349`), `Phân hiệu TP.HCM - Quảng Ninh` (`0.485` vs `0.743`, `-0.258`), `Tin tức - Xếp hạng - NCKH` (`0.556` vs `0.809`, `-0.253`), `Hợp tác quốc tế - Kiểm định` (`0.580` vs `0.807`, `-0.227`), and `Học tập - Quy chế` (`0.482` vs `0.705`, `-0.223`).
   - Output hygiene: `1` leaked `</think>` tag in Genie (`Q017`; `0` in `ftu-assistant`), `0` `<tool>` tag leaks, and `44` false refusals on answerable questions (fewer than `ftu-assistant`'s `84`).

---

### 8. Prioritized Next Improvements

| Priority | Action | Expected Impact |
| :---: | :--- | :--- |
| **P0** | **Chunk-centered reranking + neighbor-chunk fetching** in [`server/services/embedding.py`](server/services/embedding.py), [`server/services/rerank.py`](server/services/rerank.py), [`server/services/stash.py`](server/services/stash.py), and [`oui_tool.py`](oui_tool.py) (center `rerank_text` & snippets on matched child chunk, expose `Section: i/N` + `chunk_index`/`window` in `babel_fetch`, keep whole-doc for short docs, add preamble from chunk `0` for long docs). | Eliminates `>131k` context crashes (`F2-099`), cuts `43k` avg prompt tokens and `62s` latency sharply, and lifts `factual` / `multi_hop` / `Quy chế` recall. |
| **P0** | **Enforce `babel_query` routing on multi-turn & factual turns** in [`SYSTEM_PROMPT.md`](SYSTEM_PROMPT.md) (explicitly clarify that prior assistant turns are untrusted/unverified for new follow-ups so `multi_turn` never skips `babel_query`, and ban `web_search`-only first turns for FTU topics). | Closes the `-0.409` `multi_turn` gap (`16/25` zero-tool failures) and eliminates `92` `web_search`-only misses (`0.233` avg). |
| **P1** | **Verify & expand KB coverage on weakest domains** (`Điểm rèn luyện`, `Phân hiệu TP.HCM - Quảng Ninh`, `Thư viện - CNTT - CSVC`, `Tuyển sinh` 2025–2026) and strengthen `correct_premise` verification prompt rule. | Directly raises the lowest-scoring categories (`0.426–0.499`) and `correct_premise` (`0.391`). |
| **P2** | **Hard-strip stray `</think>` tags** if present in OpenWebUI filter (`strip_tool_and_thinking_traces`) or post-processing. | Cleans up residual `Q017`-style thinking leak. |

---

### 9. Reproducibility Commands

```bash
# Collect answers from both models (resumable, live colored logs)
.venv/bin/python bench/collect_and_batch_judge.py --concurrency 4 --submit

# Check/download & score DeepInfra batch job
.venv/bin/python bench/score_batch_results.py --batch-id batch_OvTybUMabNqIVHuqoNG5XRUa

# Direct online benchmark for genie-01-alpha only
.venv/bin/python bench/run_benchmark.py --concurrency 4
```
