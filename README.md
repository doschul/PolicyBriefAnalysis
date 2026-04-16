# Policy Brief Analysis Pipeline

A reproducible research pipeline for automated analysis of policy brief PDFs. The pipeline extracts structured metadata, detects theoretical policy-instrument frames, and identifies policy recommendations using a broad-content, multi-pass LLM architecture with lightweight deterministic guardrails.

## Overview

The pipeline processes PDF policy briefs through four LLM passes, bookended by deterministic extraction and validation:

1. **PDF extraction** — text and metadata from PDF files (PyMuPDF / pypdf fallback)
2. **Pass A: Front-matter** — title, authors, affiliations, funding via LLM (first 3 + last page)
3. **Pass B: Structural core** — problem statement, solutions, implementation, narrative hook via LLM (sampled pages)
4. **Pass C: Recommendations** — broad-content LLM extraction over page-window chunks, then deduplication
5. **Pass D: Frames** — all five policy-instrument frames assessed in a single LLM call per chunk

Deterministic rails handle PDF parsing, metrics, reference-boundary exclusion, evidence verification, normalization, deduplication, and output generation. Each stage is independently toggleable. If any stage fails at runtime, the pipeline logs a warning and continues (graceful degradation).

## Quick Start

### Installation

```bash
git clone <repository-url>
cd PolicyBriefAnalysis
python -m venv .venv
.venv\Scripts\activate       # Linux/macOS: source .venv/bin/activate
pip install -e .
```

### Configuration

```bash
# Set your OpenAI API key
# Option 1: environment variable
set OPENAI_API_KEY=your-key-here          # Windows
export OPENAI_API_KEY=your-key-here       # Linux/macOS

# Option 2: .env file in project root
echo OPENAI_API_KEY=your-key-here > .env
```

### Run

```bash
# Process all PDFs in a directory
python cli.py extract --input_dir ./data/policy_briefs --output_dir ./output --config ./config

# Force reprocessing (ignore cache)
python cli.py extract --input_dir ./data/policy_briefs --output_dir ./output --config ./config --force_reprocess

# Validate configuration only
python cli.py validate-config --config ./config
```

### Programmatic usage

```python
from src.policybrief.pipeline import PolicyBriefPipeline
from pathlib import Path

pipeline = PolicyBriefPipeline(
    config_dir=Path("config"),
    output_dir=Path("output"),
    max_workers=1
)

pdf_files = list(Path("data/policy_briefs").glob("*.pdf"))
results = pipeline.process_documents(pdf_files)

summary = pipeline.compute_extraction_summary(results["processed"])
print(summary)
```

## Requirements

- **Python 3.11+**
- **OpenAI API key** with access to a model supporting structured outputs (e.g. `gpt-4o-mini`, `gpt-4o`)
- PDF policy briefs for analysis

### Dependencies

PyMuPDF (fitz), pypdf, pandas, pyarrow, click, pyyaml, openai, pydantic, tenacity, textstat, python-dotenv

## Architecture

```
PolicyBriefAnalysis/
├── cli.py                          # Click CLI (extract, validate-config)
├── config/
│   ├── config.yaml                 # Pipeline settings and module switches
│   ├── frames.yaml                 # Policy-instrument frame definitions
│   └── enums.yaml                  # Controlled vocabularies
├── src/policybrief/
│   ├── pipeline.py                 # Orchestrator — Pass A (front-matter) & B (structural core) inline
│   ├── pdf_extractor.py            # PDF text/metadata extraction (PyMuPDF + pypdf fallback)
│   ├── recommendation_extractor.py # Pass C — broad-content recommendation extraction + dedup
│   ├── frame_detector.py           # Pass D — all-frames-at-once chunk-based detection
│   ├── llm_client.py               # OpenAI client with schema patching and retry
│   ├── metrics_calculator.py       # Document metrics (readability, structure)
│   ├── models.py                   # Pydantic V2 data models and enums
│   └── utils.py                    # Shared helpers (I/O, config, text cleaning)
├── tests/                          # pytest suite (127 tests)
└── output/                         # Generated outputs
    ├── documents.csv
    ├── frames.csv
    ├── recommendations.csv
    ├── structural_core.csv
    └── audit/                      # Per-document JSON audit trails
```

## Pipeline Stages

### 1. PDF Extraction (`pdf_extractor.py`)

Extracts per-page text and PDF-level metadata using PyMuPDF (primary) or pypdf (fallback).

- Per-page text with word/character counts
- PDF metadata: title, author, subject, creator, dates
- Layout-preserving text extraction via text-block sorting
- Scanned-document detection via text-density heuristics (< 50 chars/page average)
- Text extraction quality score (0–1)
- SHA-256 file hashing for incremental processing

### 2. Pass A — Front-Matter (inline LLM in `pipeline.py`)

Extracts content-derived metadata from the first few pages via a single LLM call:

- Document title, author names
- Institutional affiliations, email addresses, URLs
- Funding statements and linked study references
- Input: first 3 pages + last page (≤ 8,000 chars)

### 3. Pass B — Structural Core (inline LLM in `pipeline.py`)

Assesses whether the document follows a standard policy-brief structure via a single LLM call:

| Component | Detection |
|---|---|
| Problem statement | Presence (`present` / `weak` / `absent`) and summary |
| Solutions | Count of distinct solutions, whether explicitly proposed |
| Implementation details | Presence and count of implementation considerations |
| Narrative hook | Whether the brief opens with a case study, statistic, anecdote, or question |
| Heading labels | Whether problem, solutions, and implementation sections are explicitly labelled with recognisable headings |
| Procedural clarity | Whether the brief provides concrete, actionable steps for implementation (`present` / `weak` / `absent`) |

Input: sampled pages (first 5 + middle + last 3), truncated to 10,000 chars.

### 4. Pass C — Recommendation Extraction (`recommendation_extractor.py`)

Broad-content LLM extraction replaces the previous sentence-level prescriptive-cue filtering. The LLM receives full page-window chunks and decides what constitutes a recommendation.

**Reference-page exclusion** (deterministic):
- Scans the last 40% of the document for headings like "References", "Bibliography", "Works Cited"
- Excludes all pages from the references section onward

**Chunked LLM extraction**:
- Builds a `DocumentContent` from filtered pages with `[Page N]` markers
- Splits into page-window chunks (default ≤ 30,000 chars each)
- Each chunk goes to the LLM in one call, which returns all recommendations found
- The LLM classifies each extraction as: `recommendation`, `policy_option`, `implementation_step`, `expected_outcome`, `trade_off`, or `actor_responsibility`
- Extracts structured fields: actor (raw + type), action, target, instrument type, strength, geographic scope, timeframe, policy domain

**Post-processing** (deterministic):
- Evidence verification — confirms source text exists in the original document (whitespace-normalized)
- Deduplication — merges near-duplicate extractions across chunks (cosine similarity on action text)
- Normalization of actor types, instrument types, and strength via keyword mapping

### 5. Pass D — Frame Detection (`frame_detector.py`)

Classifies each document against five policy-instrument categories in a single LLM call per chunk — replacing the previous keyword-prefilter + per-frame call approach.

**All-frames-at-once assessment**:
- Builds `DocumentContent` from all pages, splits into chunks (≤ 30,000 chars)
- Each chunk is sent to the LLM with all five frame definitions and classification criteria
- The LLM returns a structured response assessing every frame simultaneously
- Multi-chunk results are aggregated: the highest-confidence result per frame wins

**Evidence validation**:
- Evidence quotes are verified verbatim against source text (with fuzzy prefix fallback)
- Confidence threshold (default 0.7) and minimum evidence requirements applied

**Policy-mix detection**: Flags documents where ≥ 2 distinct frames are marked `present` with sufficient confidence.

#### Frame Definitions (`config/frames.yaml`)

| Frame | Description |
|---|---|
| Command-and-Control | Legally binding rules, permits, bans, enforcement |
| Economic Instruments | Financial incentives, payments, subsidies, market mechanisms |
| Self-Regulation | Collective industry norms, certification schemes, standards |
| Voluntarism | Unilateral non-binding commitments by individual firms |
| Information Strategies | Transparency, disclosure, monitoring, traceability |

Each frame definition includes analytical notes, positive examples, and false-positive guidance.

### Document Metrics (`metrics_calculator.py`)

Computes per-document metrics:

| Category | Metrics |
|---|---|
| Structural | page count, word count, char count, paragraph count, sentence count |
| Linguistic | average sentence length, lexical diversity (type-token ratio), average word length |
| Readability | Flesch-Kincaid grade level, Flesch reading ease |
| Voice | Passive voice share — heuristic estimate of the proportion of passive-voice sentences |
| Content | URL count, email count |

## Output Format

The pipeline produces four CSV output tables and per-document audit files.

### `documents.csv`

One row per document. Contains PDF metadata, front-matter fields, metrics, and summary counts.

| Column group | Key columns |
|---|---|
| Identity | `doc_id`, `title` (PDF metadata), `author` |
| Front matter | `fm_title`, `fm_authors`, `fm_affiliations`, `fm_emails`, `fm_urls` |
| Metrics | `page_count`, `word_count`, `char_count`, `sentence_count`, `paragraph_count`, `avg_sentence_length`, `lexical_diversity`, `avg_word_length`, `flesch_kincaid_grade`, `flesch_reading_ease`, `passive_voice_share`, `url_count`, `email_count` |
| Funding | `funding_statement_present` (bool), `funding_statements_raw` — extracted from document content, not PDF metadata |
| Processing | `parser_used`, `likely_scanned`, `text_extraction_quality`, `processing_duration_seconds` |
| Summary | `frames_processed`, `recommendations_extracted`, `policy_mix_present`, `warnings` |

### `frames.csv`

One row per document × frame (5 rows per document by default).

| Column | Description |
|---|---|
| `doc_id` | Document identifier |
| `frame_id` | Machine identifier (e.g. `command_and_control`) |
| `frame_label` | Human-readable label |
| `decision` | `present`, `absent`, or `insufficient_evidence` |
| `confidence` | 0–1 confidence score |
| `evidence_count` | Number of supporting quotes |
| `evidence_1_page` | Page number of first evidence quote |
| `evidence_1_quote` | Text of first evidence quote |
| `rationale` | LLM-generated reasoning |

### `recommendations.csv`

One row per extracted recommendation / policy extraction.

| Column | Description |
|---|---|
| `doc_id` | Document identifier |
| `rec_id` | Recommendation ID (`doc_XX_rec_NNN`) |
| `extraction_type` | `recommendation`, `policy_option`, `implementation_step`, `expected_outcome`, `trade_off`, `actor_responsibility` |
| `confidence` | 0–1 confidence score |
| `source_text` | Original text span (truncated to 500 chars) |
| `page` | Source page number |
| `actor_raw`, `actor_type` | Who should act (raw text + normalized type) |
| `action_raw` | What action is recommended |
| `target_raw` | What/whom the action targets |
| `instrument_type` | Policy instrument category |
| `strength` | Prescriptive strength (must/should/could/may/consider) |
| `geographic_scope` | Spatial scope |
| `timeframe` | Temporal horizon |
| `policy_domain` | Policy area |

### `structural_core.csv`

One row per document describing structural completeness.

| Column | Description |
|---|---|
| `doc_id` | Document identifier |
| `problem_status` | `present`, `weak`, or `absent` |
| `problem_summary` | Brief description of the identified problem |
| `solutions_count` | Number of proposed solutions |
| `solutions_explicit` | Whether solutions are explicitly proposed |
| `implementation_status` | `present`, `weak`, or `absent` |
| `implementation_count` | Number of implementation considerations |
| `narrative_hook_present` | Whether a narrative hook is used |
| `narrative_hook_type` | Type of hook (statistic, case study, anecdote, question) |
| `problem_explicitly_labelled` | Whether the problem section has a recognisable heading |
| `solutions_explicitly_labelled` | Whether the solutions section has a recognisable heading |
| `implementation_explicitly_labelled` | Whether the implementation section has a recognisable heading |
| `procedural_clarity_status` | Whether concrete implementation steps are provided (`present`, `weak`, `absent`) |

### `audit/<doc_id>.json`

Per-document JSON file containing the complete extraction record:

- All extracted pages with full text
- PDF metadata and front-matter results
- Structural-core assessment
- All frame assessments with evidence quotes and LLM rationale
- All recommendation extractions with classification detail
- Processing status, warnings, and timing

## Configuration

### Module Switches (`config/config.yaml`)

Each extraction stage can be independently enabled or disabled:

```yaml
modules:
  front_matter: true
  structural_core: true
  frames: true
  recommendations: true
```

Disabled modules are skipped entirely (no LLM calls). Downstream stages receive `None` / empty lists.

### OpenAI Settings

```yaml
openai:
  model: "gpt-4o-mini"          # Must support structured outputs
  temperature: 0.1               # Low variability for reproducibility
  max_tokens: 4000
  timeout: 60
  max_retries: 5
  retry_delay: 2.0
```

### Frame Detection Settings

```yaml
frames:
  min_confidence: 0.7
  min_evidence_quotes: 1
  max_evidence_quotes: 3
```

### Recommendation Settings

```yaml
recommendations:
  min_confidence: 0.6
  max_chars_per_chunk: 30000       # Page-window chunk size for LLM
```

## LLM Integration (`llm_client.py`)

The pipeline uses OpenAI's structured output mode (JSON schema) to enforce type-safe responses from all LLM calls. All responses are validated against Pydantic V2 models.

**Schema patching**: Pydantic V2 model schemas are automatically patched before each API call to satisfy OpenAI's strict-mode requirements:
1. All object properties listed in `required`
2. `additionalProperties: false` on every object
3. Sibling keywords stripped from `$ref` nodes

**Retry logic**: Exponential backoff for transient errors (rate limits, timeouts, server errors). Schema validation failures are retried up to 2 times with error feedback appended to the conversation.

### API call budget per document

| Pass | LLM calls | Input |
|---|---|---|
| A — Front-matter | 1 | First 3 pages + last page (≤ 8K chars) |
| B — Structural core | 1 | Sampled pages: first 5 + middle + last 3 (≤ 10K chars) |
| C — Recommendations | 1 per chunk | Page-window chunks (≤ 30K chars each) |
| D — Frames | 1 per chunk | Page-window chunks (≤ 30K chars each), all 5 frames at once |

A typical 20-page policy brief generates ~6 API calls (2 fixed + ~2 recommendation chunks + ~2 frame chunks). Short briefs (< 30K chars) need only 4 calls total.

## CLI Reference

```
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Commands:
  extract          Process PDF policy briefs
  validate-config  Validate configuration files

Extract options:
  --input_dir PATH        Directory containing PDF files [required]
  --output_dir PATH       Output directory for results [required]
  --config PATH           Configuration directory [required]
  --max_workers INTEGER   Concurrent processing threads (default: 4)
  --force_reprocess       Ignore cache, reprocess all files
  --verbose / -v          Enable debug logging
  --dry_run               Show what would be processed without running
```

## Testing

```bash
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_pipeline.py -v
pytest tests/test_integration.py -v
pytest tests/test_models.py -v
```

The test suite includes 127 tests covering:
- Pydantic model validation and enum handling
- Frame detection logic (chunk-based assessment, aggregation, policy-mix)
- Recommendation extraction (reference detection, broad-content extraction, deduplication, evidence verification)
- Pipeline initialization, single-document processing, output generation
- Integration tests for module switches and graceful degradation

## Known Limitations

1. **No OCR support** — Scanned PDFs are detected (`likely_scanned` flag) but not processed. The pipeline requires text-based PDFs.

2. **Quote validation is approximate** — Evidence quotes generated by the LLM are validated against source text with whitespace normalization and a 40-char prefix fallback. Minor paraphrasing causes validation failures and the evidence is silently dropped.

3. **Single-language support** — The pipeline is designed for English-language policy briefs. LLM prompts, frame definitions, and controlled vocabularies are all English-only.

4. **Deterministic but not reproducible across models** — Results are relatively stable at temperature 0.1 for a given model version, but switching models or model versions will produce different extractions.

5. **References detection uses positional heuristic** — The references/bibliography section is detected by scanning headings in the last 40% of the document. Documents with references sections in the middle may not be detected.

6. **Chunk boundaries may split context** — Page-window chunking (≤ 30K chars) can split a recommendation or frame discussion across chunk boundaries. Overlap and deduplication mitigate but don't eliminate this.

## License

MIT License — see LICENSE file for details.
