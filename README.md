# Policy Brief Analysis Pipeline

A reproducible research pipeline for automated analysis of policy brief PDFs. The pipeline extracts structured metadata, detects theoretical policy-instrument frames, and identifies policy recommendations using an LLM-first architecture with deterministic guardrails.

## Overview

The pipeline processes PDF policy briefs through five sequential extraction stages:

1. **PDF extraction** — text and metadata from PDF files (PyMuPDF / pypdf)
2. **Front-matter extraction** — content-derived metadata via LLM (title, authors, affiliations, emails, URLs, funding)
3. **Structural core analysis** — LLM-based assessment of problem identification, solutions, implementation details, and narrative hooks
4. **Frame detection** — two-stage classification (keyword pre-filter + LLM) against five policy-instrument categories
5. **Recommendation extraction** — deterministic candidate generation (prescriptive-language filter + citation rejection) followed by batched LLM classification

Each stage is independently toggleable via configuration. If any stage fails at runtime, the pipeline logs a warning and continues with the remaining stages (graceful degradation). Without an API key, all LLM-dependent stages degrade gracefully and the pipeline still produces PDF metrics and metadata.

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
├── cli.py                          # Click-based CLI (extract, validate-config)
├── config/
│   ├── config.yaml                 # Pipeline settings and module switches
│   ├── frames.yaml                 # Theoretical frame definitions
│   └── enums.yaml                  # Controlled vocabularies
├── src/policybrief/
│   ├── pipeline.py                 # Main orchestrator (inline LLM calls for front-matter & structural core)
│   ├── pdf_extractor.py            # PDF text/metadata extraction (PyMuPDF + pypdf fallback)
│   ├── frame_detector.py           # Two-stage frame detection (keyword + LLM)
│   ├── recommendation_extractor.py # Candidate-span recommendation extraction (prescriptive filter + LLM)
│   ├── llm_client.py               # OpenAI client with schema patching and retry
│   ├── metrics_calculator.py       # Document metrics (readability, structure)
│   ├── models.py                   # Pydantic V2 data models and enums
│   └── utils.py                    # Shared helpers (I/O, config, text cleaning)
├── tests/                          # pytest suite (93 tests)
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

### 2. Front-Matter Extraction (inline LLM in `pipeline.py`)

Extracts content-derived metadata from the first few pages of the document via a single LLM call:

- Document title, author names
- Institutional affiliations, email addresses, URLs
- Funding statements and linked study references
- Uses first 3 pages + last page as input (truncated to 8,000 chars)

### 3. Structural Core Analysis (inline LLM in `pipeline.py`)

Determines whether the document follows a standard policy-brief structure via a single LLM call:

| Component | Detection |
|---|---|
| Problem statement | Presence (`present` / `weak` / `absent`) and summary |
| Solutions | Count of distinct solutions, whether explicitly proposed |
| Implementation details | Presence and count of implementation considerations |
| Narrative hook | Whether the brief opens with a case study, statistic, anecdote, or question |

Uses a sampled set of pages (first 5 + middle + last 3) truncated to 10,000 chars.

### 4. Frame Detection (`frame_detector.py`)

Classifies each document against five policy-instrument categories derived from the smart-regulation literature. Uses a two-stage approach to balance cost and accuracy.

**Stage 1 — Keyword-based span selection** (deterministic, no LLM calls):
- Matches inclusion cues from `frames.yaml` against the full text
- Extracts context windows (default 500 chars) around keyword hits
- Deduplicates overlapping spans
- Returns top-N candidate spans per frame (default 5)

**Stage 2 — LLM assessment** (one API call per frame):
- Sends candidate spans plus the frame definition to the LLM
- Returns a structured `FrameDetectionOutput`: decision (`present` / `absent` / `insufficient_evidence`), confidence score, evidence quotes, and rationale
- Validates evidence quotes verbatim against source text (with fuzzy prefix fallback)
- Applies confidence threshold (default 0.7) and minimum evidence requirements

**Policy-mix detection**: Flags documents where ≥ 2 distinct frames are marked `present` with sufficient confidence.

#### Frame Definitions (`config/frames.yaml`)

| Frame | Description | Example cues |
|---|---|---|
| Command-and-Control | Legally binding rules, permits, bans, zoning, enforcement | permit, license, regulation, FLEGT, EUTR, penalty |
| Economic Instruments | Financial incentives, payments, subsidies, market mechanisms | PES, subsidy, carbon credit, REDD+, green bond |
| Self-Regulation | Collective industry norms, certification schemes, standards | FSC, PEFC, ISO 14001, RSPO, code of practice |
| Voluntarism | Unilateral non-binding commitments by individual firms | corporate pledge, zero-deforestation, CSR |
| Information Strategies | Transparency, disclosure, monitoring, traceability | traceability, satellite monitoring, due diligence |

Each frame definition includes analytical notes, positive examples, and false-positive guidance to steer the LLM.

### 5. Recommendation Extraction (`recommendation_extractor.py`)

Identifies and classifies prescriptive policy statements using a candidate-span architecture.

**Reference-page exclusion** (deterministic):
- Detects the start of the references/bibliography section by scanning the last 40% of the document for headings like "References", "Bibliography", "Works Cited"
- Excludes all pages from the references section onward

**Candidate generation** (deterministic):
- Splits text into sentence-level spans (regex-based)
- Filters for prescriptive language: `should`, `must`, `need to`, `recommend`, `propose`, `suggest`, `call for`, `urge`, `require`, `ensure`, `promote`, `encourage`, `implement`, `establish`, `strengthen`, `foster`, `prioritize`, `advocate`, `advise`
- Rejects citation-heavy spans (> 40% citation markup by character count)
- Skips sentences shorter than 5 words

**LLM classification** (batched, 10 candidates per API call):
- Classifies each candidate into: `recommendation`, `policy_option`, `implementation_step`, `expected_outcome`, `trade_off`, `actor_responsibility`, or `non_recommendation`
- Extracts structured fields: actor (raw + normalized type), action, target, instrument type, strength
- Applies minimum confidence threshold (default 0.6)

**Post-validation**:
- Verifies candidate source text exists in the original document (whitespace-normalized)
- Normalizes actor types, instrument types, and recommendation strength via keyword mapping
- Requires evidence for recommendations and policy options

**Controlled vocabularies** (defined in `models.py`):
- **Actor types** (9): government, EU institutions, international organizations, private sector, civil society, research institutions, individuals, multiple actors, unspecified
- **Instrument types** (12): regulation, subsidy, tax, information, voluntary, planning, monitoring, research, procurement, infrastructure, institutional, other
- **Geographic scope** (11): local → global, EU, bilateral, multilateral, transboundary, unspecified
- **Timeframe** (6): immediate, short-term, medium-term, long-term, ongoing, unspecified
- **Strength** (6): must, should, could, may, consider, unspecified

### Document Metrics (`metrics_calculator.py`)

Computes per-document metrics:

| Category | Metrics |
|---|---|
| Structural | page count, word count, char count, paragraph count, sentence count |
| Linguistic | average sentence length, lexical diversity (type-token ratio), average word length |
| Readability | Flesch-Kincaid grade level, Flesch reading ease |
| Content | URL count, email count |

## Output Format

The pipeline produces four CSV output tables and per-document audit files.

### `documents.csv`

One row per document. Contains PDF metadata, front-matter fields, metrics, and summary counts.

| Column group | Key columns |
|---|---|
| Identity | `doc_id`, `title` (PDF metadata), `author` |
| Front matter | `fm_title`, `fm_authors`, `fm_affiliations`, `fm_emails`, `fm_urls` |
| Metrics | `page_count`, `word_count`, `char_count`, `sentence_count`, `paragraph_count`, `avg_sentence_length`, `lexical_diversity`, `avg_word_length`, `flesch_kincaid_grade`, `flesch_reading_ease`, `url_count`, `email_count` |
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
  max_spans_per_frame: 5         # Top-N candidate spans sent to LLM
  context_window: 500            # Characters around keyword hits
  min_evidence_quotes: 1
  max_evidence_quotes: 3
```

### Recommendation Settings

```yaml
recommendations:
  min_confidence: 0.6
  recommendation_signals:        # Prescriptive language cues
    - recommend
    - should
    - must
    - need to
    # ...
```

## LLM Integration (`llm_client.py`)

The pipeline uses OpenAI's structured output mode (JSON schema) to enforce type-safe responses from all LLM calls. All responses are validated against Pydantic V2 models.

**Schema patching**: Pydantic V2 model schemas are automatically patched before each API call to satisfy OpenAI's strict-mode requirements:
1. All object properties listed in `required`
2. `additionalProperties: false` on every object
3. Sibling keywords stripped from `$ref` nodes

**Retry logic**: Exponential backoff for transient errors (rate limits, timeouts, server errors). Schema validation failures are retried up to 2 times with error feedback appended to the conversation.

**Token optimization**: Frame detection sends only keyword-matched context windows (not full documents) to the LLM. Front-matter and structural-core extraction sample specific pages rather than sending the entire document.

### API call budget per document

| Stage | LLM calls | Input |
|---|---|---|
| Front-matter | 1 | First 3 pages + last page (≤ 8K chars) |
| Structural core | 1 | Sampled pages: first 5 + middle + last 3 (≤ 10K chars) |
| Frame detection | up to 5 | Keyword-matched spans per frame |
| Recommendations | N / 10 | Batches of 10 prescriptive candidates |

A typical 20-page policy brief generates ~10 API calls. Longer documents with many prescriptive sentences generate proportionally more recommendation classification calls.

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

The test suite includes 93 tests covering:
- Pydantic model validation and enum handling
- Frame detection logic (keyword matching, assessment, policy-mix)
- Recommendation extraction (references detection, prescriptive cues, citation rejection, candidate generation)
- Pipeline initialization, single-document processing, output generation
- Integration tests for module switches and graceful degradation
- Evaluation summary computation

## Known Limitations

1. **No OCR support** — Scanned PDFs are detected (`likely_scanned` flag) but not processed. The pipeline requires text-based PDFs.

2. **Quote validation is approximate** — Evidence quotes generated by the LLM are validated verbatim against source text (with whitespace normalization and a 40-char prefix fallback). Minor paraphrasing causes validation failures and the evidence is silently dropped.

3. **LLM candidate count mismatches** — When the LLM returns fewer classifications than candidates submitted (batch classification), the shortfall is padded with `non_recommendation`.

4. **Single-language support** — The pipeline is designed for English-language policy briefs. Frame keywords, prescriptive-language cues, and LLM prompts are all English-only.

5. **Token cost scales with candidate count** — Recommendation extraction makes one LLM call per batch of 10 candidates. Documents with many prescriptive sentences (e.g. 260 candidates for a 170-page report) can require 27+ API calls for recommendations alone.

6. **Deterministic but not reproducible across models** — Results are relatively stable at temperature 0.1 for a given model version, but switching models (e.g. `gpt-4o-mini` to `gpt-4o`) or model versions will produce different extractions.

7. **Instrument type and strength often unset** — The LLM classification conservatively returns `null` for instrument type and recommendation strength when not clearly determinable from the source text. These fields are populated only when the LLM has high confidence.

8. **References detection uses positional heuristic** — The references/bibliography section is detected by scanning headings in the last 40% of the document. Documents with unconventional structure (e.g. short front-matter containing "Reference" citation lines) are handled correctly by this constraint, but documents with references sections in the middle may not be detected.

## License

MIT License — see LICENSE file for details.
