# Policy Brief Analysis Pipeline

A reproducible research pipeline for automated analysis of policy brief PDFs. The pipeline extracts structured metadata, detects theoretical policy-instrument frames, and identifies policy recommendations using LLM-powered structured outputs.

## Overview

The pipeline processes a collection of PDF policy briefs through six sequential extraction stages:

1. **PDF extraction** — text and metadata from PDF files (PyMuPDF / PyPDF)
2. **Front-matter extraction** — content-derived metadata (title, authors, affiliations, emails, URLs, funding)
3. **Section segmentation** — heading detection and section labelling using font-size heuristics
4. **Structural core analysis** — identifies whether the document contains a problem statement, proposed solutions, and implementation details
5. **Frame detection** — classifies the document against five policy-instrument categories with evidence quotes
6. **Recommendation extraction** — identifies prescriptive policy statements using a candidate-span + LLM classification architecture

Each stage is independently toggleable via configuration. If any stage fails at runtime, the pipeline logs a warning and continues processing the remaining stages (graceful degradation).

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

# Lightweight evaluation summary
summary = pipeline.compute_extraction_summary(results)
print(summary)
```

## Requirements

- **Python 3.11+**
- **OpenAI API key** with access to a model supporting structured outputs (e.g. `gpt-4o-mini`, `gpt-4o`)
- PDF policy briefs for analysis

## Architecture

```
PolicyBriefAnalysis/
├── cli.py                          # Click-based CLI (extract, validate-config)
├── example.py                      # Programmatic usage example
├── config/
│   ├── config.yaml                 # Pipeline settings and module switches
│   ├── frames.yaml                 # Theoretical frame definitions
│   └── enums.yaml                  # Controlled vocabularies
├── src/policybrief/
│   ├── pipeline.py                 # Main orchestrator
│   ├── pdf_extractor.py            # PDF text/metadata extraction
│   ├── frame_detector.py           # Two-stage frame detection
│   ├── recommendation_extractor.py # Candidate-span recommendation extraction
│   ├── llm_client.py               # OpenAI client with schema patching
│   ├── metrics_calculator.py       # Structural/linguistic/readability metrics
│   ├── models.py                   # Pydantic data models and enums
│   └── utils.py                    # Shared helpers
├── tests/                          # pytest suite (~200 tests)
└── output/                         # Generated outputs
    ├── documents.csv / .parquet
    ├── frames.csv / .parquet
    ├── recommendations.csv / .parquet
    ├── sections.csv / .parquet
    ├── structural_core.csv / .parquet
    └── audit/                      # Per-document JSON audit trails
```

## Pipeline Stages

### 1. PDF Extraction (`pdf_extractor.py`)

Extracts per-page text and PDF-level metadata using PyMuPDF (primary) or PyPDF (fallback).

- Per-page text with word/character counts
- PDF metadata: title, author, subject, creator, dates
- Layout line features (font size, bold flags) for downstream section detection
- Scanned-document detection via text-density heuristics (< 100 chars/page)
- Text extraction quality score (0–1)
- File content hashing (SHA-256) for incremental processing

### 2. Front-Matter Extraction

Extracts content-derived metadata from the first pages of the document:

- Document title and author names (parsed from text, not PDF metadata)
- Institutional affiliations, email addresses, URLs
- Funding statements and linked study references

### 3. Section Segmentation

Identifies document sections from PDF layout features:

- Detects headings using relative font-size thresholds
- Assigns normalized labels to recognized section types (e.g. `introduction`, `recommendations`, `references`)
- Provides a section map used by downstream modules to scope their analysis

### 4. Structural Core Analysis

Determines whether the document follows a standard policy-brief structure:

| Component | Detection |
|---|---|
| Problem statement | Presence and explicitness of a described problem |
| Solutions | Count of proposed solutions, whether explicitly labelled |
| Implementation details | Presence of implementation steps or guidance |
| Narrative hook | Whether the brief opens with an anecdote, statistic, or question |

### 5. Frame Detection (`frame_detector.py`)

Classifies each document against five policy-instrument categories derived from the smart-regulation literature. Detection uses a two-stage approach to balance cost and accuracy.

**Stage 1 — Keyword-based span selection** (no LLM calls):
- Matches inclusion cues from `frames.yaml` against the full text
- Applies exclusion cues (score × 0.3) and must-have boosts (score × 2.0)
- Extracts context windows (default 500 chars) around keyword hits
- Returns top-N candidate spans per frame

**Stage 2 — LLM assessment** (one API call per frame):
- Sends candidate spans plus the frame definition to the LLM
- Returns a structured `FrameAssessment`: decision (`present` / `absent` / `insufficient_evidence`), confidence score, evidence quotes with page numbers, rationale, and counterevidence
- Validates evidence quotes verbatim against source pages

**Policy-mix detection**:
- Flags documents where ≥ 2 frames are marked `present` AND the text contains explicit policy-mix language (`"policy mix"`, `"instrument mix"`, `"complementarity"`, `"smart regulation"`, etc.)

#### Frame Definitions (`config/frames.yaml`)

| Frame | Description | Example cues |
|---|---|---|
| Command-and-Control | Legally binding rules, permits, bans, zoning, enforcement | permit, license, regulation, FLEGT, EUTR, penalty |
| Economic Instruments | Financial incentives, payments, subsidies, market mechanisms | PES, subsidy, carbon credit, REDD+, green bond |
| Self-Regulation | Collective industry norms, certification schemes, standards | FSC, PEFC, ISO 14001, RSPO, code of practice |
| Voluntarism | Unilateral non-binding commitments by individual firms | corporate pledge, zero-deforestation, CSR |
| Information Strategies | Transparency, disclosure, monitoring, traceability | traceability, satellite monitoring, due diligence |

Each frame definition includes analytical notes, positive examples, and false-positive guidance to steer the LLM.

### 6. Recommendation Extraction (`recommendation_extractor.py`)

Identifies and classifies prescriptive policy statements using a candidate-span architecture.

**Candidate generation** (deterministic):
- Splits text into sentence-level spans
- Restricts candidates to target sections: `recommendations`, `key_messages`, `executive_summary`, `policy_options`, `implementation`, `conclusion`
- Excludes reference-heavy sections: `references`, `acknowledgements`, `about_authors`, `appendix`
- Filters for prescriptive language (`should`, `must`, `recommend`, `propose`, `call for`, `urge`, `ensure`, etc.)
- Rejects citation-heavy spans (≥ 2 citation patterns like `(Author, Year)`, `[1]`, `et al.`)

**LLM classification** (batched, 10 candidates per API call):
- Classifies each candidate into one of: `recommendation`, `policy_option`, `implementation_step`, `expected_outcome`, `trade_off`, `actor_responsibility`, `non_recommendation`
- Extracts structured fields: actor (normalized), action, target, instrument type, geographic scope, timeframe, strength
- Provides evidence quotes with page numbers

**Controlled vocabularies** (from `config/enums.yaml`):
- **Actor types** (9): government, EU institutions, international organizations, private sector, civil society, research institutions, individuals, multiple actors, unspecified
- **Instrument types** (12): regulation, subsidy, tax, information, voluntary, planning, monitoring, research, procurement, infrastructure, institutional, other
- **Geographic scope** (11): local → global, EU, bilateral, multilateral, transboundary, unspecified
- **Timeframe** (6): immediate (< 1 yr), short-term (1–3 yr), medium-term (3–10 yr), long-term (> 10 yr), ongoing, unspecified
- **Strength** (6): must, should, could, may, consider, unspecified

### Document Metrics (`metrics_calculator.py`)

Computes 20+ metrics per document:

| Category | Metrics |
|---|---|
| Structural | page count, word count, char count, heading count, paragraph count, sentence count, list item count |
| Linguistic | average sentence length, lexical diversity (type-token ratio), average word length, passive voice % |
| Readability | Flesch-Kincaid grade level, Flesch reading ease |
| Content density | table count, figure count, reference count, URL count |

## Output Format

The pipeline produces five output tables (CSV + Parquet) and per-document audit files.

### `documents.csv`

One row per document. Contains PDF metadata, front-matter fields, all computed metrics, and summary counts.

| Column group | Key columns |
|---|---|
| Identity | `doc_id`, `file_path`, `file_name` |
| PDF metadata | `title`, `author`, `creation_date`, `subject` |
| Front matter | `content_title`, `content_authors`, `affiliations`, `emails`, `urls`, `funding_statements`, `linked_studies` |
| Processing | `processing_timestamp`, `processing_duration_seconds`, `likely_scanned`, `text_extraction_quality` |
| Metrics | `page_count`, `word_count`, `char_count`, `heading_count`, `paragraph_count`, `sentence_count`, `list_item_count`, `avg_sentence_length`, `lexical_diversity`, `avg_word_length`, `flesch_kincaid_grade`, `flesch_reading_ease`, `table_count`, `figure_count`, `reference_count`, `url_count`, `passive_voice_percent` |
| Summary | `frames_present`, `frames_absent`, `policy_mix_present`, `recommendations_count` |

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
| `evidence_quotes` | Pipe-separated verbatim quotes |
| `evidence_pages` | Comma-separated page numbers |
| `rationale` | LLM-generated reasoning |
| `counterevidence_count` | Number of contradicting quotes |

### `recommendations.csv`

One row per extracted recommendation.

| Column | Description |
|---|---|
| `doc_id` | Document identifier |
| `rec_id` | Recommendation ID (`doc_XX_rec_NN`) |
| `extraction_type` | `recommendation`, `policy_option`, `implementation_step`, `expected_outcome`, `trade_off`, `actor_responsibility` |
| `confidence` | 0–1 confidence score |
| `source_section` | Section where the span was found |
| `page` | Source page number |
| `source_text_raw` | Original text span |
| `actor_text_raw`, `actor_type_normalized` | Who should act |
| `action_text_raw` | What action is recommended |
| `target_text_raw` | What/whom the action targets |
| `instrument_type` | Policy instrument category |
| `policy_domain` | Policy area |
| `geographic_scope` | Spatial scope |
| `timeframe` | Temporal horizon |
| `strength` | Prescriptive strength (must/should/could/may/consider) |
| `expected_outcomes`, `implementation_steps`, `trade_offs` | Additional detail fields |
| `evidence_count`, `evidence_quotes`, `evidence_pages` | Supporting evidence |

### `sections.csv`

One row per detected text segment.

| Column | Description |
|---|---|
| `doc_id` | Document identifier |
| `section_index` | Sequential position in document |
| `raw_title` | Original text of the segment heading or line |
| `normalized_label` | Standardized section label (if recognized) |
| `start_page`, `end_page` | Page range |
| `confidence` | Detection confidence |
| `rule_source` | Which heuristic matched |
| `detection_method` | How the section was identified |

### `structural_core.csv`

One row per document describing structural completeness.

| Column | Description |
|---|---|
| `doc_id` | Document identifier |
| `problem_status` | Whether a problem statement was found |
| `problem_section` | Which section contains the problem |
| `problem_labeled` | Whether the section is explicitly labelled |
| `solutions_count` | Number of proposed solutions |
| `solutions_explicit`, `solutions_labeled` | Explicitness flags |
| `implementation_status` | Whether implementation details are present |
| `implementation_count`, `implementation_labeled` | Implementation detail flags |
| `narrative_hook_status` | Whether a narrative hook opens the brief |
| `narrative_hook_type` | Type of hook (anecdote, statistic, question) |

### `audit/<doc_id>.json`

Per-document JSON file containing the complete extraction record:

- PDF metadata and front-matter results
- Section map and structural-core assessment
- All frame assessments with full LLM context
- All recommendation extractions with classification detail
- Processing status, warnings, and timing

## Configuration

### Module Switches (`config/config.yaml`)

Each extraction stage can be independently enabled or disabled:

```yaml
modules:
  front_matter: true
  section_segmentation: true
  structural_core: true
  frames: true
  recommendations: true
```

Disabled modules are skipped entirely (no LLM calls). Downstream stages receive `None` and adapt accordingly.

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
  target_sections:               # Where to search for recommendations
    - recommendation
    - conclusion
    - summary
    - policy implication
    # ...
```

### Processing and Output

```yaml
processing:
  max_workers: 1                 # Concurrent threads
  batch_size: 10
  incremental: true              # Skip unchanged files (hash-based)
  hash_algorithm: "sha256"

output:
  formats: [csv, parquet, json]
  generate_audit: true
  compress_json: true
  include_raw_text: false

validation:
  verify_quotes: true
  max_quote_length: 500
  min_quote_length: 20
  strict_schema: true
```

## LLM Integration (`llm_client.py`)

The pipeline uses OpenAI's structured output mode to enforce JSON schema compliance on all LLM responses.

**Schema patching**: Pydantic V2 model schemas are automatically patched before each API call to satisfy OpenAI's strict-mode requirements:
1. All object properties added to `required`
2. `additionalProperties: false` set on all object types
3. Sibling keywords stripped from `$ref` nodes

**Retry logic**: Exponential backoff for transient errors (rate limits, timeouts, server errors). Schema validation failures are retried up to 2 times with error feedback appended to the prompt.

**Token optimization**: The two-stage frame detection sends only relevant text spans (not full documents) to the LLM, reducing token costs significantly for long documents.

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

The test suite includes ~200 tests covering:
- Unit tests for all extraction modules
- Integration tests for module switches and graceful degradation
- Output format validation (CSV/Parquet schema correctness)
- Evaluation summary computation
- Cache-based incremental processing
- Backward compatibility

## Known Limitations

1. **No OCR support** — Scanned PDFs are detected (`likely_scanned` flag) but not processed. The pipeline requires text-based PDFs.

2. **Quote validation is approximate** — The LLM generates evidence quotes that are validated verbatim against source pages. Minor paraphrasing or whitespace differences cause validation warnings. Unvalidated quotes are still included but flagged.

3. **Section segmentation granularity** — The segmenter operates at the text-block level based on font-size heuristics. For documents with inconsistent formatting, this may produce very granular segments rather than logical sections.

4. **LLM candidate count mismatches** — When the LLM returns fewer classifications than candidates submitted (batch classification), the shortfall is padded with `non_recommendation`. A warning is logged.

5. **Single-language support** — The pipeline is designed for English-language policy briefs. Frame keywords, prescriptive-language cues, and LLM prompts are all English-only.

6. **Token cost scales with document length** — Frame detection makes one LLM call per frame per document (5 calls/doc by default). Recommendation extraction makes one call per batch of 10 candidates. Long documents with many prescriptive sentences will generate more API calls.

7. **Deterministic but not reproducible across models** — Results are deterministic at temperature 0.1 for a given model version, but switching models (e.g. `gpt-4o-mini` to `gpt-4o`) will produce different extractions.

8. **Pydantic V2 deprecation warnings** — Some models use V1-style `@validator` decorators. These work correctly but produce deprecation warnings under Pydantic V2.

## License

MIT License — see LICENSE file for details.

## 🙋 Support

For questions and support:
1. Check this README and configuration examples
2. Review test files for usage patterns
3. Examine audit outputs for debugging
4. Open an issue with detailed error information

---

**Built for reproducible policy research** 📊🔬
