# Policy Brief Analysis Pipeline

A production-ready, reproducible research pipeline for comprehensive analysis of policy brief PDFs using AI-powered structured extraction.

## 🎯 Overview

This pipeline provides automated, systematic analysis of policy documents with:

- **Document-level metrics**: Structural and linguistic analysis 
- **Theoretical frame detection**: Closed-vocabulary identification with evidence
- **Policy recommendations extraction**: Structured analysis with validation
- **Structured outputs**: JSON schema validation for reproducibility
- **Audit trails**: Complete traceability of extraction decisions
- **Incremental processing**: Efficient handling of document collections

## ✨ Key Features

### 🔍 **Comprehensive Analysis**
- PDF text extraction with layout preservation
- Document structure and readability metrics
- Theoretical framework detection (closed vocabulary)
- Policy recommendation extraction and classification

### 🤖 **AI-Powered Extraction** 
- OpenAI GPT-4 with structured outputs (JSON schema validation)
- Two-stage frame detection to optimize token usage
- Evidence-based assessments with verbatim quote validation
- Deterministic processing with low temperature settings

### 📊 **Structured Data Outputs**
- CSV/Parquet files for analysis
- JSON audit files for transparency
- Pydantic model validation for data quality
- Tabular format ready for statistical analysis

### ⚡ **Production Ready**
- Concurrent document processing
- Incremental updates (skip unchanged files)
- Comprehensive error handling and logging  
- Extensive test coverage

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd PolicyBriefAnalysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Run setup script (optional but recommended)
python setup.py
```

### 2. Configuration

Set your OpenAI API key:
```bash
# Copy environment template
cp .env.template .env

# Edit .env and add your API key
OPENAI_API_KEY=your-api-key-here
```

### 3. Basic Usage

```bash
# Process PDFs in a directory
python cli.py extract \
  --input_dir ./pdfs \
  --output_dir ./out \
  --config ./config
```

## 📋 Requirements

- **Python 3.11+**
- **OpenAI API key** with access to structured output models
- PDF files for analysis

## 🏗️ Architecture

```
PolicyBriefAnalysis/
├── cli.py                 # Command-line interface
├── src/policybrief/       # Main package
│   ├── pipeline.py        # Main orchestrator
│   ├── pdf_extractor.py   # PDF text extraction
│   ├── frame_detector.py  # Theoretical frame detection
│   ├── recommendation_extractor.py  # Policy recommendations
│   ├── llm_client.py      # OpenAI integration
│   ├── metrics_calculator.py  # Document metrics
│   └── models.py          # Pydantic data models
├── config/               # Configuration files
│   ├── config.yaml       # Main pipeline settings
│   ├── frames.yaml       # Theoretical frameworks
│   └── enums.yaml        # Controlled vocabularies
├── tests/               # Test suite
└── out/                 # Output directory
    ├── documents.csv    # Document-level results  
    ├── frames.csv       # Frame detection results
    ├── recommendations.csv  # Policy recommendations
    └── audit/          # Per-document audit trails
```

## 🎛️ Configuration

### Theoretical Frames (`config/frames.yaml`)

Define the theoretical frameworks to detect:

```yaml
frames:
  - id: "market_based_instruments"
    label: "Market-Based Instruments"
    short_definition: "Policies using market mechanisms and economic incentives"
    inclusion_cues: ["carbon pricing", "cap and trade", "emissions trading"]
    exclusion_cues: ["command and control", "direct regulation"]
    must_have: [["market", "price", "economic"], ["incentive", "mechanism"]]
```

### Pipeline Settings (`config/config.yaml`)

Customize processing parameters:

```yaml
openai:
  model: "gpt-4o-2024-08-06"
  temperature: 0.1
  max_tokens: 4000

frames:
  min_confidence: 0.7
  max_spans_per_frame: 5

recommendations:
  min_confidence: 0.6
  max_recommendations: 10
```

## 📊 Output Format

### Documents Table (`documents.csv`)
Document-level metadata and metrics:
- `doc_id`, `file_path`, `title`, `author`
- `page_count`, `word_count`, `char_count`  
- `avg_sentence_length`, `lexical_diversity`
- `frames_present`, `recommendations_count`

### Frames Table (`frames.csv`) 
Frame detection results (one row per document × frame):
- `doc_id`, `frame_id`, `frame_label`
- `decision` (present/absent/insufficient_evidence)
- `confidence`, `evidence_quotes`, `evidence_pages`
- `rationale`

### Recommendations Table (`recommendations.csv`)
Policy recommendations (one row per recommendation):
- `doc_id`, `rec_id`, `actor`, `action`, `target`
- `instrument_type`, `policy_domain`, `geographic_scope`
- `timeframe`, `strength`, `evidence_quotes`

### Audit Files (`audit/<doc_id>.json`)
Complete extraction details per document:
- Raw extracted text and metadata
- All LLM inputs and outputs  
- Validation results and processing status
- Evidence locations and verification

## 🔧 Advanced Usage

### Custom Theoretical Frames

Add new frameworks by editing `config/frames.yaml`:

```yaml
frames:
  - id: "your_custom_frame"
    label: "Your Custom Framework"
    short_definition: "Description of your theoretical framework"
    inclusion_cues: ["keyword1", "keyword2", "phrase"]
    exclusion_cues: ["contradictory", "opposing"] 
    must_have: [["required", "terms"], ["essential", "concepts"]]
```

### Programmatic Usage

```python
from src.policybrief.pipeline import PolicyBriefPipeline
from pathlib import Path

# Initialize pipeline
pipeline = PolicyBriefPipeline(
    config_dir=Path("config"),
    output_dir=Path("output"),
    max_workers=4
)

# Process documents
pdf_files = list(Path("pdfs").glob("*.pdf"))
results = pipeline.process_documents(pdf_files)

print(f"Processed: {len(results['processed'])} documents")
```

### Batch Processing

```bash
# Process large document collections
python cli.py extract \
  --input_dir ./large_collection \
  --output_dir ./results \
  --config ./config \
  --max_workers 8 \
  --verbose

# Force reprocess all files (ignore cache)
python cli.py extract \
  --input_dir ./pdfs \
  --output_dir ./out \
  --config ./config \
  --force_reprocess
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Performance

- **Throughput**: ~2-5 documents per minute (depends on length and complexity)
- **Concurrency**: Configurable worker threads for parallel processing  
- **Caching**: Incremental processing skips unchanged files
- **Token Usage**: Two-stage frame detection optimizes LLM costs

## 🛠️ Troubleshooting

### Common Issues

**API Key Not Found**
```bash
export OPENAI_API_KEY=your-key-here
# or add to .env file
```

**PDF Extraction Fails**
- Check if PDF is scanned (OCR may be needed)
- Verify PDF is not corrupted
- Try different extraction method in config

**Frame Detection Issues**
- Review inclusion/exclusion cues in frames.yaml
- Check confidence thresholds
- Examine audit files for detailed reasoning

**Low Text Quality**
- Check `likely_scanned` flag in results
- Review `text_extraction_quality` metrics
- Consider OCR preprocessing for scanned documents

### Validation Commands

```bash
# Validate configuration
python cli.py validate-config --config ./config

# Test with small sample
python cli.py extract \
  --input_dir ./tests/data \
  --output_dir ./test_out \
  --config ./config \
  --dry_run
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure tests pass: `pytest tests/`
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- **Documentation**: [Additional docs location]
- **Issues**: [GitHub Issues URL]
- **Examples**: See `example.py` for usage examples

## 🙋 Support

For questions and support:
1. Check this README and configuration examples
2. Review test files for usage patterns
3. Examine audit outputs for debugging
4. Open an issue with detailed error information

---

**Built for reproducible policy research** 📊🔬
