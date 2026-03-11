#!/usr/bin/env python3
"""
Example script demonstrating how to use the Policy Brief Analysis Pipeline.

This script shows different ways to use the pipeline:
1. Basic usage with default settings
2. Customized configuration
3. Processing specific files
4. Analyzing results
"""

import logging
import os
from pathlib import Path

# Set up logging for the example
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipeline components
from src.policybrief.pipeline import PolicyBriefPipeline
from src.policybrief.utils import load_json, validate_file_paths


def basic_pipeline_example():
    """Basic pipeline usage example."""
    print("🔄 Running basic pipeline example...")
    
    # Set up directories
    input_dir = Path("pdfs")  # Directory containing PDF files
    output_dir = Path("output") 
    config_dir = Path("config")
    
    # Ensure input directory exists (create if needed for demo)
    input_dir.mkdir(exist_ok=True)
    
    # Find PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found in pdfs/ directory")
        print("   Add some policy brief PDFs to pdfs/ and run again")
        return
    
    try:
        # Initialize pipeline
        pipeline = PolicyBriefPipeline(
            config_dir=config_dir,
            output_dir=output_dir,
            max_workers=2,  # Use 2 workers for this example
            force_reprocess=False
        )
        
        # Process documents
        results = pipeline.process_documents(pdf_files)
        
        # Print summary
        print(f"✅ Processing complete!")
        print(f"   Processed: {len(results['processed'])} documents")
        print(f"   Skipped: {len(results['skipped'])} documents") 
        print(f"   Errors: {len(results['errors'])} documents")
        
        if results['errors']:
            print("❌ Errors encountered:")
            for error in results['errors']:
                print(f"   - {error}")
        
        # Show output files
        if results['processed']:
            print(f"\n📁 Output files generated in {output_dir}/:")
            for file in output_dir.glob("*.csv"):
                print(f"   - {file.name}")
        
        return results
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return None


def analyze_results_example(output_dir: Path):
    """Example of analyzing pipeline results."""
    print(f"\n🔍 Analyzing results from {output_dir}...")
    
    try:
        import pandas as pd
        
        # Load results
        documents_file = output_dir / "documents.csv"
        frames_file = output_dir / "frames.csv"
        recommendations_file = output_dir / "recommendations.csv"
        
        if not all(f.exists() for f in [documents_file, frames_file, recommendations_file]):
            print("⚠️  Result files not found. Run processing first.")
            return
        
        # Load data
        documents_df = pd.read_csv(documents_file)
        frames_df = pd.read_csv(frames_file)
        recommendations_df = pd.read_csv(recommendations_file)
        
        print(f"📊 Analysis Summary:")
        print(f"   Documents processed: {len(documents_df)}")
        print(f"   Total frame assessments: {len(frames_df)}")
        print(f"   Total recommendations: {len(recommendations_df)}")
        
        # Frame analysis
        frame_stats = frames_df.groupby('decision').size()
        print(f"\n🎯 Frame Detection Results:")
        for decision, count in frame_stats.items():
            print(f"   {decision}: {count}")
        
        # Most common frames
        if len(frames_df[frames_df['decision'] == 'present']) > 0:
            present_frames = frames_df[frames_df['decision'] == 'present']['frame_label'].value_counts()
            print(f"\n🏆 Most detected theoretical frames:")
            for frame, count in present_frames.head().items():
                print(f"   {frame}: {count} documents")
        
        # Recommendation analysis
        if len(recommendations_df) > 0:
            instrument_stats = recommendations_df['instrument_type'].value_counts()
            print(f"\n📋 Policy Instruments:")
            for instrument, count in instrument_stats.head().items():
                print(f"   {instrument}: {count}")
            
            actor_stats = recommendations_df['actor'].value_counts()
            print(f"\n👥 Implementation Actors:")
            for actor, count in actor_stats.head().items():
                print(f"   {actor}: {count}")
        
        # Document quality metrics
        avg_quality = documents_df['text_extraction_quality'].mean()
        scanned_docs = documents_df['likely_scanned'].sum()
        
        print(f"\n📈 Document Quality:")
        print(f"   Average text extraction quality: {avg_quality:.2f}")
        print(f"   Likely scanned documents: {scanned_docs}")
        
    except ImportError:
        print("⚠️  pandas not available for analysis")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


def custom_configuration_example():
    """Example with customized configuration."""
    print("\n⚙️  Custom configuration example...")
    
    # Create custom config (in practice, modify config files)
    custom_output_dir = Path("custom_output")
    
    try:
        pipeline = PolicyBriefPipeline(
            config_dir=Path("config"),
            output_dir=custom_output_dir,
            max_workers=1,  # Single worker
            force_reprocess=True  # Force reprocess all files
        )
        
        print(f"✅ Pipeline initialized with custom settings:")
        print(f"   Max workers: {pipeline.max_workers}")
        print(f"   Force reprocess: {pipeline.force_reprocess}")
        print(f"   Theoretical frames: {len(pipeline.frames)}")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Custom configuration failed: {e}")
        return None


def audit_file_example(output_dir: Path):
    """Example of examining audit files."""
    print(f"\n🔍 Examining audit files in {output_dir}/audit/...")
    
    audit_dir = output_dir / "audit"
    
    if not audit_dir.exists():
        print("⚠️  No audit directory found")
        return
    
    audit_files = list(audit_dir.glob("*.json"))
    
    if not audit_files:
        print("⚠️  No audit files found")
        return
    
    # Load first audit file as example
    audit_file = audit_files[0]
    
    try:
        audit_data = load_json(audit_file)
        
        print(f"📄 Audit file: {audit_file.name}")
        print(f"   Document ID: {audit_data['doc_id']}")
        print(f"   Processing time: {audit_data['processing_status']['processing_duration_seconds']:.2f}s")
        print(f"   Pages processed: {audit_data['processing_status']['pages_processed']}")
        print(f"   Frames detected: {audit_data['processing_status']['frames_processed']}")
        print(f"   Recommendations: {audit_data['processing_status']['recommendations_extracted']}")
        
        # Show frame results
        print(f"\n🎯 Frame assessments:")
        for assessment in audit_data['frame_assessments']:
            decision = assessment['decision']
            confidence = assessment['confidence']
            frame_label = assessment['frame_label']
            print(f"   {frame_label}: {decision} (confidence: {confidence:.2f})")
        
        # Show recommendations
        if audit_data['recommendations']:
            print(f"\n📋 Recommendations:")
            for i, rec in enumerate(audit_data['recommendations'][:3]):  # Show first 3
                actor = rec['actor']
                action = rec['action'][:50] + "..." if len(rec['action']) > 50 else rec['action']
                print(f"   {i+1}. {actor} should {action}")
        
    except Exception as e:
        print(f"❌ Failed to load audit file: {e}")


def main():
    """Main example function."""
    print("🚀 Policy Brief Analysis Pipeline Examples")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Please set your OpenAI API key and run again")
        return
    
    # Example 1: Basic pipeline usage
    results = basic_pipeline_example()
    
    # Example 2: Analyze results if processing was successful
    if results and results['processed']:
        analyze_results_example(Path("output"))
        audit_file_example(Path("output"))
    
    # Example 3: Custom configuration
    custom_pipeline = custom_configuration_example()
    
    print("\n" + "=" * 60)
    print("🎉 Examples completed!")
    print("\nNext steps:")
    print("1. Modify config files to customize theoretical frames")
    print("2. Run pipeline on your own policy brief PDFs")
    print("3. Analyze results using pandas or other tools")
    print("4. Check audit files for detailed extraction results")


if __name__ == "__main__":
    main()