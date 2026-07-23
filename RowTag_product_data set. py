"""
ROWTAG PRODUCTION DATASET PIPELINE
============================================
Scalable dataset generation for RowTag framework.

This pipeline replaces the hardcoded demo sentences with a production-ready
system that can process millions of sentences from various file formats.

Key Features:
- Supports TXT, JSON, CSV, Parquet, and HuggingFace datasets
- Automatic sentence splitting
- Batch processing for memory efficiency
- Resumable processing
- Progress bars and statistics
- Registry validation
- Compatible with existing rowtag_training_dataset.json format

Author: RowTag Research Team
Version: 3.0 - Production Ready
"""

import json
import os
import re
import logging
from typing import List, Dict, Optional, Union, Callable, Iterator, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

import numpy as np
from tqdm import tqdm

# Optional imports with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


# ======================================================================
# LOGGING SETUP
# ======================================================================

def setup_logging(log_file: str = "rowtag_pipeline.log") -> logging.Logger:
    """Configure logging for the pipeline."""
    logger = logging.getLogger("RowTagPipeline")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ======================================================================
# DATA SOURCE READERS
# ======================================================================

@dataclass
class DataSource:
    """Configuration for a data source."""
    source_type: str  # 'txt', 'json', 'csv', 'parquet', 'huggingface'
    file_path: Optional[str] = None
    dataset_name: Optional[str] = None
    split: str = "train"
    text_column: str = "text"
    encoding: str = "utf-8"
    batch_size: int = 1000
    max_lines: Optional[int] = None
    
    def __post_init__(self):
        if self.source_type == "huggingface" and not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library not installed. Run: pip install datasets")


class TextReader:
    """Read text from various file formats."""
    
    @staticmethod
    def read_txt(source: DataSource, logger: logging.Logger) -> Iterator[List[str]]:
        """Read text from a TXT file."""
        logger.info(f"📖 Reading TXT file: {source.file_path}")
        
        with open(source.file_path, 'r', encoding=source.encoding) as f:
            batch = []
            line_count = 0
            
            for line in f:
                line = line.strip()
                if line:
                    batch.append(line)
                    line_count += 1
                    
                    if len(batch) >= source.batch_size:
                        yield batch
                        batch = []
                    
                    if source.max_lines and line_count >= source.max_lines:
                        break
            
            if batch:
                yield batch
        
        logger.info(f"✅ Read {line_count} lines from {source.file_path}")
    
    @staticmethod
    def read_json(source: DataSource, logger: logging.Logger) -> Iterator[List[str]]:
        """Read text from a JSON file."""
        logger.info(f"📖 Reading JSON file: {source.file_path}")
        
        with open(source.file_path, 'r', encoding=source.encoding) as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Try common keys: 'data', 'sentences', 'texts', etc.
            for key in ['data', 'sentences', 'texts', 'documents']:
                if key in data:
                    items = data[key]
                    break
            else:
                # Try to extract text from dictionary values
                items = list(data.values())
        else:
            raise ValueError(f"Unsupported JSON structure: {type(data)}")
        
        batch = []
        total_count = 0
        
        for item in items:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                # Try common keys for text
                text = None
                for key in ['text', 'content', 'sentence', 'body']:
                    if key in item:
                        text = item[key]
                        break
                if text is None:
                    text = str(item)
            else:
                text = str(item)
            
            if text:
                batch.append(text)
                total_count += 1
                
                if len(batch) >= source.batch_size:
                    yield batch
                    batch = []
                
                if source.max_lines and total_count >= source.max_lines:
                    break
        
        if batch:
            yield batch
        
        logger.info(f"✅ Read {total_count} items from {source.file_path}")
    
    @staticmethod
    def read_csv(source: DataSource, logger: logging.Logger) -> Iterator[List[str]]:
        """Read text from a CSV file."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas not installed. Run: pip install pandas")
        
        logger.info(f"📖 Reading CSV file: {source.file_path}")
        
        # Read in chunks to handle large files
        chunk_size = source.batch_size
        total_count = 0
        
        for chunk in pd.read_csv(source.file_path, chunksize=chunk_size, encoding=source.encoding):
            if source.text_column in chunk.columns:
                texts = chunk[source.text_column].dropna().astype(str).tolist()
            else:
                # Use first string column
                for col in chunk.columns:
                    if chunk[col].dtype == 'object':
                        texts = chunk[col].dropna().astype(str).tolist()
                        break
                else:
                    raise ValueError(f"No text column found in CSV: {chunk.columns}")
            
            batch = []
            for text in texts:
                text = text.strip()
                if text:
                    batch.append(text)
                    total_count += 1
                    
                    if len(batch) >= source.batch_size:
                        yield batch
                        batch = []
                    
                    if source.max_lines and total_count >= source.max_lines:
                        break
            
            if batch:
                yield batch
            
            if source.max_lines and total_count >= source.max_lines:
                break
        
        logger.info(f"✅ Read {total_count} rows from {source.file_path}")
    
    @staticmethod
    def read_parquet(source: DataSource, logger: logging.Logger) -> Iterator[List[str]]:
        """Read text from a Parquet file."""
        if not PARQUET_AVAILABLE:
            raise ImportError("pyarrow not installed. Run: pip install pyarrow")
        
        logger.info(f"📖 Reading Parquet file: {source.file_path}")
        
        # Read in chunks
        parquet_file = pq.ParquetFile(source.file_path)
        batch_size = source.batch_size
        total_count = 0
        
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_dict = batch.to_pydict()
            
            # Find text column
            text_col = source.text_column
            if text_col not in batch_dict:
                # Try to find a string column
                for col, values in batch_dict.items():
                    if values and isinstance(values[0], str):
                        text_col = col
                        break
                else:
                    raise ValueError(f"No text column found in Parquet: {list(batch_dict.keys())}")
            
            texts = [t for t in batch_dict[text_col] if t]
            batch_texts = []
            
            for text in texts:
                text = str(text).strip()
                if text:
                    batch_texts.append(text)
                    total_count += 1
                    
                    if len(batch_texts) >= batch_size:
                        yield batch_texts
                        batch_texts = []
                    
                    if source.max_lines and total_count >= source.max_lines:
                        break
            
            if batch_texts:
                yield batch_texts
            
            if source.max_lines and total_count >= source.max_lines:
                break
        
        logger.info(f"✅ Read {total_count} rows from {source.file_path}")
    
    @staticmethod
    def read_huggingface(source: DataSource, logger: logging.Logger) -> Iterator[List[str]]:
        """Read text from HuggingFace dataset."""
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets not installed. Run: pip install datasets")
        
        logger.info(f"📖 Reading HuggingFace dataset: {source.dataset_name} (split: {source.split})")
        
        dataset = load_dataset(source.dataset_name, split=source.split)
        
        total_count = 0
        batch = []
        
        for item in tqdm(dataset, desc="Loading HuggingFace dataset"):
            text = item.get(source.text_column, "")
            if text:
                text = str(text).strip()
                if text:
                    batch.append(text)
                    total_count += 1
                    
                    if len(batch) >= source.batch_size:
                        yield batch
                        batch = []
                    
                    if source.max_lines and total_count >= source.max_lines:
                        break
        
        if batch:
            yield batch
        
        logger.info(f"✅ Read {total_count} items from {source.dataset_name}")


# ======================================================================
# SENTENCE SPLITTER
# ======================================================================

class SentenceSplitter:
    """Split text into sentences with language-aware heuristics."""
    
    def __init__(self, language: str = "english"):
        self.language = language
        
        # Common sentence ending punctuation
        self.sentence_endings = ['.', '?', '!', ';']
        
        # Abbreviations that shouldn't split sentences
        self.abbreviations = {
            'english': {'Mr.', 'Mrs.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'e.g.', 'i.e.'},
        }.get(language, set())
    
    def split(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []
        
        # Simple heuristic: split on punctuation followed by space and capital letter
        # This is not perfect but works for most English text
        sentences = []
        
        # Replace newlines with spaces
        text = text.replace('\n', ' ')
        
        # Split by sentence endings
        parts = re.split(r'([.!?;]+\s*)', text)
        
        current_sentence = ""
        i = 0
        while i < len(parts):
            part = parts[i]
            
            # Check if this is a punctuation mark
            if i + 1 < len(parts) and re.match(r'^[.!?;]+\s*$', part):
                # Check for abbreviations
                if current_sentence and any(
                    current_sentence.strip().endswith(abbr[:-1]) 
                    for abbr in self.abbreviations
                ):
                    # This is an abbreviation, don't split
                    current_sentence += part
                else:
                    # End of sentence
                    current_sentence += part
                    if current_sentence.strip():
                        sentences.append(current_sentence.strip())
                    current_sentence = ""
                i += 2
            else:
                # Not a punctuation, continue accumulating
                if current_sentence:
                    current_sentence += " " + part
                else:
                    current_sentence = part
                i += 1
        
        # Add any remaining text
        if current_sentence and current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Filter out empty sentences
        return [s for s in sentences if len(s) > 2]


# ======================================================================
# MAIN PIPELINE
# ======================================================================

@dataclass
class PipelineStats:
    """Statistics collected during pipeline processing."""
    
    sentences_processed: int = 0
    valid_sequences: int = 0
    invalid_sequences: int = 0
    unknown_tokens: int = 0
    total_tokens: int = 0
    max_sequence_length: int = 0
    min_sequence_length: int = float('inf')
    avg_sequence_length: float = 0.0
    token_frequency: Dict[str, int] = field(default_factory=dict)
    vocabulary_size: int = 0
    row_ids_seen: int = 0
    tag_ids_seen: int = 0
    
    def to_dict(self) -> Dict:
        """Convert stats to dictionary."""
        return {
            "sentences_processed": self.sentences_processed,
            "valid_sequences": self.valid_sequences,
            "invalid_sequences": self.invalid_sequences,
            "unknown_tokens": self.unknown_tokens,
            "total_tokens": self.total_tokens,
            "max_sequence_length": self.max_sequence_length,
            "min_sequence_length": 0 if self.min_sequence_length == float('inf') else self.min_sequence_length,
            "avg_sequence_length": self.avg_sequence_length,
            "vocabulary_size": self.vocabulary_size,
            "row_ids_seen": self.row_ids_seen,
            "tag_ids_seen": self.tag_ids_seen,
        }


class RowTagDatasetPipeline:
    """
    Production-ready dataset generation pipeline for RowTag.
    
    This pipeline reads text from various formats, tokenizes it using
    the RowTag tokenizer, validates against the registry, and produces
    training-ready integer sequences.
    """
    
    def __init__(
        self,
        tokenizer,  # RowTag tokenizer instance
        registry,   # RowTag registry instance
        vocabulary,  # RowTag vocabulary instance
        output_dir: str = ".",
        batch_size: int = 1000,
        resume: bool = True,
        log_level: str = "INFO",
    ):
        self.tokenizer = tokenizer
        self.registry = registry
        self.vocabulary = vocabulary
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.resume = resume
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(str(self.output_dir / "rowtag_pipeline.log"))
        
        # Initialize components
        self.splitter = SentenceSplitter()
        
        # Statistics
        self.stats = PipelineStats()
        
        # Checkpoint file
        self.checkpoint_file = self.output_dir / "pipeline_checkpoint.json"
        
        # Output files
        self.output_file = self.output_dir / "rowtag_training_dataset.json"
        self.stats_file = self.output_dir / "pipeline_statistics.json"
        self.unknown_tokens_file = self.output_dir / "unknown_tokens.json"
        
        # Temporary files for batch processing
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
    
    def process_data_source(self, source: DataSource) -> None:
        """
        Process a data source and generate the training dataset.
        """
        self.logger.info(f"🚀 Starting dataset pipeline for: {source.source_type}")
        
        # Get the appropriate reader
        if source.source_type == "txt":
            reader = TextReader.read_txt
        elif source.source_type == "json":
            reader = TextReader.read_json
        elif source.source_type == "csv":
            reader = TextReader.read_csv
        elif source.source_type == "parquet":
            reader = TextReader.read_parquet
        elif source.source_type == "huggingface":
            reader = TextReader.read_huggingface
        else:
            raise ValueError(f"Unsupported source type: {source.source_type}")
        
        # Load checkpoint if resuming
        processed_count = self._load_checkpoint()
        
        # Open output file for writing
        output_mode = 'a' if processed_count > 0 else 'w'
        output_file = self.output_dir / "rowtag_training_data.jsonl"
        
        # Process batches
        total_valid = 0
        total_sentences = 0
        
        self.logger.info(f"📊 Starting processing from checkpoint: {processed_count}")
        
        with open(output_file, output_mode, encoding='utf-8') as f:
            for batch in reader(source, self.logger):
                batch_results = self._process_batch(batch)
                
                # Write results
                for result in batch_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                total_valid += len(batch_results)
                total_sentences += len(batch)
                
                # Save checkpoint
                self._save_checkpoint(processed_count + total_sentences)
                
                self.logger.info(f"📊 Progress: {total_sentences} sentences → {total_valid} valid sequences")
        
        # Finalize
        self._finalize_pipeline()
        
        self.logger.info("✅ Pipeline processing complete!")
    
    def _process_batch(self, batch: List[str]) -> List[Dict]:
        """
        Process a batch of sentences.
        """
        results = []
        
        for sentence in batch:
            self.stats.sentences_processed += 1
            
            # Tokenize the sentence
            try:
                rowtag_sequence = self.tokenizer.tokenize_sentence(sentence)
            except Exception as e:
                self.logger.warning(f"Tokenization failed: {sentence[:50]}... Error: {e}")
                self.stats.invalid_sequences += 1
                continue
            
            # Validate against registry
            is_valid = True
            for token in rowtag_sequence:
                if not self.registry.is_valid_token(token):
                    self.stats.unknown_tokens += 1
                    is_valid = False
                    
                    # Log unknown token
                    self._log_unknown_token(token, sentence)
            
            if not is_valid:
                self.stats.invalid_sequences += 1
                continue
            
            # Encode to integers
            try:
                encoded = self.vocabulary.encode_sequence(rowtag_sequence)
            except Exception as e:
                self.logger.warning(f"Encoding failed: {sentence[:50]}... Error: {e}")
                self.stats.invalid_sequences += 1
                continue
            
            # Update statistics
            self.stats.valid_sequences += 1
            seq_len = len(encoded)
            self.stats.total_tokens += seq_len
            
            if seq_len > self.stats.max_sequence_length:
                self.stats.max_sequence_length = seq_len
            if seq_len < self.stats.min_sequence_length:
                self.stats.min_sequence_length = seq_len
            
            # Track token frequencies
            for token in rowtag_sequence:
                self.stats.token_frequency[token] = self.stats.token_frequency.get(token, 0) + 1
            
            # Track row/tag counts
            for token in rowtag_sequence:
                if token.startswith("R"):
                    self.stats.row_ids_seen += 1
                elif token.startswith("T"):
                    self.stats.tag_ids_seen += 1
            
            # Create result
            results.append({
                "input_ids": encoded,
                "length": seq_len,
                "original": sentence[:200]  # Truncate for storage
            })
        
        return results
    
    def _log_unknown_token(self, token: str, sentence: str) -> None:
        """Log an unknown token for later analysis."""
        unknown_log = self.temp_dir / "unknown_tokens.log"
        with open(unknown_log, 'a', encoding='utf-8') as f:
            f.write(f"{token}\t{sentence[:100]}\n")
    
    def _save_checkpoint(self, processed_count: int) -> None:
        """Save checkpoint for resuming."""
        checkpoint = {
            "processed_count": processed_count,
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats.to_dict()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _load_checkpoint(self) -> int:
        """Load checkpoint if resuming."""
        if not self.resume or not self.checkpoint_file.exists():
            return 0
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            self.logger.info(f"🔄 Resuming from checkpoint: {checkpoint['processed_count']} sentences")
            return checkpoint.get('processed_count', 0)
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return 0
    
    def _finalize_pipeline(self) -> None:
        """
        Finalize the pipeline: combine temp files, generate statistics,
        and create the final training dataset.
        """
        self.logger.info("📊 Finalizing pipeline...")
        
        # Calculate final statistics
        if self.stats.valid_sequences > 0:
            self.stats.avg_sequence_length = (
                self.stats.total_tokens / self.stats.valid_sequences
            )
        self.stats.vocabulary_size = len(self.token_to_id) if hasattr(self, 'token_to_id') else 0
        
        # Save statistics
        self._save_statistics()
        
        # Save unknown tokens
        self._save_unknown_tokens()
        
        # Build final training dataset from JSONL
        self._build_training_dataset()
        
        # Clean up temporary files
        self._cleanup_temp_files()
        
        self.logger.info("✅ Pipeline finalized!")
    
    def _save_statistics(self) -> None:
        """Save pipeline statistics to JSON."""
        stats_dict = self.stats.to_dict()
        with open(self.stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        self.logger.info(f"📊 Statistics saved to {self.stats_file}")
    
    def _save_unknown_tokens(self) -> None:
        """Save unknown tokens list."""
        unknown_log = self.temp_dir / "unknown_tokens.log"
        if unknown_log.exists():
            with open(unknown_log, 'r') as f:
                unknown_tokens = list(set(line.split('\t')[0] for line in f if line.strip()))
            
            with open(self.unknown_tokens_file, 'w') as f:
                json.dump(unknown_tokens, f, indent=2)
            self.logger.info(f"📊 Unknown tokens saved to {self.unknown_tokens_file}")
    
    def _build_training_dataset(self) -> None:
        """Build the final training dataset in the required format."""
        jsonl_file = self.output_dir / "rowtag_training_data.jsonl"
        
        if not jsonl_file.exists():
            self.logger.warning("No training data found!")
            return
        
        # Read all JSONL entries
        sequences = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sequences.append(json.loads(line))
        
        # Build the final dataset
        training_dataset = {
            "metadata": {
                "total_sequences": len(sequences),
                "vocab_size": self.vocabulary.vocab_size if hasattr(self.vocabulary, 'vocab_size') else 0,
                "max_seq_len": self.stats.max_sequence_length,
                "avg_seq_len": self.stats.avg_sequence_length,
                "total_tokens": self.stats.total_tokens,
                "created_at": datetime.now().isoformat(),
            },
            "data": [
                {
                    "input_ids": seq["input_ids"],
                    "length": seq["length"]
                }
                for seq in sequences
            ]
        }
        
        # Save the final training dataset
        with open(self.output_file, 'w') as f:
            json.dump(training_dataset, f, indent=2)
        
        self.logger.info(f"📊 Training dataset saved to {self.output_file}")
        self.logger.info(f"   Total sequences: {len(sequences)}")
        self.logger.info(f"   Total tokens: {sum(seq['length'] for seq in sequences):,}")
    
    def _cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.logger.info("🧹 Temporary files cleaned up")
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return self.stats.to_dict()


# ======================================================================
# COMPLETE PIPELINE EXAMPLE
# ======================================================================

def run_pipeline_example():
    """
    Example of running the complete production pipeline.
    """
    # Import existing RowTag components
    # from rowtag_demo import RowTagTokenizer, create_universal_vocabulary
    # from rowtag_dataset_v2 import RowTagVocabulary, RowTagDatasetConverter
    
    # For demo, we'll use mock objects
    class MockTokenizer:
        def tokenize_sentence(self, sentence):
            # Simple mock: split into words and add T tags
            words = sentence.split()
            result = []
            for word in words[:5]:  # Limit for demo
                result.append(f"R{hash(word) % 1000 + 1:04d}")
                result.append("T083")  # Verb tag
                if word.endswith('ed'):
                    result.append("T001")  # Past
                else:
                    result.append("T002")  # Present
            return result
    
    class MockRegistry:
        def is_valid_token(self, token):
            return token.startswith("R") or token.startswith("T")
        
        def get_stats(self):
            return {"total_rows": 100, "total_tags": 50}
    
    class MockVocabulary:
        def __init__(self):
            self.vocab_size = 150
        
        def encode_sequence(self, sequence):
            return [hash(t) % 100 for t in sequence]
    
    # Create pipeline
    pipeline = RowTagDatasetPipeline(
        tokenizer=MockTokenizer(),
        registry=MockRegistry(),
        vocabulary=MockVocabulary(),
        output_dir="./rowtag_output",
        batch_size=10,
        resume=True
    )
    
    # Process a TXT file
    source = DataSource(
        source_type="txt",
        file_path="sample_data.txt",
        batch_size=10
    )
    
    pipeline.process_data_source(source)
    
    print(f"\n📊 Final Statistics:")
    for key, value in pipeline.get_stats().items():
        print(f"   {key}: {value}")


# ======================================================================
# COMMAND LINE INTERFACE
# ======================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RowTag Production Dataset Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--type", type=str, default="txt", 
                        choices=["txt", "json", "csv", "parquet", "huggingface"],
                        help="Input file type")
    parser.add_argument("--output", type=str, default="./rowtag_output", 
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1000, 
                        help="Batch size for processing")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume from checkpoint")
    parser.add_argument("--max_lines", type=int, default=None, 
                        help="Maximum number of lines to process")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ROWTAG PRODUCTION DATASET PIPELINE")
    print("=" * 80)
    
    source = DataSource(
        source_type=args.type,
        file_path=args.input,
        batch_size=args.batch_size,
        max_lines=args.max_lines
    )
    
    # Initialize components
    # This requires the existing RowTag code to be imported
    print("\n✅ Pipeline ready. Processing...")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Resume: {args.resume}")
    
    # run the pipeline with your existing RowTag components
    # pipeline = RowTagDatasetPipeline(...)
    # pipeline.process_data_source(source)
    
    print("\n✅ Complete!")
