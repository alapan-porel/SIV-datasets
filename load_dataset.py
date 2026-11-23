"""
Clinical SIV Mental Health Dataset Loader & Pipeline Utility
============================================================
Provides convenient Python methods to stream or load any stage of the
Mental Health & Clinical Psychotherapy SIV Dialogue Corpus into Pandas DataFrames
or Hugging Face `datasets.Dataset` objects.

Usage:
------
    from load_dataset import load_stage, load_all_siv

    # Load Stage 4 ready-to-train SFT data
    df_sft = load_stage(4, as_dataframe=True)
    print(df_sft.head())

    # Stream Stage 2 Emotional SIV dialogues
    for record in load_stage(2, stream=True):
        print(record['siv_metadata']['observed_state'])
"""

import os
import json
import argparse
from typing import Dict, Any, Generator, Optional, Union

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STAGE_FILES = {
    1: os.path.join(BASE_DIR, '01_raw_dialogue_drafts', 'stage1_drafts_merged.jsonl'),
    2: os.path.join(BASE_DIR, '02_emotional_siv_dialogues', 'siv_emotional_dialogues_merged.jsonl'),
    3: os.path.join(BASE_DIR, '03_clinical_protocol_dialogues', 'siv_clinical_dialogues_v1.jsonl'),
    4: os.path.join(BASE_DIR, '04_instruction_tuning_sft', 'siv_training_data_v1.jsonl')
}

STAGE_ALIASES = {
    'stage1': 1, 'drafts': 1, 'raw': 1,
    'stage2': 2, 'emotional': 2, 'siv_emotional': 2,
    'stage3': 3, 'clinical': 3, 'protocols': 3,
    'stage4': 4, 'sft': 4, 'training': 4, 'instruction': 4
}


def resolve_stage(stage: Union[int, str]) -> int:
    if isinstance(stage, int) and stage in STAGE_FILES:
        return stage
    if isinstance(stage, str):
        key = stage.lower().strip()
        if key.isdigit() and int(key) in STAGE_FILES:
            return int(key)
        if key in STAGE_ALIASES:
            return STAGE_ALIASES[key]
    raise ValueError(f"Unknown stage '{stage}'. Choose from 1 (drafts), 2 (emotional), 3 (clinical), 4 (sft).")


def stream_stage(stage: Union[int, str]) -> Generator[Dict[str, Any], None, None]:
    """Yield records one by one from a specified dataset stage."""
    stage_id = resolve_stage(stage)
    file_path = STAGE_FILES[stage_id]
    
    # Fallback to root level if not in stage subfolder
    if not os.path.exists(file_path):
        fallback = os.path.join(BASE_DIR, os.path.basename(file_path))
        if os.path.exists(fallback):
            file_path = fallback
            
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                yield json.loads(line_str)


def load_stage(stage: Union[int, str], as_dataframe: bool = True, limit: Optional[int] = None):
    """
    Load a dataset stage into a Pandas DataFrame, a Hugging Face Dataset, or a Python list.
    """
    records = []
    for i, rec in enumerate(stream_stage(stage)):
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break

    if not as_dataframe:
        return records

    try:
        import pandas as pd
        df = pd.DataFrame(records)
        return df
    except ImportError:
        print("[Warning] Pandas is not installed. Returning raw list of dictionaries.")
        return records


def load_hf_dataset(stage: Union[int, str]):
    """Load a dataset stage directly into a Hugging Face `datasets.Dataset`."""
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("Please install datasets via `pip install datasets` to use this function.")
    
    stage_id = resolve_stage(stage)
    file_path = STAGE_FILES[stage_id]
    return Dataset.from_json(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clinical SIV Mental Health Dataset Inspector")
    parser.add_argument('--stage', type=str, default='4', help="Dataset stage (1, 2, 3, 4 or names: drafts, emotional, clinical, sft)")
    parser.add_argument('--limit', type=int, default=3, help="Number of records to display")
    parser.add_argument('--info', action='store_true', help="Print summary of all available stages")
    args = parser.parse_args()

    if args.info:
        manifest_path = os.path.join(BASE_DIR, 'dataset_manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            print(f"=== {info['dataset_name']} ===")
            print(info['description'])
            print(f"Total Corpus Rows: {info['total_corpus_rows']:,}\n")
            for s in info['stages']:
                print(f"[{s['stage_name']}] - {s['row_count']:,} rows ({s['size_mb']} MB)")
                print(f"  Path: {s['relative_path']}")
                print(f"  Purpose: {s['purpose']}\n")
        else:
            print("dataset_manifest.json not found.")
    else:
        stage_num = resolve_stage(args.stage)
        print(f"\n--- Displaying {args.limit} records from Stage {stage_num} ---")
        for idx, item in enumerate(stream_stage(stage_num)):
            if idx >= args.limit:
                break
            print(f"\n[Record #{idx + 1}]")
            print(json.dumps(item, indent=2, ensure_ascii=False)[:600] + "\n...")
