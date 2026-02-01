#!/usr/bin/env python3
"""
Script to count individual input IDs across all JSONL files in the cache folder.
Uses streaming to avoid loading entire files into RAM.
"""

import json
import glob
from pathlib import Path
from collections import defaultdict

def count_input_ids_in_jsonl(file_path):
    """
    Count total input IDs in a JSONL file by reading line by line.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Integer count of total input IDs in the file
    """
    total_count = 0
    line_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    data = json.loads(line)
                    if 'input_ids' in data and isinstance(data['input_ids'], list):
                        total_count += len(data['input_ids'])
                        line_count += 1
                except json.JSONDecodeError as e:
                    print(f"  Warning: Could not parse line in {file_path}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0
    
    return total_count

def main():
    """Main function to scan cache directory and count input IDs."""
    
    # Find all JSONL files in the cache directory
    cache_dir = Path('/root/Latent-Space-Model/cache')
    jsonl_files = sorted(cache_dir.glob('**/*.jsonl'))
    
    if not jsonl_files:
        print(f"No JSONL files found in {cache_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL file(s)\n")
    print("=" * 70)
    
    file_counts = {}
    total_ids = 0
    
    # Process each JSONL file
    for file_path in jsonl_files:
        relative_path = file_path.relative_to(cache_dir)
        print(f"\nProcessing: {relative_path}")
        
        count = count_input_ids_in_jsonl(file_path)
        file_counts[str(relative_path)] = count
        total_ids += count
        
        print(f"  Input IDs: {count:,}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("\nSUMMARY")
    print("-" * 70)
    
    for file_name in sorted(file_counts.keys()):
        count = file_counts[file_name]
        print(f"{file_name:<50} {count:>15,}")
    
    print("-" * 70)
    print(f"{'TOTAL':<50} {total_ids:>15,}")
    print("=" * 70)

if __name__ == '__main__':
    main()
