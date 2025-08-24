#!/usr/bin/env python3
"""
Converts scraped GitHub PR JSON data to an improved, structured format
optimized for Retrieval-Augmented Generation (RAG) pipelines.

This script reads JSON files from an input directory, parses the monolithic 'diff'
string into a structured list of file changes, and adds other useful metadata.

How to run:
1. Make sure you have your scraped data in a directory (e.g., 'scraped_data').
2. Create an empty directory for the output (e.g., 'converted_data').
3. Run the script from your terminal:
   python convert_data.py scraped_data converted_data
"""

import json
import os
import re
import argparse
from pathlib import Path
from tqdm import tqdm # For a nice progress bar. Install with: pip install tqdm

def parse_diff(diff_text: str) -> list:
    """
    Parses a monolithic diff string into a list of structured file objects.

    Args:
        diff_text: The entire diff content as a single string.

    Returns:
        A list of dictionaries, where each dictionary represents one file's changes.
    """
    if not diff_text:
        return []

    # Split the main diff text into individual file diffs.
    # The delimiter is 'diff --git', so we split on that and re-add it.
    file_diffs = diff_text.split('\ndiff --git ')
    if len(file_diffs) > 1:
        # The first element is often empty or contains commit info, handle it
        if file_diffs[0].startswith('diff --git'):
             file_diffs = ['diff --git ' + f for f in file_diffs[1:]]
        else:
             file_diffs = ['diff --git ' + f for f in file_diffs[1:]]


    structured_files = []
    for chunk in file_diffs:
        if not chunk.strip():
            continue

        lines = chunk.split('\n')
        header = lines[0]
        
        # Extract file path from 'diff --git a/path/to/file b/path/to/file'
        # We primarily care about the 'b' path as it's the new state.
        path_match = re.search(r'b/(.*)', header)
        if not path_match:
            continue
        file_path = path_match.group(1)

        # Determine file status (added, deleted, modified)
        status = 'modified'
        if any('new file mode' in line for line in lines[:5]):
            status = 'added'
        elif any('deleted file mode' in line for line in lines[:5]):
            status = 'deleted'
        
        # Count additions and deletions for this specific file
        additions = 0
        deletions = 0
        for line in lines:
            if line.startswith('+') and not line.startswith('+++'):
                additions += 1
            elif line.startswith('-') and not line.startswith('---'):
                deletions += 1
        
        structured_files.append({
            "file_path": file_path,
            "status": status,
            "additions": additions,
            "deletions": deletions,
            "file_diff": chunk # The raw diff content for this file
        })
        
    return structured_files

def transform_pr_data(old_data: dict) -> dict:
    """
    Transforms a single PR data dictionary to the new, improved structure.

    Args:
        old_data: The original dictionary loaded from a scraped JSON file.

    Returns:
        A new dictionary with the RAG-optimized structure.
    """
    pr_number = old_data.get('pr_number')
    
    # Parse the monolithic diff into structured file data
    files_data = parse_diff(old_data.get('diff', ''))
    
    # Create the new data structure
    new_data = {
        # Core metadata
        "pr_number": pr_number,
        "url": f"https://github.com/ansible/ansible/pull/{pr_number}",
        "title": old_data.get('title'),
        "author_login": old_data.get('author_login'),
        
        # State and merge info
        "state": "MERGED" if old_data.get('merged_by_login') else "CLOSED",
        "merged_by_login": old_data.get('merged_by_login'),
        
        # Content
        "body": old_data.get('body'),
        "labels": old_data.get('labels', []),
        
        # NOTE: Timestamps like 'created_at' and 'merged_at' are not in the
        # original scrape, so they are omitted here. A re-scrape would be
        # needed to add them.
        
        # Aggregate stats
        "additions": old_data.get('additions', 0),
        "deletions": old_data.get('deletions', 0),
        "changed_files_count": old_data.get('changed_files', 0),
        
        # New structured file lists
        "changed_files_list": [f['file_path'] for f in files_data],
        "files": files_data,
        
        # Discussion
        "comments": old_data.get('comments', []),
        "reviews": old_data.get('reviews', [])
    }
    
    return new_data

def main():
    """Main function to run the conversion process."""
    input_path = Path("./scraped_data")
    output_path = Path("scraped_data_converted")
        
    # Create the output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"🤷 No JSON files found in '{input_path}'.")
        return
        
    print(f"Found {len(json_files)} JSON files. Starting conversion...")
    
    success_count = 0
    fail_count = 0
    
    # Process files with a progress bar
    for file_path in tqdm(json_files, desc="Converting files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # Transform the data
            converted_data = transform_pr_data(original_data)
            
            # Save the new file
            output_file_path = output_path / file_path.name
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed to process {file_path.name}: {e}")
            fail_count += 1
            
    print("\n" + "="*40)
    print("🎉 Conversion Complete!")
    print(f"✅ Successfully converted: {success_count} files")
    if fail_count > 0:
        print(f"❌ Failed to convert: {fail_count} files")
    print(f"📁 Converted data saved in: '{output_path}'")
    print("="*40)

if __name__ == "__main__":
    main()