#!/usr/bin/env python3
"""
Test script to verify file document creation in ColabPRIndexer
"""

import os
import sys
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from colab_indexer import ColabPRIndexer, ColabIndexingConfig

def test_file_document_creation():
    """Test that file documents are being created correctly."""
    print("🧪 Testing File Document Creation")
    print("=" * 50)
    
    # Sample PR data matching your JSON structure
    sample_pr_data = {
        "pr_number": 85673,
        "url": "https://github.com/ansible/ansible/pull/85673",
        "title": "vars deprecation",
        "author_login": "bcoca",
        "state": "MERGED",
        "merged_by_login": "bcoca",
        "body": "Removing no longer use internal  `vars`  cache dictionary\r\n\r\n##### ISSUE TYPE\r\n\r\n- Feature Pull Request\r\n",
        "labels": ["feature"],
        "additions": 31,
        "deletions": 2,
        "changed_files_count": 4,
        "changed_files_list": [
            "ansible/vars/manager.py b/lib/ansible/vars/manager.py",
            "test/integration/targets/var_templating/runme.sh",
            "test/integration/targets/var_templating/vars_deprecation.yml"
        ],
        "files": [
            {
                "file_path": "ansible/vars/manager.py b/lib/ansible/vars/manager.py",
                "status": "modified",
                "additions": 10,
                "deletions": 2,
                "file_diff": "diff --git a/lib/ansible/vars/manager.py b/lib/ansible/vars/manager.py\nindex e026ff50ec8d56..37615a4a704dc9 100644\n--- a/lib/ansible/vars/manager.py\n+++ b/lib/ansible/vars/manager.py\n@@ -56,6 +56,12 @@\n     deprecator=_deprecator.ANSIBLE_CORE_DEPRECATOR,\n     help_text='Use `ansible_facts[\"fact_name\"]` (no `ansible_` prefix) instead.',\n )\n+_DEPRECATE_VARS = _tags.Deprecated(\n+    msg='The internal \"vars\" dictionary is deprecated.',\n+    version='2.24',\n+    deprecator=_deprecator.ANSIBLE_CORE_DEPRECATOR,\n+    help_text='Use the `vars` and `varnames` lookups instead.',\n+)\n"
            },
            {
                "file_path": "test/integration/targets/var_templating/runme.sh",
                "status": "modified",
                "additions": 3,
                "deletions": 0,
                "file_diff": "diff --git a/test/integration/targets/var_templating/runme.sh b/test/integration/targets/var_templating/runme.sh\nindex 69782f112a077d..591de0124612f8 100755\n--- a/test/integration/targets/var_templating/runme.sh\n+++ b/test/integration/targets/var_templating/runme.sh\n@@ -13,3 +13,6 @@ ansible-playbook task_vars_templating.yml -v \"$@\"\n \n # there should be an attempt to use 'sudo' in the connection debug output\n ANSIBLE_BECOME_ALLOW_SAME_USER=true ansible-playbook test_connection_vars.yml -vvvv \"$@\" | tee /dev/stderr | grep 'sudo \\-H \\-S'\n+\n+# test vars deprecation\n+ANSIBLE_DEPRECATION_WARNINGS=1 ansible-playbook vars_deprecation.yml \"$@\""
            }
        ],
        "comments": [
            {
                "author_login": "webknjaz",
                "body": "/azp run"
            },
            {
                "author_login": "azure-pipelines[bot]",
                "body": "<samp>\nAzure Pipelines successfully started running 1 pipeline(s).<br>\r\n\n</samp>"
            }
        ],
        "reviews": []
    }
    
    try:
        # Create a test configuration
        config = ColabIndexingConfig(
            data_dir="test_data",  # This won't exist, but we can test the document creation
            output_dir="test_output"
        )
        
        # Initialize the indexer (this will set up logging)
        print("📝 Setting up logging...")
        indexer = ColabPRIndexer(config)
        
        print(f"✅ Logging setup complete!")
        print(f"📁 Log file location: {indexer.log_file}")
        
        # Test the document creation
        print("\n🧪 Testing document creation...")
        docs = indexer.format_pr_to_document(sample_pr_data)
        
        print(f"✅ Created {len(docs)} documents")
        
        # Analyze what was created
        summary_docs = [d for d in docs if d.metadata.get('document_type') == 'summary_discussion']
        file_docs = [d for d in docs if d.metadata.get('document_type') == 'file_change']
        
        print(f"📄 Summary documents: {len(summary_docs)}")
        print(f"📁 File change documents: {len(file_docs)}")
        
        # Show details of each document type
        if summary_docs:
            print(f"\n📋 Summary document metadata:")
            for i, doc in enumerate(summary_docs):
                print(f"  {i+1}. Type: {doc.metadata.get('document_type')}")
                print(f"     PR: #{doc.metadata.get('pr_number')}")
                print(f"     Title: {doc.metadata.get('title')}")
                print(f"     Content length: {doc.metadata.get('content_length')} chars")
        
        if file_docs:
            print(f"\n📁 File change document metadata:")
            for i, doc in enumerate(file_docs):
                print(f"  {i+1}. Type: {doc.metadata.get('document_type')}")
                print(f"     PR: #{doc.metadata.get('pr_number')}")
                print(f"     File: {doc.metadata.get('file_path')}")
                print(f"     Status: {doc.metadata.get('file_status')}")
                print(f"     Content length: {doc.metadata.get('content_length')} chars")
        
        # Check the log file to see what was logged
        if indexer.log_file.exists():
            print(f"\n📖 Log file contents:")
            print("-" * 50)
            with open(indexer.log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content)
        
        return len(docs), len(summary_docs), len(file_docs)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0

if __name__ == "__main__":
    total_docs, summary_docs, file_docs = test_file_document_creation()
    
    print(f"\n🎯 Test Results:")
    print(f"   Total documents: {total_docs}")
    print(f"   Summary documents: {summary_docs}")
    print(f"   File documents: {file_docs}")
    
    if file_docs > 0:
        print("✅ File documents are being created correctly!")
    else:
        print("❌ No file documents were created - there's still an issue!")
