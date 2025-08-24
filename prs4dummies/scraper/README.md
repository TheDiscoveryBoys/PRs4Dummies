# Ansible GitHub Repository Scraper

This script scrapes the Ansible GitHub repository to extract pull request data for use in a vector store RAG system.

## Features

- **Repository Cloning**: Clones the Ansible repository locally for access to all code files
- **Pull Request Fetching**: Efficiently retrieves the most recent N **merged** pull requests using GitHub Search API (configurable)
- **Comprehensive Data Extraction**: For each PR, extracts:
  - Title and body description
  - All comments from the conversation thread
  - Review comments (inline code comments)
  - Code reviews
  - The complete diff file with exact code changes
- **Structured Output**: Saves data as JSON documents, one per PR, optimized for LLM consumption
- **LLM-Optimized Format**: Excludes metadata noise (IDs, timestamps, URLs) and focuses on meaningful content

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional but recommended) Set up a GitHub personal access token:
   - Get a token from https://github.com/settings/tokens
   - This increases rate limits from 60 to 5,000 requests per hour
   - Basic read permissions are sufficient
   
   **Option 1: .env file (recommended)**
   Create a `.env` file in the same directory as the scraper:
   ```
   GITHUB_TOKEN=your_token_here
   ```
   
   Note: You can also use `GITHUB_PERSONAL_ACCESS_TOKEN` as the environment variable name.
   
   **Option 2: Environment variable**
   ```bash
   export GITHUB_TOKEN=your_token_here
   ```
   
   **Option 3: Command line argument**
   Use the `--token` argument (see usage examples below)

## Usage

### Basic Usage (with .env file)
```bash
python scraper.py --num-prs 50
```

### Without .env file (using command line token)
```bash
python scraper.py --token YOUR_GITHUB_TOKEN --num-prs 100
```

### No token (rate limited)
```bash
python scraper.py --num-prs 10
```

### Include unmerged PRs
```bash
python scraper.py --num-prs 50 --include-unmerged
```

### Advanced Usage
```bash
python scraper.py \
  --token YOUR_GITHUB_TOKEN \
  --num-prs 100 \
  --output-dir my_scraped_data \
  --repo-dir my_ansible_repo \
  --force-refresh
```

## Command Line Options

- `--token`: GitHub personal access token (overrides .env file and environment variables)
- `--num-prs`: Number of pull requests to scrape (default: 50)
- `--output-dir`: Directory to save scraped data (default: scraped_data)
- `--repo-dir`: Directory to clone the repository (default: ansible_repo)
- `--no-clone`: Skip cloning the repository locally
- `--force-refresh`: Force refresh of the local repository (delete and re-clone)
- `--include-unmerged`: Include unmerged/closed PRs (default: only merged PRs)

## Output Structure

The scraper creates:

1. **Individual PR files**: `pr_{number}.json` - Complete data for each PR
2. **Summary file**: `scrape_summary.json` - Overview of all scraped PRs
3. **Log file**: `scraper.log` - Detailed execution logs

### Example PR JSON Structure (LLM-Optimized)
```json
{
  "pr_number": 123,
  "title": "Fix authentication bug",
  "body": "This PR fixes the issue where users couldn't log in...",
  "author_login": "username",
  "merged_by_login": "maintainer",
  "labels": ["bugfix", "authentication", "high-priority"],
  "additions": 45,
  "deletions": 12,
  "changed_files": 3,
  "comments": [
    {
      "author_login": "reviewer",
      "body": "Looks good! Just one small suggestion..."
    },
    {
      "author_login": "reviewer",
      "body": "Perfect fix, thanks!",
      "file_path": "auth/login.py",
      "diff_hunk": "@@ -15,7 +15,7 @@...",
      "comment_type": "review_comment"
    }
  ],
  "reviews": [
    {
      "author_login": "maintainer",
      "body": "Great work on this fix!",
      "state": "APPROVED"
    }
  ],
  "diff": "diff --git a/auth/login.py b/auth/login.py\nindex 1234567..abcdefg 100644\n--- a/auth/login.py\n+++ b/auth/login.py\n@@ -15,7 +15,7 @@ def authenticate(user):\n-    if user.password == stored_password:\n+    if hash(user.password) == stored_hash:\n         return True"
}
```

## Rate Limits

- **Without token**: 60 requests per hour
- **With token**: 5,000 requests per hour

For scraping many PRs, a GitHub token is highly recommended.

## Error Handling

The scraper includes comprehensive error handling:
- Continues processing if individual PRs fail
- Logs all errors to `scraper.log`
- Gracefully handles network issues and API limits
- Skips corrupted or inaccessible PRs

## Why Merged PRs Only?

By default, the scraper only includes **successfully merged** pull requests because:

- **Quality Assurance**: Merged PRs represent accepted solutions and best practices
- **Relevance**: The code changes and discussions led to actual improvements
- **Trust**: RAG systems should learn from vetted, approved changes
- **Efficiency**: Reduces noise from abandoned or rejected proposals
- **API Efficiency**: Uses GitHub Search API to filter merged PRs server-side, not client-side

You can include unmerged PRs with `--include-unmerged` if needed for research purposes.

## Data Optimization for LLMs

The scraper outputs **LLM-optimized JSON** by excluding metadata that creates noise:

**❌ Excluded (low-signal data)**:
- Machine IDs: `id`, `node_id`, `head_sha`, `base_sha`, `merge_commit_sha`
- Timestamps: `created_at`, `updated_at`, `closed_at`, `merged_at`
- URLs: `url`, `html_url`, `issue_url`
- Project management: `assignees`, `milestone`, `state`
- Nested user objects with redundant `id` fields

**✅ Included (meaningful content)**:
- Core text: `title`, `body`, comment `body`, review `body`
- Context: `author_login`, `merged_by_login`, `labels`
- Code metrics: `additions`, `deletions`, `changed_files`
- Review states: `APPROVED`, `CHANGES_REQUESTED`, `COMMENTED`
- File context: `file_path`, `diff_hunk` for inline comments
- Complete `diff` content as **plain text** (actual code changes, not JSON)

## Use Cases

This scraper is designed for:
- Building RAG systems with GitHub repository knowledge
- Training models on code review conversations
- Analyzing pull request patterns and workflows
- Creating searchable archives of development discussions