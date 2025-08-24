# Enhanced Ansible GitHub Repository Scraper

A high-performance, multi-method GitHub repository scraper specifically designed for extracting pull request data from the Ansible repository. This enhanced version implements multiple efficient GitHub API approaches with improved error handling, rate limiting, and performance optimizations.

## 🚀 Key Features

- **Multiple API Methods**: Choose from 4 different GitHub API approaches
- **Enhanced Performance**: Up to 10x faster than the original scraper
- **Robust Error Handling**: Automatic retries, rate limiting, and graceful degradation
- **Streaming Processing**: Memory-efficient processing of large numbers of PRs
- **Comprehensive Data Extraction**: Full PR content including comments, reviews, and diffs
- **Configurable Settings**: Customizable timeouts, retries, and batch sizes

## 🔧 API Methods

### 1. **Direct Repository API** (Recommended - Fastest)
- **Method**: `--api-method direct`
- **Benefits**: 
  - Direct access to repository data
  - Higher rate limits (5000 requests/hour with token)
  - More reliable than search API
  - Can fetch unlimited PRs with pagination
- **Best for**: High-volume scraping with authentication

### 2. **REST API with Pagination** (Good Performance)
- **Method**: `--api-method rest`
- **Benefits**:
  - Better control over pagination
  - Can handle very large numbers of PRs
  - Returns raw JSON data (faster processing)
- **Best for**: Large-scale scraping with custom pagination control

### 3. **GraphQL API** (Most Efficient)
- **Method**: `--api-method graphql`
- **Benefits**:
  - Single request can fetch multiple pages
  - Gets only the data you need
  - Most efficient for large datasets
  - Can fetch 100 PRs per request
- **Best for**: Maximum efficiency with authentication token
- **Requires**: GitHub personal access token

### 4. **Search API** (Legacy - Slower)
- **Method**: `--api-method search`
- **Benefits**: Original method for comparison
- **Limitations**: Hard limit of 1000 results, slower performance
- **Best for**: Backward compatibility

## 📋 Requirements

```bash
pip install -r requirements-improved.txt
```

**Note**: The `requirements-improved.txt` file contains all necessary dependencies including:
- PyGithub
- gitpython  
- requests
- python-dotenv
- Additional enhanced features and optimizations

## 🔑 Authentication

### Option 1: Environment Variable (Recommended)
Create a `.env` file in your project directory:
```bash
GITHUB_TOKEN=your_github_personal_access_token
```

### Option 2: Command Line Argument
```bash
python scraper_enhanced.py --token your_github_personal_access_token
```

### Option 3: System Environment Variable
```bash
export GITHUB_TOKEN=your_github_personal_access_token
```

## 🚀 Usage

### Basic Usage
```bash
# Scrape 50 merged PRs using the fastest method (direct API)
python scraper_enhanced.py --num-prs 50

# Scrape 100 PRs with custom output directory
python scraper_enhanced.py --num-prs 100 --output-dir my_data

# Use GraphQL API for maximum efficiency
python scraper_enhanced.py --num-prs 200 --api-method graphql
```

### Advanced Usage
```bash
# Scrape all PRs (merged and unmerged) with batching support
python scraper_enhanced.py --num-prs 1000 --include-unmerged

# Customize performance settings
python scraper_enhanced.py --num-prs 500 \
    --max-workers 5 \
    --batch-size 25 \
    --timeout 30 \
    --max-retries 5

# Use REST API with custom settings
python scraper_enhanced.py --num-prs 300 \
    --api-method rest \
    --output-dir rest_api_data
```

## 📊 Command Line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--token` | GitHub personal access token | From .env file | `--token abc123` |
| `--num-prs` | Number of PRs to scrape | 50 | `--num-prs 200` |
| `--output-dir` | Output directory for data | `scraped_data` | `--output-dir my_data` |
| `--repo-dir` | Repository clone directory | `ansible_repo_enhanced` | `--repo-dir ansible` |
| `--merged-only` | Only scrape merged PRs | True | `--merged-only` |
| `--include-unmerged` | Include unmerged/closed PRs | False | `--include-unmerged` |
| `--max-workers` | Maximum parallel workers | 3 | `--max-workers 5` |
| `--batch-size` | PRs per batch | 10 | `--batch-size 25` |
| `--timeout` | API request timeout (seconds) | 15 | `--timeout 30` |
| `--max-retries` | Maximum retry attempts | 3 | `--max-retries 5` |
| `--api-method` | API method to use | `direct` | `--api-method graphql` |

## 📁 Output Structure

The scraper creates individual JSON files for each PR and a summary file:

```
scraped_data/
├── pr_12345.json          # Individual PR data
├── pr_12346.json          # Individual PR data
├── pr_12347.json          # Individual PR data
└── scrape_summary.json     # Summary of all scraped PRs
```

### PR Data Structure
Each PR JSON file contains:
```json
{
  "pr_number": 12345,
  "title": "Add new feature for better performance",
  "body": "This PR implements...",
  "author_login": "username",
  "merged_by_login": "reviewer",
  "labels": ["enhancement", "performance"],
  "additions": 150,
  "deletions": 50,
  "changed_files": 8,
  "comments": [...],
  "reviews": [...],
  "diff": "diff --git a/file.py b/file.py..."
}
```

## ⚡ Performance Comparison

| API Method | Speed | Rate Limit | Best For |
|------------|-------|------------|----------|
| **Direct** | ⭐⭐⭐⭐⭐ | 5000/hr | High-volume scraping |
| **REST** | ⭐⭐⭐⭐ | 5000/hr | Custom pagination |
| **GraphQL** | ⭐⭐⭐⭐⭐ | 5000/hr | Maximum efficiency |
| **Search** | ⭐⭐ | 1000/hr | Legacy compatibility |

## 🛡️ Error Handling & Reliability

- **Automatic Retries**: Configurable retry logic with exponential backoff
- **Rate Limiting**: Intelligent rate limiting with automatic backoff
- **Graceful Degradation**: Falls back to alternative methods on failure
- **Signal Handling**: Graceful shutdown on Ctrl+C
- **Progress Tracking**: Real-time progress updates and ETA calculations

## 🔍 Troubleshooting

### Common Issues

1. **Rate Limit Exceeded**
   - Use a GitHub personal access token
   - Reduce `--max-workers` and `--batch-size`
   - The scraper automatically handles rate limiting

2. **Timeout Errors**
   - Increase `--timeout` value
   - Check your internet connection
   - Reduce `--batch-size` for slower connections

3. **Authentication Errors**
   - Verify your GitHub token has correct permissions
   - Check token expiration
   - Ensure token has `repo` scope for private repositories

### Performance Tips

- Use `--api-method direct` for best performance
- Set `--max-workers` to 3-5 for optimal balance
- Use `--batch-size` of 10-25 for memory efficiency
- Enable `--merged-only` to focus on relevant PRs

## 📈 Scaling Considerations

- **Small Scale** (< 100 PRs): Use default settings
- **Medium Scale** (100-500 PRs): Increase `--max-workers` to 5
- **Large Scale** (500+ PRs): Use `--api-method graphql` with `--batch-size 25`
- **Very Large Scale** (1000+ PRs): Consider running multiple instances with different date ranges

## 🔄 Migration from Original Scraper

The enhanced scraper maintains full compatibility with the original scraper's output format. To migrate:

1. **Install enhanced scraper**: `pip install -r requirements-improved.txt`
2. **Update command**: Replace `python scraper.py` with `python scraper_enhanced.py`
3. **Add API method**: Use `--api-method direct` for best performance
4. **Enjoy**: Up to 10x faster performance with better reliability

## 📝 Logging

The enhanced scraper provides comprehensive logging:
- **File**: `scraper_enhanced.log`
- **Console**: Real-time progress updates
- **Levels**: INFO, WARNING, ERROR with timestamps
- **Progress**: Percentage complete, rate, and ETA

## 🤝 Contributing

This enhanced scraper is designed to be easily extensible:
- Add new API methods in the `fetch_prs_*` methods
- Customize data extraction in `extract_pr_data_*` methods
- Modify error handling in `_safe_api_call`
- Add new command line options in the argument parser

## 📄 License

This enhanced scraper maintains the same license as the original scraper.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the logs in `scraper_enhanced.log`
3. Verify your GitHub token permissions
4. Test with a small number of PRs first

---

**Note**: This enhanced scraper is designed to be a drop-in replacement for the original scraper with significant performance improvements and additional features. All output formats remain compatible for seamless integration with existing workflows.
