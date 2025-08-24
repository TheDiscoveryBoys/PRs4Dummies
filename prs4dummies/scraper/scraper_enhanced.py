#!/usr/bin/env python3
"""
Enhanced Ansible GitHub Repository Scraper with Better API Methods

This version implements multiple efficient GitHub API approaches:
1. Direct Repository PRs API (recommended)
2. REST API with pagination
3. GraphQL API (most efficient)
"""

import os
import json
import logging
import argparse
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Generator, Union
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from github import Github, Auth
    from github.PullRequest import PullRequest
    from github.Repository import Repository
    import git
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install PyGithub gitpython requests python-dotenv")
    exit(1)


class EnhancedAnsibleGitHubScraper:
    """Enhanced scraper with multiple efficient GitHub API methods."""
    
    def __init__(self, github_token: Optional[str] = None, 
                 output_dir: str = "scraped_data",
                 repo_dir: str = "ansible_repo_enhanced",
                 max_workers: int = 5,
                 batch_size: int = 20,
                 timeout: int = 15,
                 max_retries: int = 3,
                 api_method: str = "direct"):
        """
        Initialize the enhanced scraper.
        
        Args:
            github_token: GitHub personal access token for API access
            output_dir: Directory to save scraped PR data
            repo_dir: Directory to clone the repository
            max_workers: Maximum number of parallel workers
            batch_size: Number of PRs to process in each batch
            timeout: Timeout for API requests in seconds
            max_retries: Maximum number of retries for failed requests
            api_method: API method to use ('direct', 'rest', 'graphql', 'search')
        """
        # Setup logging first
        self._setup_logging()
        
        # Load environment variables from .env file if it exists
        load_dotenv()
        
        # Determine GitHub token to use
        self.github_token = github_token or os.getenv('GITHUB_TOKEN') or os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
        
        self.output_dir = Path(output_dir)
        self.repo_dir = Path(repo_dir)
        self.max_workers = 5
        self.batch_size = 20
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_method = api_method
        
        # Rate limiting
        self.requests_per_hour = 5000 if self.github_token else 60
        self.request_count = 0
        self.last_request_time = time.time()
        self.rate_limit_reset = time.time() + 3600  # 1 hour
        
        # Progress tracking
        self.total_prs = 0
        self.processed_prs = 0
        self.failed_prs = 0
        self.start_time = time.time()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize GitHub client with robust settings
        if self.github_token:
            auth = Auth.Token(self.github_token)
            self.github = Github(auth=auth, timeout=timeout, per_page=100)
            self.logger.info(f"GitHub token loaded successfully - {self.requests_per_hour} requests/hour available")
        else:
            self.github = Github(timeout=timeout, per_page=100)  # Rate limited without token
            self.logger.warning(f"No GitHub token found. API rate limits will apply ({self.requests_per_hour} requests/hour)")
        
        # Repository details
        self.repo_owner = "ansible"
        self.repo_name = "ansible"
        self.repo_url = f"https://github.com/{self.repo_owner}/{self.repo_name}.git"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup robust HTTP session
        self.session = self._setup_robust_session()
        
        # Graceful shutdown flag
        self.shutdown_requested = False
        
        self.logger.info(f"Initialized Enhanced Scraper using '{api_method}' API method")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown_requested = True
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper_enhanced.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_robust_session(self):
        """Setup a robust HTTP session with retry logic."""
        session = requests.Session()
        
        # Retry strategy - handle different urllib3 versions
        try:
            # Try newer urllib3 version (>=2.0.0)
            retry_strategy = Retry(
                total=self.max_retries,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
                backoff_factor=1
            )
        except TypeError:
            # Fallback for older urllib3 version (<2.0.0)
            retry_strategy = Retry(
                total=self.max_retries,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["HEAD", "GET", "OPTIONS"],
                backoff_factor=1
            )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default timeout
        session.timeout = self.timeout
        
        return session
    
    def _rate_limit_check(self):
        """Check and enforce rate limiting with exponential backoff."""
        current_time = time.time()
        
        # Reset counter if hour has passed
        if current_time >= self.rate_limit_reset:
            self.request_count = 0
            self.rate_limit_reset = current_time + 3600
        
        # Check if we're approaching the limit
        if self.request_count >= self.requests_per_hour * 0.8:  # 80% of limit
            wait_time = self.rate_limit_reset - current_time
            if wait_time > 0:
                self.logger.warning(f"Rate limit approaching. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
                self.request_count = 0
                self.rate_limit_reset = time.time() + 3600
        
        # Add delay between requests to be respectful
        if current_time - self.last_request_time < 0.2:  # 200ms between requests
            time.sleep(0.2)
        
        self.last_request_time = current_time
        self.request_count += 1
    
    def _safe_api_call(self, func, *args, **kwargs):
        """Make a safe API call with rate limiting and retry logic."""
        for attempt in range(self.max_retries):
            try:
                if self.shutdown_requested:
                    raise Exception("Shutdown requested")
                
                self._rate_limit_check()
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "403" in str(e):
                    wait_time = (2 ** attempt) * 60  # Exponential backoff: 1min, 2min, 4min
                    self.logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                elif "timeout" in str(e).lower():
                    wait_time = (2 ** attempt) * 10  # Exponential backoff: 10s, 20s, 40s
                    self.logger.warning(f"Timeout, retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
        
        raise Exception(f"Failed after {self.max_retries} retries")
    
    def get_repository(self) -> Repository:
        """Get the GitHub repository object."""
        return self._safe_api_call(self.github.get_repo, f"{self.repo_owner}/{self.repo_name}")
    
    # METHOD 1: Direct Repository PRs API (RECOMMENDED - FASTEST)
    def fetch_prs_direct_api(self, num_prs: int = 50, merged_only: bool = True) -> Generator[PullRequest, None, None]:
        """
        Fetch PRs using direct repository API - FASTEST method.
        
        Benefits:
        - Direct access to repository data
        - Higher rate limits (5000 requests/hour with token)
        - More reliable than search API
        - Can fetch unlimited PRs with pagination
        
        Args:
            num_prs: Number of PRs to fetch
            merged_only: If True, only fetch successfully merged PRs
            
        Yields:
            PullRequest objects one at a time
        """
        try:
            repo = self.get_repository()
            
            if merged_only:
                self.logger.info(f"📦 Fetching {num_prs} merged PRs using DIRECT repository API...")
                
                # Get closed PRs (includes both merged and just closed)
                prs = self._safe_api_call(
                    repo.get_pulls,
                    state="closed",  # Gets both merged and closed
                    sort="updated",
                    direction="asc"
                )
                
                # Stream results and filter for merged PRs only
                count = 0
                processed = 0
                for pr in prs:
                    if count >= num_prs or self.shutdown_requested:
                        break
                    
                    processed += 1
                    
                    # Check if PR was actually merged (not just closed)
                    if pr.merged_at:
                        count += 1
                        yield pr
                        
                        # Log progress every 10 merged PRs
                        if count % 10 == 0:
                            self.logger.info(f"✅ Found {count}/{num_prs} merged PRs (checked {processed} total PRs)")
                
                self.logger.info(f"🎉 Successfully fetched {count} merged PRs using direct API (checked {processed} total)")
                
            else:
                self.logger.info(f"📦 Fetching {num_prs} PRs (all states) using direct API...")
                
                # Use direct PR listing with pagination
                prs = self._safe_api_call(
                    repo.get_pulls, 
                    state="all", 
                    sort="updated", 
                    direction="asc"
                )
                
                count = 0
                for pr in prs:
                    if count >= num_prs or self.shutdown_requested:
                        break
                    
                    count += 1
                    yield pr
                    
                    # Log progress every 10 PRs
                    if count % 10 == 0:
                        self.logger.info(f"✅ Fetched {count}/{num_prs} PRs using direct API...")
                
                self.logger.info(f"🎉 Successfully fetched {count} PRs using direct API")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch PRs using direct API: {e}")
            return
    
    # METHOD 2: REST API with Pagination (GOOD PERFORMANCE)
    def fetch_prs_rest_api(self, num_prs: int = 50, merged_only: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch PRs using GitHub REST API with pagination - GOOD performance.
        
        Benefits:
        - Better control over pagination
        - Can handle very large numbers of PRs
        - Returns raw JSON data (faster processing)
        
        Args:
            num_prs: Number of PRs to fetch
            merged_only: If True, only fetch successfully merged PRs
            
        Yields:
            PR data as dictionaries
        """
        try:
            self.logger.info(f"🌐 Fetching {num_prs} PRs using REST API with pagination...")
            
            # Build URL
            base_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/pulls"
            params = {
                "state": "closed" if merged_only else "all",
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,  # Maximum per page
                "page": 1
            }
            
            headers = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"
            }
            if self.github_token:
                headers['Authorization'] = f'Bearer {self.github_token}'
            
            count = 0
            processed = 0
            page = 1
            
            while count < num_prs and not self.shutdown_requested:
                params["page"] = page
                
                self.logger.debug(f"🔄 Fetching page {page}...")
                response = self.session.get(base_url, params=params, headers=headers)
                response.raise_for_status()
                
                prs = response.json()
                if not prs:  # No more PRs
                    self.logger.info(f"📄 No more PRs found on page {page}")
                    break
                
                # Process PRs from this page
                for pr in prs:
                    if count >= num_prs:
                        break
                    
                    processed += 1
                    
                    # Filter for merged PRs if needed
                    if merged_only:
                        if pr.get("merged_at"):  # Only merged PRs
                            count += 1
                            yield pr
                            
                            if count % 10 == 0:
                                self.logger.info(f"✅ Found {count}/{num_prs} merged PRs via REST API (checked {processed} total)")
                    else:
                        count += 1
                        yield pr
                        
                        if count % 10 == 0:
                            self.logger.info(f"✅ Fetched {count}/{num_prs} PRs via REST API...")
                
                page += 1
                
                # Rate limiting between pages
                time.sleep(0.1)
            
            self.logger.info(f"🎉 Successfully fetched {count} PRs using REST API")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch PRs using REST API: {e}")
            return
    
    # METHOD 3: GraphQL API (MOST EFFICIENT)
    def fetch_prs_graphql_api(self, num_prs: int = 50, merged_only: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch PRs using GitHub GraphQL API - MOST EFFICIENT method.
        
        Benefits:
        - Single request can fetch multiple pages
        - Gets only the data you need
        - Most efficient for large datasets
        - Can fetch 100 PRs per request
        
        Args:
            num_prs: Number of PRs to fetch
            merged_only: If True, only fetch successfully merged PRs
            
        Yields:
            PR data as dictionaries
        """
        try:
            if not self.github_token:
                self.logger.error("❌ GraphQL API requires a GitHub token!")
                return
            
            self.logger.info(f"⚡ Fetching {num_prs} PRs using GraphQL API (most efficient)...")
            
            # GraphQL query - updated to include reviews count
            query = """
            query($owner: String!, $name: String!, $first: Int!, $after: String, $states: [PullRequestState!]) {
                repository(owner: $owner, name: $name) {
                    pullRequests(
                        first: $first,
                        after: $after,
                        states: $states,
                        orderBy: {field: UPDATED_AT, direction: DESC}
                    ) {
                        nodes {
                            number
                            title
                            body
                            state
                            mergedAt
                            createdAt
                            updatedAt
                            additions
                            deletions
                            changedFiles
                            author { login }
                            mergedBy { login }
                            labels(first: 100) { 
                                nodes { name } 
                            }
                            comments {
                                totalCount
                            }
                            reviews {
                                totalCount
                            }
                        }
                        pageInfo {
                            hasNextPage
                            endCursor
                        }
                    }
                }
            }
            """
            
            variables = {
                "owner": self.repo_owner,
                "name": self.repo_name,
                "first": min(100, num_prs),  # GraphQL allows up to 100 per request
                "states": ["MERGED"] if merged_only else ["OPEN", "CLOSED", "MERGED"]
            }
            
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Content-Type": "application/json"
            }
            
            count = 0
            cursor = None
            
            while count < num_prs and not self.shutdown_requested:
                if cursor:
                    variables["after"] = cursor
                
                self.logger.debug(f"🔄 GraphQL request for {variables['first']} PRs...")
                
                response = self.session.post(
                    "https://api.github.com/graphql",
                    json={"query": query, "variables": variables},
                    headers=headers
                )
                response.raise_for_status()
                
                data = response.json()
                
                if "errors" in data:
                    self.logger.error(f"❌ GraphQL errors: {data['errors']}")
                    break
                
                prs = data["data"]["repository"]["pullRequests"]["nodes"]
                
                if not prs:
                    self.logger.info("📄 No more PRs found via GraphQL")
                    break
                
                # Process PRs from this request
                for pr in prs:
                    if count >= num_prs:
                        break
                    
                    count += 1
                    
                    # Get diff for this PR
                    diff_content = ""
                    has_diff = False
                    try:
                        pr_number = pr["number"]
                        diff_response = self.session.get(
                            f"https://github.com/{self.repo_owner}/{self.repo_name}/pull/{pr_number}.diff",
                            timeout=self.timeout
                        )
                        if diff_response.status_code == 200:
                            diff_content = diff_response.text
                            has_diff = True
                    except Exception as e:
                        self.logger.debug(f"Could not fetch diff for PR #{pr.get('number', 'unknown')}: {e}")
                    
                    # Get comments and reviews for this PR
                    comments = self._fetch_pr_comments(pr["number"])
                    reviews = self._fetch_pr_reviews(pr["number"])
                    
                    # Convert GraphQL format to consistent format matching existing structure EXACTLY
                    pr_data = {
                        # Keep number as metadata for identification
                        "pr_number": pr["number"],
                        
                        # Core content that LLMs should understand
                        "title": pr["title"],
                        "body": pr["body"] or "",
                        "author_login": pr["author"]["login"] if pr["author"] else None,
                        "merged_by_login": pr["mergedBy"]["login"] if pr["mergedBy"] else None,
                        
                        # Meaningful labels that describe the PR content
                        "labels": [label["name"] for label in pr["labels"]["nodes"]],
                        
                        # Code change statistics (useful context)
                        "additions": pr["additions"] or 0,
                        "deletions": pr["deletions"] or 0,
                        "changed_files": pr["changedFiles"] or 0,
                        
                        # Comments and reviews (matching original structure exactly)
                        "comments": comments,
                        "reviews": reviews,
                        
                        # Diff content
                        "diff": diff_content
                    }
                    
                    yield pr_data
                    
                    if count % 10 == 0:
                        self.logger.info(f"⚡ Fetched {count}/{num_prs} PRs via GraphQL...")
                
                # Get next page cursor
                page_info = data["data"]["repository"]["pullRequests"]["pageInfo"]
                if not page_info["hasNextPage"]:
                    self.logger.info("📄 Reached end of PRs via GraphQL")
                    break
                cursor = page_info["endCursor"]
                
                # Update variables for next request
                variables["first"] = min(100, num_prs - count)
                
                # Rate limiting
                time.sleep(0.1)
            
            self.logger.info(f"🎉 Successfully fetched {count} PRs using GraphQL API")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch PRs using GraphQL API: {e}")
            return
    
    # METHOD 4: Search API (LEGACY - SLOWER)
    def fetch_prs_search_api(self, num_prs: int = 50, merged_only: bool = True) -> Generator[PullRequest, None, None]:
        """
        Fetch PRs using GitHub Search API - LEGACY method (slower).
        
        This is the original method that was causing hanging issues.
        Kept for comparison purposes.
        
        Args:
            num_prs: Number of PRs to fetch
            merged_only: If True, only fetch successfully merged PRs
            
        Yields:
            PullRequest objects one at a time
        """
        try:
            repo = self.get_repository()
            
            if merged_only:
                self.logger.info(f"🔍 Fetching {num_prs} merged PRs using SEARCH API (legacy method)...")
                
                # Use GitHub Search API with pagination
                search_query = f"repo:{self.repo_owner}/{self.repo_name} is:pr is:merged"
                
                # Search for merged PRs, sorted by updated date (most recent first)
                search_results = self._safe_api_call(
                    self.github.search_issues,
                    query=search_query,
                    sort="updated",
                    order="desc"
                )
                
                # Stream results to avoid memory buildup
                count = 0
                for issue in search_results:
                    if count >= num_prs or self.shutdown_requested:
                        break
                    
                    try:
                        # Convert issue to PR object with timeout
                        pr = self._safe_api_call(repo.get_pull, issue.number)
                        count += 1
                        yield pr
                        
                        # Log progress every 10 PRs
                        if count % 10 == 0:
                            self.logger.info(f"🔍 Found {count}/{num_prs} PRs via search API...")
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to fetch PR #{issue.number}: {e}")
                        continue
                
                self.logger.info(f"🎉 Successfully fetched {count} merged PRs using search API")
        
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch PRs using search API: {e}")
            return
    
    def fetch_pull_requests_enhanced(self, num_prs: int = 50, merged_only: bool = True) -> Generator[Union[PullRequest, Dict[str, Any]], None, None]:
        """
        Fetch PRs using the selected API method.
        
        Args:
            num_prs: Number of PRs to fetch
            merged_only: If True, only fetch successfully merged PRs
            
        Yields:
            PullRequest objects or dictionaries depending on API method
        """
        if self.api_method == "direct":
            yield from self.fetch_prs_direct_api(num_prs, merged_only)
        elif self.api_method == "rest":
            yield from self.fetch_prs_rest_api(num_prs, merged_only)
        elif self.api_method == "graphql":
            yield from self.fetch_prs_graphql_api(num_prs, merged_only)
        elif self.api_method == "search":
            yield from self.fetch_prs_search_api(num_prs, merged_only)
        else:
            self.logger.error(f"❌ Unknown API method: {self.api_method}")
            return
    
    def extract_pr_data_from_object(self, pr: PullRequest) -> Optional[Dict[str, Any]]:
        """Extract PR data from PyGithub PullRequest object to match existing structure."""
        try:
            if self.shutdown_requested:
                return None
            
            # Get PR diff if available
            diff_content = ""
            has_diff = False
            try:
                # Try to get the diff with timeout
                diff_response = self._safe_api_call(
                    lambda: self.session.get(
                        f"https://github.com/{self.repo_owner}/{self.repo_name}/pull/{pr.number}.diff",
                        timeout=self.timeout
                    )
                )
                if diff_response.status_code == 200:
                    diff_content = diff_response.text
                    has_diff = True
            except Exception as e:
                self.logger.debug(f"Could not fetch diff for PR #{pr.number}: {e}")
            
            # Get comments and reviews (full content approach)
            comments = self._fetch_pr_comments(pr.number)
            reviews = self._fetch_pr_reviews(pr.number)
            
            # Basic PR information matching existing structure EXACTLY
            pr_data = {
                # Keep number as metadata for identification
                "pr_number": pr.number,
                
                # Core content that LLMs should understand
                "title": pr.title,
                "body": pr.body or "",
                "author_login": pr.user.login if pr.user else None,
                "merged_by_login": pr.merged_by.login if pr.merged_by else None,
                
                # Meaningful labels that describe the PR content
                "labels": [label.name for label in pr.labels] if pr.labels else [],
                
                # Code change statistics (useful context)
                "additions": pr.additions or 0,
                "deletions": pr.deletions or 0,
                "changed_files": pr.changed_files or 0,
                
                # Comments and reviews (matching original structure exactly)
                "comments": comments,
                "reviews": reviews,
                
                # Diff content
                "diff": diff_content
            }
            
            return pr_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract data for PR #{pr.number}: {e}")
            return None
    
    def extract_pr_data_from_dict(self, pr: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract PR data from dictionary (REST/GraphQL API responses) to match existing structure."""
        try:
            if self.shutdown_requested:
                return None
            
            # Get PR diff if available (for REST API)
            diff_content = ""
            if self.api_method == "rest":
                try:
                    pr_number = pr.get("number")
                    if pr_number:
                        diff_response = self.session.get(
                            f"https://github.com/{self.repo_owner}/{self.repo_name}/pull/{pr_number}.diff",
                            timeout=self.timeout
                        )
                        if diff_response.status_code == 200:
                            diff_content = diff_response.text
                except Exception as e:
                    self.logger.debug(f"Could not fetch diff for PR #{pr.get('number', 'unknown')}: {e}")
            
            # Get comments and reviews for REST API responses
            comments = []
            reviews = []
            if self.api_method == "rest":
                pr_number = pr.get("number")
                if pr_number:
                    comments = self._fetch_pr_comments(pr_number)
                    reviews = self._fetch_pr_reviews(pr_number)
            
            # Basic PR information from dictionary matching existing structure EXACTLY
            pr_data = {
                # Keep number as metadata for identification
                "pr_number": pr.get("number"),
                
                # Core content that LLMs should understand
                "title": pr.get("title", ""),
                "body": pr.get("body", ""),
                "author_login": pr.get("user", {}).get("login") if pr.get("user") else None,
                "merged_by_login": pr.get("merged_by", {}).get("login") if pr.get("merged_by") else None,
                
                # Meaningful labels that describe the PR content
                "labels": [label.get("name") for label in pr.get("labels", [])],
                
                # Code change statistics (useful context)
                "additions": pr.get("additions", 0),
                "deletions": pr.get("deletions", 0),
                "changed_files": pr.get("changed_files", 0),
                
                # Comments and reviews (matching original structure exactly)
                "comments": comments,
                "reviews": reviews,
                
                # Diff content
                "diff": diff_content
            }
            
            return pr_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract data for PR #{pr.get('number', 'unknown')}: {e}")
            return None
    
    def save_pr_data(self, pr_data: Dict[str, Any]) -> bool:
        """Save individual PR data to file."""
        try:
            if not pr_data:
                return False
                
            filename = f"pr_{pr_data['pr_number']}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(pr_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save PR #{pr_data.get('pr_number', 'unknown')}: {e}")
            return False
    
    def process_pr_streaming(self, num_prs: int) -> List[Dict[str, Any]]:
        """Process PRs using streaming approach with the selected API method."""
        scraped_prs = []
        
        try:
            # Get PRs as a stream using the selected method
            pr_stream = self.fetch_pull_requests_enhanced(num_prs)
            
            # Process PRs one by one to avoid memory buildup
            for pr in pr_stream:
                if self.shutdown_requested:
                    self.logger.info("🛑 Shutdown requested, stopping processing...")
                    break
                
                try:
                    # Extract data based on type (object vs dict)
                    if isinstance(pr, dict):
                        pr_data = self.extract_pr_data_from_dict(pr)
                    else:
                        pr_data = self.extract_pr_data_from_object(pr)
                    
                    if pr_data and self.save_pr_data(pr_data):
                        scraped_prs.append(pr_data)
                        self.processed_prs += 1
                        
                        # Log progress
                        if self.processed_prs % 5 == 0:
                            elapsed = time.time() - self.start_time
                            rate = self.processed_prs / elapsed if elapsed > 0 else 0
                            eta = (num_prs - self.processed_prs) / rate if rate > 0 else 0
                            self.logger.info(f"📊 Progress: {self.processed_prs}/{num_prs} ({self.processed_prs/num_prs*100:.1f}%) - Rate: {rate:.1f} PRs/min - ETA: {eta/60:.1f} min")
                    else:
                        self.failed_prs += 1
                        
                except Exception as e:
                    self.logger.error(f"❌ Failed to process PR: {e}")
                    self.failed_prs += 1
                
                # Check if we should continue
                if self.processed_prs + self.failed_prs >= num_prs:
                    break
            
        except Exception as e:
            self.logger.error(f"❌ Error during streaming processing: {e}")
        
        return scraped_prs
    
    def scrape_pull_requests(self, num_prs: int = 50, merged_only: bool = True) -> List[Dict[str, Any]]:
        """Main method to scrape pull requests using enhanced methods."""
        scraped_prs = []
        
        try:
            self.logger.info(f"🚀 Starting enhanced scraping of {num_prs} PRs...")
            self.logger.info(f"⚙️  API Method: {self.api_method}")
            self.logger.info(f"⚙️  Settings: max_workers={self.max_workers}, batch_size={self.batch_size}, timeout={self.timeout}s")
            
            # Process PRs using streaming approach
            scraped_prs = self.process_pr_streaming(num_prs)
            
            # Create summary
            if not self.shutdown_requested:
                self.create_summary_file(scraped_prs)
            
            total_time = time.time() - self.start_time
            self.logger.info(f"🎉 Scraping completed in {total_time:.1f}s!")
            self.logger.info(f"✅ Successfully scraped: {self.processed_prs} PRs")
            self.logger.info(f"❌ Failed: {self.failed_prs} PRs")
            if self.processed_prs > 0:
                self.logger.info(f"⏱️  Average time per PR: {total_time/self.processed_prs:.1f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Error during scraping: {e}")
        finally:
            # Cleanup
            if hasattr(self, 'session'):
                self.session.close()
        
        return scraped_prs
    
    def create_summary_file(self, scraped_prs: List[Dict[str, Any]]) -> bool:
        """Create a summary file of all scraped PRs matching original structure."""
        try:
            summary = {
                "scrape_timestamp": datetime.now().isoformat(),
                "repository": f"{self.repo_owner}/{self.repo_name}",
                "total_prs": len(scraped_prs),
                "prs": [
                    {
                        "pr_number": pr["pr_number"],
                        "title": pr["title"],
                        "author_login": pr["author_login"],
                        "merged_by_login": pr["merged_by_login"],
                        "labels": pr["labels"],
                        "comments_count": len(pr["comments"]),
                        "reviews_count": len(pr["reviews"]),
                        "has_diff": bool(pr["diff"]),
                        "additions": pr["additions"],
                        "deletions": pr["deletions"],
                        "changed_files": pr["changed_files"],
                    }
                    for pr in scraped_prs
                ]
            }
            
            summary_file = self.output_dir / "scrape_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📋 Created summary file: {summary_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create summary file: {e}")
            return False

    def _fetch_pr_comments(self, pr_number: int) -> List[Dict[str, Any]]:
        """
        Fetch ALL comments for a specific PR (full content approach).
        
        Args:
            pr_number: PR number to fetch comments for
            
        Returns:
            List of comment dictionaries matching original structure exactly
        """
        try:
            comments = []
            
            # Get comments via GitHub API
            if self.github_token:
                # Use PyGithub for authenticated requests
                repo = self.get_repository()
                pr = repo.get_pull(pr_number)
                
                # Get ALL regular comments (matching original structure)
                pr_comments = self._safe_api_call(pr.get_issue_comments)
                for comment in pr_comments:
                    comments.append({
                        "author_login": comment.user.login if comment.user else None,
                        "body": comment.body or "",
                    })
                
                # Get ALL review comments (inline code comments) - matching original structure
                review_comments = self._safe_api_call(pr.get_review_comments)
                for comment in review_comments:
                    comments.append({
                        "author_login": comment.user.login if comment.user else None,
                        "body": comment.body or "",
                        "file_path": comment.path,
                        "diff_hunk": comment.diff_hunk,
                        "comment_type": "review_comment"  # To distinguish from general comments
                    })
            else:
                # Fallback to REST API for unauthenticated requests
                # This is a simplified version since we can't get full comment details without auth
                self.logger.warning(f"⚠️  No GitHub token available, skipping comment fetching for PR #{pr_number}")
            
            return comments
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch comments for PR #{pr_number}: {e}")
            return []
    
    def _fetch_pr_reviews(self, pr_number: int) -> List[Dict[str, Any]]:
        """
        Fetch ALL reviews for a specific PR (full content approach).
        
        Args:
            pr_number: PR number to fetch reviews for
            
        Returns:
            List of review dictionaries matching original structure exactly
        """
        try:
            reviews = []
            
            # Get reviews via GitHub API
            if self.github_token:
                # Use PyGithub for authenticated requests
                repo = self.get_repository()
                pr = repo.get_pull(pr_number)
                
                # Get ALL reviews (matching original structure)
                pr_reviews = self._safe_api_call(pr.get_reviews)
                for review in pr_reviews:
                    reviews.append({
                        "author_login": review.user.login if review.user else None,
                        "body": review.body or "",
                        "state": review.state,  # APPROVED, CHANGES_REQUESTED, COMMENTED - useful for understanding
                    })
            else:
                # Fallback to REST API for unauthenticated requests
                # This is a simplified version since we can't get full review details without auth
                self.logger.warning(f"⚠️  No GitHub token available, skipping review fetching for PR #{pr_number}")
            
            return reviews
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch reviews for PR #{pr_number}: {e}")
            return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced Ansible GitHub repository PR scraper with multiple API methods")
    parser.add_argument("--token", 
                       help="GitHub personal access token (overrides .env file)")
    parser.add_argument("--num-prs", type=int, default=50, 
                       help="Number of PRs to scrape (default: 50)")
    parser.add_argument("--output-dir", default="scraped_data",
                       help="Output directory for scraped data (default: scraped_data)")
    parser.add_argument("--repo-dir", default="ansible_repo_enhanced",
                       help="Directory to clone repository (default: ansible_repo_enhanced)")
    parser.add_argument("--merged-only", action="store_true", default=True,
                       help="Only scrape merged PRs (default: True)")
    parser.add_argument("--include-unmerged", action="store_true",
                       help="Include unmerged/closed PRs")
    parser.add_argument("--max-workers", type=int, default=3,
                       help="Maximum number of parallel workers (default: 3)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Number of PRs to process in each batch (default: 10)")
    parser.add_argument("--timeout", type=int, default=15,
                       help="Timeout for API requests in seconds (default: 15)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum number of retries for failed requests (default: 3)")
    parser.add_argument("--api-method", choices=["direct", "rest", "graphql", "search"], default="direct",
                       help="API method to use (default: direct)")
    
    args = parser.parse_args()
    
    # Determine merged_only based on arguments
    merged_only = args.merged_only and not args.include_unmerged
    
    # Initialize enhanced scraper
    scraper = EnhancedAnsibleGitHubScraper(
        github_token=args.token,
        output_dir=args.output_dir,
        repo_dir=args.repo_dir,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        timeout=args.timeout,
        max_retries=args.max_retries,
        api_method=args.api_method
    )
    
    try:
        # Run scraping
        scraped_prs = scraper.scrape_pull_requests(
            num_prs=args.num_prs,
            merged_only=merged_only
        )
        
        print(f"\n🎉 Scraping completed! {len(scraped_prs)} PRs processed.")
        print(f"📁 Data saved to: {args.output_dir}")
        print(f"⚡ API method used: {args.api_method}")
        
        # Performance summary
        if scraped_prs:
            total_time = time.time() - scraper.start_time
            print(f"⏱️  Total time: {total_time:.1f}s")
            print(f"📊 Average time per PR: {total_time/len(scraped_prs):.1f}s")
        
    except KeyboardInterrupt:
        print("\n🛑 Scraping interrupted by user.")
        scraper.shutdown_requested = True
    except Exception as e:
        print(f"\n❌ Scraping failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
