#!/usr/bin/env python3
"""
Ansible GitHub Repository Scraper

This script scrapes the Ansible GitHub repository to extract pull request data
for use in a vector store RAG system. It clones the repository locally and
fetches detailed information about the most recent pull requests.

Dependencies:
- PyGithub
- gitpython
- requests

Usage:
    python scraper.py --token YOUR_GITHUB_TOKEN --num-prs 50
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import subprocess
import shutil

try:
    from github import Github, Auth
    from github.PullRequest import PullRequest
    from github.Repository import Repository
    import git
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install PyGithub gitpython requests python-dotenv")
    exit(1)


class AnsibleGitHubScraper:
    """Scraper for Ansible GitHub repository pull requests."""
    
    def __init__(self, github_token: Optional[str] = None, 
                 output_dir: str = "scraped_data",
                 repo_dir: str = "ansible_repo"):
        """
        Initialize the scraper.
        
        Args:
            github_token: GitHub personal access token for API access
            output_dir: Directory to save scraped PR data
            repo_dir: Directory to clone the repository
        """
        # Setup logging first
        self._setup_logging()
        
        # Load environment variables from .env file if it exists
        load_dotenv()
        
        # Determine GitHub token to use
        self.github_token = github_token or os.getenv('GITHUB_TOKEN') or os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
        
        self.output_dir = Path(output_dir)
        self.repo_dir = Path(repo_dir)
        
        # Initialize GitHub client
        if self.github_token:
            auth = Auth.Token(self.github_token)
            self.github = Github(auth=auth)
            self.logger.info("GitHub token loaded successfully - higher rate limits available")
        else:
            self.github = Github()  # Rate limited without token
            self.logger.warning("No GitHub token found. API rate limits will apply (60 requests/hour)")
            self.logger.info("To use a token, either:")
            self.logger.info("  1. Add GITHUB_TOKEN=your_token to a .env file")
            self.logger.info("  2. Use --token argument")
            self.logger.info("  3. Set GITHUB_TOKEN environment variable")
        
        # Repository details
        self.repo_owner = "ansible"
        self.repo_name = "ansible"
        self.repo_url = f"https://github.com/{self.repo_owner}/{self.repo_name}.git"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def clone_repository(self, force_refresh: bool = False) -> bool:
        """
        Clone the Ansible repository locally.
        
        Args:
            force_refresh: Whether to delete and re-clone if repo exists
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.repo_dir.exists():
                if force_refresh:
                    self.logger.info(f"Removing existing repository at {self.repo_dir}")
                    shutil.rmtree(self.repo_dir)
                else:
                    self.logger.info(f"Repository already exists at {self.repo_dir}")
                    # Try to update existing repo
                    try:
                        repo = git.Repo(self.repo_dir)
                        origin = repo.remotes.origin
                        origin.pull()
                        self.logger.info("Successfully updated existing repository")
                        return True
                    except Exception as e:
                        self.logger.warning(f"Failed to update existing repo: {e}")
                        self.logger.info("Proceeding with fresh clone...")
                        shutil.rmtree(self.repo_dir)
            
            self.logger.info(f"Cloning repository from {self.repo_url}")
            git.Repo.clone_from(self.repo_url, self.repo_dir, depth=1)
            self.logger.info(f"Successfully cloned repository to {self.repo_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clone repository: {e}")
            return False
    
    def get_repository(self) -> Repository:
        """Get the GitHub repository object."""
        return self.github.get_repo(f"{self.repo_owner}/{self.repo_name}")
    
    def fetch_pull_requests(self, num_prs: int = 50, merged_only: bool = True) -> List[PullRequest]:
        """
        Fetch the most recent pull requests.
        
        Args:
            num_prs: Number of PRs to fetch
            merged_only: If True, only fetch successfully merged PRs using GitHub Search API
            
        Returns:
            List of PullRequest objects
        """
        try:
            repo = self.get_repository()
            
            if merged_only:
                self.logger.info(f"Fetching {num_prs} merged pull requests using GitHub Search API...")
                
                # Use GitHub Search API to filter for merged PRs directly
                # Search query: repo:owner/name is:pr is:merged
                search_query = f"repo:{self.repo_owner}/{self.repo_name} is:pr is:merged"
                
                # Search for merged PRs, sorted by updated date (most recent first)
                search_results = self.github.search_issues(
                    query=search_query,
                    sort="updated",
                    order="desc"
                )
                
                # Convert search results to PR objects and take only the requested number
                prs = []
                for i, issue in enumerate(search_results):
                    if i >= num_prs:
                        break
                    # Convert issue to PR object
                    pr = repo.get_pull(issue.number)
                    prs.append(pr)
                
                self.logger.info(f"Successfully fetched {len(prs)} merged pull requests")
                
            else:
                self.logger.info(f"Fetching {num_prs} pull requests (all states)...")
                prs = list(repo.get_pulls(state="all", sort="updated", direction="desc")[:num_prs])
                self.logger.info(f"Successfully fetched {len(prs)} pull requests")
            
            return prs
            
        except Exception as e:
            self.logger.error(f"Failed to fetch pull requests: {e}")
            return []
    
    def get_pr_diff(self, pr: PullRequest) -> Optional[str]:
        """
        Get the diff content for a pull request.
        
        Args:
            pr: PullRequest object
            
        Returns:
            Diff content as string or None if failed
        """
        try:
            # Method 1: Use GitHub's .diff URL (recommended)
            diff_url = f"https://github.com/{self.repo_owner}/{self.repo_name}/pull/{pr.number}.diff"
            
            headers = {}
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'
            
            self.logger.debug(f"Fetching diff from: {diff_url}")
            response = requests.get(diff_url, headers=headers)
            response.raise_for_status()
            
            diff_text = response.text
            
            # Validate that we got actual diff content, not JSON
            if diff_text.startswith('{') and '"url"' in diff_text:
                self.logger.error(f"Got JSON response instead of diff for PR #{pr.number}")
                # Try fallback method
                return self._get_pr_diff_fallback(pr)
            
            # Basic validation that this looks like a diff
            if diff_text.startswith('diff --git') or 'diff --git' in diff_text:
                return diff_text
            elif not diff_text.strip():
                self.logger.warning(f"Empty diff for PR #{pr.number}")
                return ""
            else:
                self.logger.warning(f"Unexpected diff format for PR #{pr.number}: {diff_text[:100]}...")
                return diff_text  # Return anyway, might still be useful
                
        except Exception as e:
            self.logger.error(f"Failed to get diff for PR #{pr.number}: {e}")
            # Try fallback method
            return self._get_pr_diff_fallback(pr)
    
    def _get_pr_diff_fallback(self, pr: PullRequest) -> Optional[str]:
        """
        Fallback method to get PR diff using GitHub API.
        
        Args:
            pr: PullRequest object
            
        Returns:
            Diff content as string or None if failed
        """
        try:
            self.logger.info(f"Trying fallback diff method for PR #{pr.number}")
            
            # Use the API endpoint with explicit diff accept header
            diff_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr.number}"
            
            headers = {
                'Accept': 'application/vnd.github.v3.diff'
            }
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'
            
            response = requests.get(diff_url, headers=headers)
            response.raise_for_status()
            
            # This should return plain text diff when using the right Accept header
            diff_text = response.text
            
            if diff_text.startswith('{'):
                self.logger.error(f"Fallback also returned JSON for PR #{pr.number}")
                return None
                
            return diff_text
            
        except Exception as e:
            self.logger.error(f"Fallback diff method failed for PR #{pr.number}: {e}")
            return None
    
    def extract_pr_data(self, pr: PullRequest) -> Dict[str, Any]:
        """
        Extract comprehensive data from a pull request.
        
        Args:
            pr: PullRequest object
            
        Returns:
            Dictionary containing all PR data
        """
        self.logger.info(f"Extracting data for PR #{pr.number}: {pr.title}")
        
        # Content-focused PR information (excluding low-signal metadata)
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
            "additions": pr.additions,
            "deletions": pr.deletions,
            "changed_files": pr.changed_files,
        }
        
        # Extract comments
        try:
            comments = []
            for comment in pr.get_issue_comments():
                comments.append({
                    "author_login": comment.user.login if comment.user else None,
                    "body": comment.body or "",
                })
            
            # Also get review comments (inline code comments)
            for comment in pr.get_review_comments():
                comments.append({
                    "author_login": comment.user.login if comment.user else None,
                    "body": comment.body or "",
                    "file_path": comment.path,
                    "diff_hunk": comment.diff_hunk,
                    "comment_type": "review_comment"  # To distinguish from general comments
                })
            
            pr_data["comments"] = comments
            self.logger.info(f"Extracted {len(comments)} comments for PR #{pr.number}")
            
        except Exception as e:
            self.logger.error(f"Failed to extract comments for PR #{pr.number}: {e}")
            pr_data["comments"] = []
        
        # Extract reviews
        try:
            reviews = []
            for review in pr.get_reviews():
                reviews.append({
                    "author_login": review.user.login if review.user else None,
                    "body": review.body or "",
                    "state": review.state,  # APPROVED, CHANGES_REQUESTED, COMMENTED - useful for understanding
                })
            
            pr_data["reviews"] = reviews
            self.logger.info(f"Extracted {len(reviews)} reviews for PR #{pr.number}")
            
        except Exception as e:
            self.logger.error(f"Failed to extract reviews for PR #{pr.number}: {e}")
            pr_data["reviews"] = []
        
        # Get diff
        diff_content = self.get_pr_diff(pr)
        pr_data["diff"] = diff_content
        
        if diff_content:
            self.logger.info(f"Extracted diff for PR #{pr.number} ({len(diff_content)} characters)")
        else:
            self.logger.warning(f"No diff content for PR #{pr.number}")
        
        return pr_data
    
    def save_pr_data(self, pr_data: Dict[str, Any]) -> bool:
        """
        Save PR data to a structured JSON file.
        
        Args:
            pr_data: Dictionary containing PR data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            filename = f"pr_{pr_data['pr_number']}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(pr_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved PR #{pr_data['pr_number']} data to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save PR data: {e}")
            return False
    
    def create_summary_file(self, scraped_prs: List[Dict[str, Any]]) -> bool:
        """
        Create a summary file with metadata about all scraped PRs.
        
        Args:
            scraped_prs: List of PR data dictionaries
            
        Returns:
            bool: True if successful, False otherwise
        """
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
            
            self.logger.info(f"Created summary file: {summary_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create summary file: {e}")
            return False
    
    def scrape_pull_requests(self, num_prs: int = 50, clone_repo: bool = True, 
                           force_refresh: bool = False, merged_only: bool = True) -> List[Dict[str, Any]]:
        """
        Main method to scrape pull requests.
        
        Args:
            num_prs: Number of PRs to scrape
            clone_repo: Whether to clone the repository
            force_refresh: Whether to force refresh the local repo
            merged_only: Whether to only scrape successfully merged PRs
            
        Returns:
            List of scraped PR data
        """
        scraped_prs = []
        
        try:
            # Clone repository if requested
            if clone_repo:
                if not self.clone_repository(force_refresh):
                    self.logger.error("Failed to clone repository. Continuing without local repo...")
            
            # Fetch pull requests
            prs = self.fetch_pull_requests(num_prs, merged_only=merged_only)
            if not prs:
                self.logger.error("No pull requests fetched. Exiting.")
                return []
            
            # Process each PR
            for i, pr in enumerate(prs, 1):
                self.logger.info(f"Processing PR {i}/{len(prs)}: #{pr.number}")
                
                try:
                    pr_data = self.extract_pr_data(pr)
                    if self.save_pr_data(pr_data):
                        scraped_prs.append(pr_data)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process PR #{pr.number}: {e}")
                    continue
            
            # Create summary
            self.create_summary_file(scraped_prs)
            
            self.logger.info(f"Successfully scraped {len(scraped_prs)} pull requests")
            
        except Exception as e:
            self.logger.error(f"Error during scraping: {e}")
        
        return scraped_prs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Scrape Ansible GitHub repository PRs")
    parser.add_argument("--token", 
                       help="GitHub personal access token (overrides .env file)")
    parser.add_argument("--num-prs", type=int, default=50, 
                       help="Number of PRs to scrape (default: 50)")
    parser.add_argument("--output-dir", default="scraped_data",
                       help="Output directory for scraped data (default: scraped_data)")
    parser.add_argument("--repo-dir", default="ansible_repo",
                       help="Directory to clone repository (default: ansible_repo)")
    parser.add_argument("--no-clone", action="store_true",
                       help="Skip cloning the repository")
    parser.add_argument("--force-refresh", action="store_true",
                       help="Force refresh of local repository")
    parser.add_argument("--include-unmerged", action="store_true",
                       help="Include unmerged/closed PRs (default: only merged PRs)")
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = AnsibleGitHubScraper(
        github_token=args.token,
        output_dir=args.output_dir,
        repo_dir=args.repo_dir
    )
    
    # Run scraping
    scraped_prs = scraper.scrape_pull_requests(
        num_prs=args.num_prs,
        clone_repo=not args.no_clone,
        force_refresh=args.force_refresh,
        merged_only=not args.include_unmerged
    )
    
    print(f"\nScraping completed! {len(scraped_prs)} PRs processed.")
    print(f"Data saved to: {args.output_dir}")
    print(f"Local repo: {args.repo_dir}")


if __name__ == "__main__":
    main()
