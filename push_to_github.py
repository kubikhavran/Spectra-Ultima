"""
Script to create a GitHub repository and push the local code base to it.

This helper uses the GitHub REST API to create a new repository for the
authenticated user and then performs a git push over HTTPS.  A personal
access token with the `repo` scope is required.  The token must be
provided via the `GITHUB_TOKEN` environment variable or as a command
line argument.  The repository name is derived from the current
directory unless overridden via the `--name` option.

Usage::

    python push_to_github.py --name my-spectral-app --description "Spectral analysis app"

The script will prompt for missing values.  **Important:** the token
will be included in the remote URL used for pushing; this script is
intended for automated environments and should be reviewed before use
with sensitive credentials.

GitHub API reference: creating a repository for the authenticated user
requires a POST to `/user/repos` with at least a `name` parameter and
appropriate token scopes【504460264174486†L5940-L5990】.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests


def create_repo(token: str, name: str, description: str = "", private: bool = False) -> dict:
    """Create a new GitHub repository using the REST API.

    Parameters
    ----------
    token : str
        Personal access token with `repo` scope.
    name : str
        Name of the repository to create.
    description : str, optional
        Repository description.
    private : bool, optional
        Whether the repository should be private (default: False).

    Returns
    -------
    dict
        JSON response from GitHub describing the repository.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "name": name,
        "description": description,
        "private": private,
        "auto_init": False,
    }
    response = requests.post("https://api.github.com/user/repos", headers=headers, data=json.dumps(payload))
    if response.status_code >= 400:
        raise RuntimeError(f"Failed to create repository: {response.status_code} {response.text}")
    return response.json()


def git(*args: str) -> None:
    """Run a git command and raise on failure."""
    result = subprocess.run(["git", *args], check=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a GitHub repository and push local code.")
    parser.add_argument("--name", help="Name of the new repository (default: current directory name)")
    parser.add_argument("--description", default="", help="Repository description")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    parser.add_argument("--token", help="GitHub personal access token (alternative to GITHUB_TOKEN env var)")
    args = parser.parse_args()
    repo_name = args.name or Path.cwd().name
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: a GitHub token must be provided via --token or GITHUB_TOKEN environment variable.")
        sys.exit(1)
    print(f"Creating repository {repo_name}...")
    repo_data = create_repo(token, repo_name, description=args.description, private=args.private)
    clone_url = repo_data.get("clone_url")
    if not clone_url:
        print("Error: unable to retrieve clone URL from response")
        sys.exit(1)
    # Configure git if needed
    if not (Path.cwd() / ".git").exists():
        git("init")
    # Add all files and commit
    git("add", ".")
    try:
        git("commit", "-m", "Initial commit")
    except subprocess.CalledProcessError:
        # commit may fail if there is nothing to commit
        pass
    # Set branch to main
    try:
        git("branch", "-M", "main")
    except subprocess.CalledProcessError:
        pass
    # Construct authenticated remote URL
    authenticated_url = clone_url.replace("https://", f"https://{token}@")
    # Add remote if not present
    try:
        git("remote", "add", "origin", authenticated_url)
    except subprocess.CalledProcessError:
        git("remote", "set-url", "origin", authenticated_url)
    # Push to remote
    git("push", "-u", "origin", "main")
    print(f"Repository created and pushed: {repo_data.get('html_url')}")


if __name__ == "__main__":
    main()