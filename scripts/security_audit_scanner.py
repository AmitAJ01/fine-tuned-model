"""Comprehensive Security Audit Scanner for FineTuneOps."""

import os
import re
from pathlib import Path

REGEX_PATTERNS = {
    "openai_api_key": re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    "hf_token": re.compile(r"hf_[a-zA-Z0-9]{20,}"),
    "github_token": re.compile(r"gh[pousr]_[a-zA-Z0-9]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "password_assignment": re.compile(r"""(?i)(?:password|passwd|secret_key|api_key|access_token)\s*[:=]\s*['\"][^'\"\s]{6,}['\"]"""),
    "database_uri_with_password": re.compile(r"""(?i)[a-z]+://[^:]+:[^@\s]+@[^\s/]+"""),
}

LOCAL_PATH_PATTERN = re.compile(r"""(?i)C:[\\/]Users[\\/][a-zA-Z0-9_\-]+""")

def run_audit():
    root_dir = Path(".")
    findings = {k: [] for k in REGEX_PATTERNS}
    local_path_findings = []
    env_files = []

    skip_dirs = {".git", ".pytest_cache", "__pycache__", "node_modules", ".next"}
    skip_exts = {".safetensors", ".bin", ".pt", ".pth", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".lock"}

    for root, dirs, files in os.walk(root_dir):
        # Prune dirs
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in files:
            path = Path(root) / file
            if file.startswith(".env") and file != ".env.example":
                env_files.append(str(path))

            if path.suffix.lower() in skip_exts:
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                for line_no, line in enumerate(content.splitlines(), start=1):
                    # Check regexes
                    for key, pat in REGEX_PATTERNS.items():
                        if pat.search(line):
                            findings[key].append((str(path), line_no))

                    # Check local machine paths
                    if LOCAL_PATH_PATTERN.search(line):
                        local_path_findings.append((str(path), line_no))
            except Exception as e:
                pass

    print("=== SECURITY SCAN REPORT ===")
    print(f"Active .env files found: {len(env_files)}")
    for ef in env_files:
        print(f"  - {ef}")

    print("\n--- Credential & Token Findings ---")
    total_creds = 0
    for key, matches in findings.items():
        print(f"{key}: {len(matches)} matches")
        total_creds += len(matches)
        for p, lno in matches[:10]:
            print(f"   {p}:{lno}")

    print(f"\n--- Absolute Local Path Findings (C:\\Users\\...): {len(local_path_findings)} occurrences ---")
    # Group by file
    files_with_paths = {}
    for p, lno in local_path_findings:
        files_with_paths[p] = files_with_paths.get(p, 0) + 1

    for p, count in sorted(files_with_paths.items(), key=lambda x: x[1], reverse=True)[:25]:
        print(f"   {count:3d} paths in: {p}")

if __name__ == "__main__":
    run_audit()
