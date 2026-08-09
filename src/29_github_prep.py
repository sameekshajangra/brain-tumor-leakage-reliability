import os
import re
from pathlib import Path

BASE_DIR = Path("/Users/sameekshasharma/Desktop/brain-tumor/brain_tumor_patient_overlap")

def find_large_files(threshold_mb=50):
    large_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        if '.git' in root or 'venv' in root or '.venv' in root:
            continue
        for file in files:
            path = Path(root) / file
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > threshold_mb:
                    large_files.append((str(path.relative_to(BASE_DIR)), size_mb))
            except:
                pass
    return large_files

def scan_secrets():
    secrets_found = []
    patterns = [
        re.compile(r'(?i)api[_-]?key[\s:=]+[\'"]?([a-zA-Z0-9_-]{10,})[\'"]?'),
        re.compile(r'(?i)password[\s:=]+[\'"]?([^"\'\s]+)[\'"]?'),
        re.compile(r'(?i)secret[\s:=]+[\'"]?([a-zA-Z0-9_-]{10,})[\'"]?'),
        re.compile(r'(?i)token[\s:=]+[\'"]?([a-zA-Z0-9_-]{10,})[\'"]?')
    ]
    for root, dirs, files in os.walk(BASE_DIR):
        if '.git' in root or 'venv' in root or '.venv' in root or '__pycache__' in root or 'checkpoints' in root or 'data' in root:
            continue
        for file in files:
            if not file.endswith(('.py', '.md', '.txt', '.csv', '.sh', '.json')):
                continue
            path = Path(root) / file
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        for p in patterns:
                            if p.search(line):
                                secrets_found.append(f"Suspicious pattern in {path.relative_to(BASE_DIR)} at line {i+1}")
            except:
                pass
    return secrets_found

def check_git_status():
    import subprocess
    if not (BASE_DIR / '.git').exists():
        subprocess.run(['git', 'init'], cwd=BASE_DIR)
    
    result = subprocess.run(['git', 'status'], cwd=BASE_DIR, capture_output=True, text=True)
    return result.stdout

def main():
    print("Scanning for large files > 50MB...")
    large_files = find_large_files()
    for f, size in large_files:
        print(f"  {f} ({size:.2f} MB)")
        
    print("Scanning for secrets...")
    secrets = scan_secrets()
    for s in secrets:
        print(f"  {s}")
        
    print("Checking git status...")
    status = check_git_status()
    print(status)

if __name__ == "__main__":
    main()
