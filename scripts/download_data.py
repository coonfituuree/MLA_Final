#!/usr/bin/env python3
"""Download the actual Kaggle archive without credentials; verify against this project's CSV."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://www.kaggle.com/api/v1/datasets/download/shivamb/real-or-fake-fake-jobposting-prediction'

def main():
    target = ROOT / 'data/raw/fake_job_postings.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try:
            with urllib.request.urlopen(URL, timeout=180) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))
            data = archive.read('fake_job_postings.csv')
            target.write_bytes(data)
        except Exception as exc:
            raise SystemExit(f'Download failed: {exc}. Download from Kaggle manually and place fake_job_postings.csv at {target}')
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    provenance = ROOT / 'data/provenance.json'
    if provenance.exists():
        expected = json.loads(provenance.read_text())['sha256']
        if digest != expected:
            raise SystemExit(f'Checksum mismatch: expected {expected}, got {digest}. Inspect source before training.')
    print(f'{target}\nSHA256 {digest}')
if __name__ == '__main__':
    main()
