"""
DermoGraph-XAI | RUN_ALL.py
Master runner — executes all 4 analysis scripts sequentially.
Run this single file to generate ALL outputs in dermograph_output/
"""

import subprocess, sys, os, time

SCRIPTS = [
    ("01_class_distribution.py",  "Class Distribution & Dataset Overview"),
    ("02_age_analysis.py",        "Age Analysis & Risk Distribution"),
    ("03_localization_gender.py", "Body Localization & Gender Analysis"),
    ("04_abcde_analysis.py",      "ABCDE Clinical Risk Score Engine"),
]

print("=" * 65)
print("  DermoGraph-XAI · HAM10000 Analytics Suite")
print("  Review 1 — Clinical Intelligence Dashboard")
print("=" * 65)
print()

total_start = time.time()
passed, failed = [], []

for fname, desc in SCRIPTS:
    print(f"▶  Running: {fname}")
    print(f"   → {desc}")
    start = time.time()
    result = subprocess.run([sys.executable, fname], capture_output=False)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"   ✅ Done in {elapsed:.1f}s\n")
        passed.append(fname)
    else:
        print(f"   ❌ FAILED after {elapsed:.1f}s\n")
        failed.append(fname)

total = time.time() - total_start
print("=" * 65)
print(f"  Completed in {total:.1f}s")
print(f"  ✅ Passed: {len(passed)}/{len(SCRIPTS)}")
if failed:
    print(f"  ❌ Failed: {', '.join(failed)}")
print()
print("  📁 Output files in: ./dermograph_output/")
print()

out = "dermograph_output"
if os.path.exists(out):
    files = sorted(os.listdir(out))
    for f in files:
        path = os.path.join(out, f)
        size = os.path.getsize(path)
        print(f"     {f:45s}  {size/1024:.0f} KB")
print("=" * 65)