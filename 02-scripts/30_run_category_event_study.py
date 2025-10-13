"""
MASTER RUNNER: CATEGORY EVENT STUDY
====================================

Runs complete category-specific event study pipeline:
1. Extract category event dates from filtered news
2. Run event study for each category × stock combination
3. Aggregate results by sector and category
4. Generate comprehensive visualizations and report

Usage:
    python 30_run_category_event_study.py

Output Structure:
    03-output/category_event_study/
    ├── event_dates/                    # Category event dates
    │   └── [TICKER]/
    │       ├── Earnings_events.csv
    │       ├── Product_Launch_events.csv
    │       └── ... (8 categories)
    ├── results/                        # Event study results
    │   └── [TICKER]/
    │       └── [CATEGORY]/
    │           ├── summary.csv
    │           └── event_study.png
    ├── by_sector/                      # Sector aggregations
    │   └── [SECTOR]/
    │       ├── category_summary.csv
    │       └── category_comparison.png
    ├── comprehensive_category_results.csv
    ├── category_effectiveness_ranking.csv
    ├── sector_category_heatmap.png
    └── CATEGORY_EVENT_STUDY_REPORT.md

Author: Category Event Study System
Date: 2025-10-13
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time


def run_script(script_name: str, description: str):
    """Run a Python script and report status"""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Running: {script_name}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed successfully ({elapsed:.1f}s)")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed ({elapsed:.1f}s)")
        print(f"Error: {str(e)}")
        return False


def main():
    """Run complete category event study pipeline"""
    print("="*80)
    print("CATEGORY EVENT STUDY PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis pipeline will:")
    print("  1. Extract category-specific event dates (27_generate_category_event_dates.py)")
    print("  2. Run event studies for each category (28_category_event_study.py)")
    print("  3. Aggregate and visualize results (29_category_sector_aggregation.py)")
    print("\nEstimated time: ~30-60 minutes for 50 stocks × 8 categories")

    script_dir = Path(__file__).parent
    pipeline_start = time.time()

    # Step 1: Extract category event dates
    step1 = script_dir / "27_generate_category_event_dates.py"
    if not run_script(str(step1), "Extract Category Event Dates"):
        print("\n❌ Pipeline failed at Step 1")
        return False

    # Step 2: Run category event studies
    step2 = script_dir / "28_category_event_study.py"
    if not run_script(str(step2), "Run Category Event Studies"):
        print("\n❌ Pipeline failed at Step 2")
        return False

    # Step 3: Aggregate and visualize
    step3 = script_dir / "29_category_sector_aggregation.py"
    if not run_script(str(step3), "Aggregate and Visualize Results"):
        print("\n❌ Pipeline failed at Step 3")
        return False

    # Success!
    total_time = time.time() - pipeline_start

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\n✅ All steps completed successfully")
    print(f"📊 Total time: {total_time/60:.1f} minutes")
    print(f"🕐 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    output_dir = script_dir.parent / "03-output" / "category_event_study"
    print(f"\n📁 Results location: {output_dir}")
    print(f"\n📄 Key outputs:")
    print(f"   - Comprehensive report: {output_dir / 'CATEGORY_EVENT_STUDY_REPORT.md'}")
    print(f"   - Category rankings: {output_dir / 'category_effectiveness_ranking.csv'}")
    print(f"   - Sector heatmap: {output_dir / 'sector_category_heatmap.png'}")
    print(f"   - All results: {output_dir / 'comprehensive_category_results.csv'}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)