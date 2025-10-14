"""
MASTER UPDATED ANALYSIS PIPELINE
==================================

Runs the complete updated analysis pipeline:
1. Download 2025 news data (Jan 1 - Jul 1, 2025)
2. Apply 3:30 PM ET cutoff and merge with existing data
3. Filter and categorize news with balanced filter
4. Generate category-specific event dates
5. Download/verify Fama-French data for 2021-2025
6. Run event study for all 50 stocks
7. Check news day balance (% news vs non-news)
8. Generate aggregate summaries and visualizations
9. Generate sector-level visualizations

Author: Master Pipeline Orchestrator
Date: 2025-10-13
"""

import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent

# Pipeline steps
PIPELINE_STEPS = [
    {
        'name': 'Download 2025 News Data',
        'script': '32_download_2025_news_update.py',
        'description': 'Downloads news from Jan 1 - Jul 1, 2025 and applies 3:30 PM ET cutoff',
        'estimated_time': '30-60 minutes',
        'required': True
    },
    {
        'name': 'Filter and Categorize News',
        'script': '33_updated_news_filter_with_timing.py',
        'description': 'Applies balanced filter and categorizes events for 2021-2025 window',
        'estimated_time': '15-20 minutes',
        'required': True
    },
    {
        'name': 'Generate Category Event Dates',
        'script': '35_generate_category_events_2021_2025.py',
        'description': 'Creates event date files for each category × stock combination',
        'estimated_time': '5-10 minutes',
        'required': True
    },
    {
        'name': 'Verify Fama-French Data',
        'script': '36_verify_fama_french_2021_2025.py',
        'description': 'Checks Fama-French data coverage for 2021-2025',
        'estimated_time': '1-2 minutes',
        'required': True
    },
    {
        'name': 'Run Event Study Analysis',
        'script': '37_event_study_2021_2025_all_stocks.py',
        'description': 'Runs comprehensive event study for all 50 stocks',
        'estimated_time': '60-90 minutes',
        'required': True
    },
    {
        'name': 'Check News Day Balance',
        'script': '38_check_news_day_balance.py',
        'description': 'Verifies % of news days vs non-news days for each stock',
        'estimated_time': '2-3 minutes',
        'required': True
    },
    {
        'name': 'Generate Aggregate Summaries',
        'script': '39_aggregate_summaries_2021_2025.py',
        'description': 'Creates cross-stock aggregate visualizations and summaries',
        'estimated_time': '5-10 minutes',
        'required': True
    },
    {
        'name': 'Generate Sector Visualizations',
        'script': '40_sector_visualizations_2021_2025.py',
        'description': 'Creates sector-level analysis charts',
        'estimated_time': '5-10 minutes',
        'required': True
    }
]

def print_header():
    print("\n" + "="*80)
    print("MASTER UPDATED ANALYSIS PIPELINE")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis window: 2021-01-01 to 2025-07-01")
    print(f"Total steps: {len(PIPELINE_STEPS)}")
    print("="*80 + "\n")

def print_step_header(step_num, step):
    print("\n" + "#"*80)
    print(f"STEP {step_num}/{len(PIPELINE_STEPS)}: {step['name']}")
    print("#"*80)
    print(f"Script: {step['script']}")
    print(f"Description: {step['description']}")
    print(f"Estimated time: {step['estimated_time']}")
    print("#"*80 + "\n")

def run_step(step_num, step):
    """Run a single pipeline step"""
    print_step_header(step_num, step)

    script_path = SCRIPT_DIR / step['script']

    if not script_path.exists():
        print(f"⚠️  WARNING: Script not found: {script_path}")
        print(f"   This script needs to be created.")

        if step['required']:
            response = input(f"\n   This is a required step. Skip anyway? (yes/no): ")
            if response.lower() != 'yes':
                return False
        return True

    # Ask user for confirmation
    response = input(f"\n▶  Run this step? (yes/no/quit): ").strip().lower()

    if response == 'quit':
        print("\n❌ Pipeline execution cancelled by user")
        return None  # Signal to quit
    elif response != 'yes':
        print("⏭️  Skipped")
        return True

    # Run the script
    print(f"\n🚀 Starting execution...")
    start_time = time.time()

    try:
        # Run using the same Python interpreter
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SCRIPT_DIR),
            capture_output=False,  # Show output in real-time
            text=True
        )

        elapsed_time = time.time() - start_time
        elapsed_str = f"{elapsed_time/60:.1f} minutes" if elapsed_time > 60 else f"{elapsed_time:.1f} seconds"

        if result.returncode == 0:
            print(f"\n✅ Step completed successfully in {elapsed_str}")
            return True
        else:
            print(f"\n❌ Step failed with return code {result.returncode}")
            if step['required']:
                response = input("\n   Continue anyway? (yes/no): ")
                if response.lower() != 'yes':
                    return False
            return True

    except Exception as e:
        print(f"\n❌ Error running step: {e}")
        if step['required']:
            response = input("\n   Continue anyway? (yes/no): ")
            if response.lower() != 'yes':
                return False
        return True

def print_summary(completed, skipped, failed):
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"✅ Completed: {completed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total steps: {len(PIPELINE_STEPS)}")
    print("="*80 + "\n")

def main():
    print_header()

    print("This pipeline will:")
    print("  1. Download new 2025 news data")
    print("  2. Apply 3:30 PM ET cutoff rule")
    print("  3. Filter and categorize all news")
    print("  4. Run event study analysis for 2021-2025")
    print("  5. Generate all summaries and visualizations")
    print("\n⏱️  Total estimated time: 2-3 hours")

    response = input("\n▶  Start pipeline? (yes/no): ").strip().lower()
    if response != 'yes':
        print("\n❌ Pipeline cancelled")
        return

    # Track progress
    completed = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    # Run each step
    for i, step in enumerate(PIPELINE_STEPS, 1):
        result = run_step(i, step)

        if result is None:  # User quit
            break
        elif result is True:
            if input(f"\n▶  Was this step successful? (yes/no): ").strip().lower() == 'yes':
                completed += 1
            else:
                skipped += 1
        else:
            failed += 1
            print(f"\n❌ Pipeline stopped at step {i} due to failure")
            break

        # Brief pause between steps
        if i < len(PIPELINE_STEPS):
            time.sleep(2)

    # Final summary
    total_time = time.time() - start_time
    total_time_str = f"{total_time/60:.1f} minutes" if total_time > 60 else f"{total_time:.1f} seconds"

    print_summary(completed, skipped, failed)
    print(f"⏱️  Total execution time: {total_time_str}")

    if completed == len(PIPELINE_STEPS):
        print("\n���� SUCCESS! All pipeline steps completed!")
        print("\n📊 Results are available in:")
        print("   - 03-output/news_filtering_2021_2025/")
        print("   - 03-output/event_study_2021_2025/")
        print("   - 03-output/event_study_2021_2025/aggregate_summary/")
        print("   - 03-output/event_study_2021_2025/sector_analysis/")
    elif failed > 0:
        print("\n⚠️  Pipeline completed with failures")
        print("    Review the error messages above and retry failed steps")
    else:
        print("\n✅ Pipeline partially completed")
        print(f"    {completed}/{len(PIPELINE_STEPS)} steps successful")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user (Ctrl+C)")
        sys.exit(1)
