#!/usr/bin/env python
"""
TooSharpe Data Pipeline Runner
-----------------------------
Standalone script to run the complete data processing pipeline.

Usage:
    python scripts/run_pipeline.py

This script:
1. Cleans all source data files
2. Integrates internal tables (Positions + Reference + Accounts)
3. Builds the intermediary dataset for analytics
4. Saves processed data to outputs/
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from fund_pipeline import data_handling as dh
from fund_pipeline import intermediary_builder as ib


def main():
    """Run the complete data pipeline."""
    print("=" * 60)
    print("TooSharpe Data Pipeline")
    print("=" * 60)
    
    # Paths
    raw_dir = project_root / "data"
    out_dir = project_root / "outputs"
    
    # Validate input directory
    if not raw_dir.exists():
        print(f"❌ Error: Data directory not found: {raw_dir}")
        sys.exit(1)
    
    print(f"\n📁 Raw data directory: {raw_dir}")
    print(f"📁 Output directory: {out_dir}")
    print()
    
    try:
        # Step 1: Clean all source data files
        print("Step 1: Cleaning source data files...")
        print("-" * 60)
        
        ref_path = raw_dir / "ReferenceData.csv"
        pos_path = raw_dir / "PositionLevelPNLAndExposure.csv"
        acct_path = raw_dir / "AccountInformation.csv"
        idx_path = raw_dir / "DailyIndexReturns.csv"
        aum_path = raw_dir / "AUM.csv"
        
        # Check required files
        required_files = [ref_path, pos_path, acct_path, idx_path, aum_path]
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            print(f"❌ Error: Missing required data files:")
            for f in missing_files:
                print(f"   - {f.name}")
            sys.exit(1)
        
        # Clean data files
        print("Cleaning ReferenceData.csv...")
        ref_df = dh.clean_reference_data(str(ref_path), str(out_dir))
        print(f"   ✓ Cleaned {len(ref_df):,} rows")
        
        print("Cleaning PositionLevelPNLAndExposure.csv...")
        pos_df = dh.clean_positions(str(pos_path), str(out_dir))
        print(f"   ✓ Cleaned {len(pos_df):,} rows")
        
        print("Cleaning AccountInformation.csv...")
        acct_df = dh.clean_accounts(str(acct_path), str(out_dir))
        print(f"   ✓ Cleaned {len(acct_df):,} rows")
        
        print("Cleaning DailyIndexReturns.csv...")
        idx_df = dh.clean_index_returns(str(idx_path), str(out_dir))
        print(f"   ✓ Cleaned {len(idx_df):,} rows")
        
        print("Cleaning AUM.csv...")
        aum_df = dh.clean_aum(str(aum_path), str(out_dir))
        print(f"   ✓ Cleaned {len(aum_df):,} rows")
        
        print()
        
        # Step 2: Integrate internal tables
        print("Step 2: Integrating internal tables...")
        print("-" * 60)
        
        integrated_df = dh.integrate_internal(pos_df, ref_df, acct_df, str(out_dir))
        print(f"   ✓ Integrated {len(integrated_df):,} rows")
        print()
        
        # Step 3: Build intermediary dataset
        print("Step 3: Building intermediary dataset...")
        print("-" * 60)
        
        intermediary_df, diagnostics = ib.build_intermediary_from_integrated(
            integrated_df,
            flow_eps=1e-8,
            widx_base=1.0,
            participation_rate=0.2,
            auto_save=True,
            out_root=str(out_dir),
            write_csv=True,
            return_diagnostics=True
        )
        
        print(f"   ✓ Intermediary dataset: {len(intermediary_df):,} rows")
        print(f"   ✓ Saved to: {diagnostics.get('latest', 'N/A')}")
        print()
        
        # Step 4: Build manifest
        print("Step 4: Building manifest...")
        print("-" * 60)
        
        manifest = dh.build_manifest(str(out_dir))
        print(f"   ✓ Manifest saved to: {out_dir / 'manifest.json'}")
        print()
        
        # Summary
        print("=" * 60)
        print("✅ Pipeline completed successfully!")
        print("=" * 60)
        print(f"\nOutput files saved to: {out_dir}")
        print(f"Intermediary dataset: {diagnostics.get('latest', 'N/A')}")
        print("\nYou can now run the Streamlit app:")
        print("  streamlit run app/streamlit_app.py")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("❌ Error during pipeline execution:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        print()
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

