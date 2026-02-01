import pandas as pd
import os
import glob
from tqdm import tqdm

class FormatMigrator:
    def __init__(self, source_dir="data/raw", target_dir="data/raw"):
        self.source_dir = source_dir
        self.target_dir = target_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    def convert_all(self):
        # Find all CSVs
        csv_files = glob.glob(f"{self.source_dir}/*.csv")
        print(f"[MIGRATION] Found {len(csv_files)} CSV files in {self.source_dir}. Starting conversion...")

        success_count = 0
        
        for file_path in tqdm(csv_files, desc="Converting"):
            try:
                ticker = os.path.basename(file_path).replace(".csv", "")
                target_path = os.path.join(self.target_dir, f"{ticker}.parquet")
                
                # Skip if already exists? Maybe not, to ensure latest format.
                
                # 1. READ (High Friction)
                # Ensure date parsing handles index correctly
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # 2. VALIDATE (Law 3)
                if df.empty:
                    continue
                
                # Ensure numerical columns are floats, not strings
                cols_to_fix = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                for col in cols_to_fix:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # 3. WRITE (Low Friction)
                # Ensure Date is a column for DuckDB compatibility
                if df.index.name != 'Date':
                    df.index.name = 'Date'
                df.reset_index(inplace=True)
                
                df.to_parquet(target_path, index=False)
                success_count += 1
                
                # Optional: Delete CSV after success? 
                # User asked to "clear it also" previously, but maybe safer to keep until verified.
                # For now, let's just convert.

            except Exception as e:
                print(f" Failed to convert {file_path}: {e}")

        print(f" MIGRATION COMPLETE. {success_count} files optimized.")

if __name__ == "__main__":
    migrator = FormatMigrator()
    migrator.convert_all()
