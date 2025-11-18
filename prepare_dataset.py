import pandas as pd
import os

DATA_DIR = "data"

# --- List 1: Files with ASSIGNED labels ---
# (Script assumes 0=Fake, 1=Real)
files_with_assigned_label = [
    ("fake.csv", 0),            # Fake = 0 
    ("True.csv", 1),            # Real = 1
    ("gossipcop_fake.csv", 0),    # Fake = 0
    ("gossipcop_real.csv", 1)     # Real = 1
]

# --- List 2: Files with INTERNAL labels ---
# (Script will find the label column)
files_with_internal_label = [
    "train (3).xlsx - Sheet1.csv",
    "valid.xlsx - Sheet1.csv"
]

# --- Columns to detect ---
TEXT_COLS = ["text", "content", "title", "headline"]
LABEL_COLS = ["label", "target", "class", "isfake"]

def detect_column(df, col_list):
    """Finds the first matching column from a list."""
    for col in col_list:
        if col in df.columns:
            return col
    return None

final_frames = []
print("✅ Starting dataset preparation...")

# --- Process 1: Files with ASSIGNED labels ---
print("\n--- Processing Files with Assigned Labels ---")
for filename, label in files_with_assigned_label:
    path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(path):
        print(f"❌ Missing file: {filename} — skipping.")
        continue

    print(f"\n✅ Loading: {filename}")
    # Add on_bad_lines='skip' to handle potential CSV errors
    df = pd.read_csv(path, on_bad_lines='skip')

    # Detect text column
    text_col = detect_column(df, TEXT_COLS)
    if not text_col:
        print(f"❌ No text column found in {filename} — skipping.")
        continue
    
    print(f"   ➤ Found text column: {text_col}")

    # Keep only text + label
    df = df[[text_col]].copy()
    df.rename(columns={text_col: "text"}, inplace=True)

    df["label"] = label
    print(f"   ➤ Assigned label: {label} (Fake=0, Real=1)")

    final_frames.append(df)


# --- Process 2: Files with INTERNAL labels ---
print("\n--- Processing Files with Internal Labels ---")
for filename in files_with_internal_label:
    path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(path):
        print(f"❌ Missing file: {filename} — skipping.")
        continue

    print(f"\n✅ Loading: {filename}")
    # Add on_bad_lines='skip' to handle potential CSV errors
    df = pd.read_csv(path, on_bad_lines='skip')

    # Detect text and label columns
    text_col = detect_column(df, TEXT_COLS)
    label_col = detect_column(df, LABEL_COLS)

    if not text_col or not label_col:
        print(f"❌ Missing required columns in {filename} (Need text and label) — skipping.")
        continue
    
    print(f"   ➤ Found text column: {text_col}")
    print(f"   ➤ Found label column: {label_col}")

    # Keep only text + label
    df = df[[text_col, label_col]].copy()
    df.rename(columns={text_col: "text", label_col: "label"}, inplace=True)
    
    # --- IMPORTANT: Remapping Labels ---
    # This script's standard is 0=Fake, 1=Real.
    # We assume your new files use 0=Real, 1=Fake (a common standard).
    # This code remaps them to match.
    if df['label'].nunique() == 2: # Check if labels are binary
        print("   ➤ Remapping labels to 0=Fake, 1=Real standard...")
        # This maps 0 -> 1 and 1 -> 0
        df['label'] = df['label'].map({0: 1, 1: 0}) 
    else:
        print(f"   ➤ Warning: Label column in {filename} is not binary. Using as-is.")

    final_frames.append(df)


# --- Final Step: Combine, Shuffle, and Save ---
if not final_frames:
    print("\n❌ No data was loaded. Exiting.")
else:
    print("\n--- Combining All Datasets ---")
    combined_df = pd.concat(final_frames, ignore_index=True)
    combined_df.dropna(subset=["text", "label"], inplace=True)
    
    # Ensure label is integer
    combined_df['label'] = combined_df['label'].astype(int)

    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save final file
    output_path = os.path.join(DATA_DIR, "combined_news.csv")
    combined_df.to_csv(output_path, index=False)

    print("\n✅ COMBINED DATASET CREATED SUCCESSFULLY ✅")
    print("✅ Saved to:", output_path)
    print("✅ Total records:", len(combined_df))

    print("\n✅ Preview:")
    print(combined_df.head())
    
    print("\n✅ Label distribution:")
    print(combined_df['label'].value_counts())