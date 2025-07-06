import numpy as np
import os

folder = "./demos_collected/run002"
all_ok = True
issues = []
    
# Check if the folder exists
if not os.path.exists(folder):
    print(f"Folder does not exist: {folder}")
    all_ok = False
    issues.append(f"Folder does not exist: {folder}")

# Check if the folder is empty
if not os.listdir(folder):
    print(f"Folder is empty: {folder}")
    all_ok = False
    issues.append(f"Folder is empty: {folder}")
    # exit early if no files to check
for fname in os.listdir(folder):
    if fname.endswith(".npz"):
        print(f"\nChecking file: {fname}")
        try:
            arr = np.load(os.path.join(folder, fname), allow_pickle=True)["data"]
        except Exception as e:
            print(f"Failed to load: {e}")
            all_ok = False
            issues.append(f"{fname}: Failed to load ({e})")
            continue
        print("Shape:", arr.shape)
        # Check if the array is empty
        if np.issubdtype(arr.dtype, np.number):
            if np.all(arr == 0):
                print("Warning: All data in this file are zeros!")
                all_ok = False
                issues.append(f"{fname}: All zeros")
            elif np.isnan(arr).any():
                print("Warning: NaN detected in data!")
                all_ok = False
                issues.append(f"{fname}: NaN detected")
            elif np.isinf(arr).any():
                print("Warning: Inf detected in data!")
                all_ok = False
                issues.append(f"{fname}: Inf detected")
            else:
                if arr.ndim == 1:
                    zero_rows = np.where(arr == 0)[0]
                else:
                    zero_rows = np.where(~arr.any(axis=-1))[0]
                if len(zero_rows) > 0:
                    print(f"{len(zero_rows)} rows are all zeros (example indices: {zero_rows[:10]})")
                    all_ok = False
                    issues.append(f"{fname}: {len(zero_rows)} all-zero rows")
                else:
                    print("Data is normal, no all-zero rows found.")
            try:
                print(f"Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean()}")
            except Exception as e:
                print(f"Could not compute statistics: {e}")
        else:
            print("Non-numeric data, skipping numeric checks.")
        try:
            print("First 2 data samples:\n", arr[:2])
        except Exception as e:
            print(f"Could not print data samples: {e}")

print("\n===== Summary =====")
if all_ok:
    print("All files checked and no issues found. All data is normal.")
else:
    print("Some issues were found:")
    for issue in issues:
        print(" -", issue)