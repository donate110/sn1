
import csv
import random
import os
import subprocess

def download_random_samples(manifest_path, output_dir, count=100):
    # Read manifest
    with open(manifest_path, 'r') as f:
        reader = csv.reader(f)
        files = [row[0] for row in reader if row]

    # Select random files
    if len(files) < count:
        selected_files = files
    else:
        selected_files = random.sample(files, count)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    base_url = "https://pub-77097c3387c340de9ff1bd5e5b443d8d.r2.dev/"

    print(f"Downloading {len(selected_files)} files to {output_dir}...")

    for filename in selected_files:
        url = base_url + filename
        output_path = os.path.join(output_dir, filename)
        
        # Skip if already exists
        if os.path.exists(output_path):
            print(f"Skipping {filename} (already exists)")
            continue

        print(f"Downloading {filename}...")
        try:
            subprocess.run(["curl", "-s", "-o", output_path, url], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {filename}: {e}")

    print("Download complete.")

if __name__ == "__main__":
    download_random_samples("manifest.csv", "samples", 1000)
