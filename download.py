import urllib.request
import zipfile
import os
import ssl

url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
zip_path = "data/raw/ml-latest-small.zip"

# Ensure the directory exists
os.makedirs("data/raw", exist_ok=True)

# Bypass SSL certificate verification for this download
ssl._create_default_https_context = ssl._create_unverified_context

print("Downloading MovieLens Small dataset...")
urllib.request.urlretrieve(url, zip_path)

print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("data/raw/")

# Clean up the zip file
os.remove(zip_path)
print("Done! Data extracted to data/raw/ml-latest-small/")