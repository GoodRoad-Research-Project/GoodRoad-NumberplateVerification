import os
import urllib.request

# Define the directory
model_dir = os.path.expanduser('~/.EasyOCR/model')
os.makedirs(model_dir, exist_ok=True)

# Correct URLs (Using HuggingFace mirror which is often more reliable for direct downloads)
urls = {
    'craft_mlt_25k.pth': 'https://huggingface.co/xiaoyao9184/easyocr/resolve/main/craft_mlt_25k.pth',
    'english_g2.pth': 'https://huggingface.co/xiaoyao9184/easyocr/resolve/main/english_g2.pth'
}

def download_file(filename, url):
    output_path = os.path.join(model_dir, filename)
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        # Check size
        size = os.path.getsize(output_path)
        if size < 1000:  # If file is smaller than 1KB, it's an error page
            print(f"❌ Failed: {filename} is too small ({size} bytes).")
            os.remove(output_path)
        else:
            print(f"✅ Success: {filename} ({size / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")

# Run downloads
for name, link in urls.items():
    download_file(name, link)