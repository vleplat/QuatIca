import urllib.request
import os

def download_kodak_images():
    """Download Kodak test images for matrix completion testing"""
    
    # Create dataset directory if it doesn't exist
    os.makedirs("dataset", exist_ok=True)
    
    # Kodak test images from https://r0k.us/graphics/kodak/
    # These are high-quality, lossless PNG images perfect for testing
    # Note: Direct download URLs may not work due to server configuration
    # Users can manually download from the website
    kodak_images = {
        "kodim16.png": "https://r0k.us/graphics/kodak/kodim16.png",  # "land ahoy" - main test image
    }
    
    print("Kodak Test Images Download Utility")
    print("Source: https://r0k.us/graphics/kodak/")
    print("These are high-quality, lossless PNG images perfect for matrix completion testing.")
    print()
    
    successful_downloads = []
    
    for filename, url in kodak_images.items():
        filepath = os.path.join("dataset", filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"✓ {filename} already exists in ../.../../data/images/")
            successful_downloads.append(filename)
            continue
            
        try:
            print(f"Attempting to download {filename} from {url}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"✓ Successfully downloaded {filename} ({os.path.getsize(filepath)} bytes)")
            successful_downloads.append(filename)
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            print(f"  This is expected - direct downloads may be restricted.")
            print(f"  Please manually download from: {url}")
            continue
    
    print()
    print("Download Summary:")
    print(f"✓ Successfully downloaded {len(successful_downloads)} images")
    
    if successful_downloads:
        print("Available test images:")
        for img in successful_downloads:
            print(f"  - ../.../../data/images/{img}")
        
        print()
        print("Usage:")
        print("  - kodim16.png: Main test image for script_real_image_completion.py")
        print("  - Other images: Additional test cases for algorithm validation")
        print()
        print("Note: These images are 768x512 or 512x768 pixels in size.")
        print("The framework will automatically resize them as needed.")
    else:
        print("No images were downloaded automatically.")
        print()
        print("Manual Download Instructions:")
        print("1. Visit: https://r0k.us/graphics/kodak/")
        print("2. Download the following images:")
        print("   - kodim16.png (\"land ahoy\") - Main test image")
        print("   - kodim01.png (\"brick wood, wood brick\")")
        print("   - kodim02.png (\"knob and bolt on red door\")")
        print("   - kodim03.png (\"hats\")")
        print("   - kodim04.png (\"Red Riding Hood\")")
        print("   - kodim05.png (\"motocross\")")
        print("3. Place them in the ../.../../data/images/ directory")
        print()
        print("The framework will use synthetic images until real images are available.")
    
    return successful_downloads

def download_sample_image():
    """Legacy function for backward compatibility"""
    return download_kodak_images()

if __name__ == "__main__":
    download_kodak_images() 