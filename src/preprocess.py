import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

def center_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = left + new_size
    bottom = top + new_size
    return image.crop((left, top, right, bottom))

def process_dataset(input_dir, output_dir):
    categories = ['real', 'fake']
    
    total_images = 0
    corrupted_count = 0
    duplicate_count = 0
    final_count = 0
    
    seen_hashes = set()
    
    for category in categories:
        cat_input_dir = os.path.join(input_dir, category)
        cat_output_dir = os.path.join(output_dir, category)
        
        if not os.path.exists(cat_output_dir):
            os.makedirs(cat_output_dir)
            
        if not os.path.exists(cat_input_dir):
            continue
            
        files = os.listdir(cat_input_dir)
        for filename in tqdm(files, desc=f"Processing {category}"):
            total_images += 1
            input_path = os.path.join(cat_input_dir, filename)
            output_path = os.path.join(cat_output_dir, filename)
            
            # Read and skip corrupted
            try:
                with Image.open(input_path) as img:
                    img.verify() # Verify that it is, in fact, an image
            except Exception:
                corrupted_count += 1
                continue
                
            try:
                # Reopen to perform modifications
                with Image.open(input_path) as img:
                    # Convert to RGB
                    img = img.convert('RGB')
                    
                    # Center crop to square
                    img = center_crop(img)
                    
                    # Resize to 256x256
                    img = img.resize((256, 256), Image.Resampling.LANCZOS)
                    
                    # Normalize pixel values to 0-255 uint8 (already true for 'RGB' PIL image)
                    img_np = np.array(img)
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                    
                    # Optional: use cv2 just because it's required by the user, though we used np and PIL mostly.
                    # Convert RGB to BGR for opencv if we were using it to save, but PIL works well.
                    # Let's stick to PIL for saving.
                    img_pil = Image.fromarray(img_np)
                    
                    # Perceptual hash to find duplicates
                    img_hash = imagehash.phash(img_pil)
                    
                    if img_hash in seen_hashes:
                        duplicate_count += 1
                        continue
                        
                    seen_hashes.add(img_hash)
                    
                    # Save cleaned image
                    img_pil.save(output_path)
                    final_count += 1
                    
            except Exception:
                corrupted_count += 1
                continue
                
    print(f"Total images: {total_images}")
    print(f"Corrupted removed: {corrupted_count}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Final images: {final_count}")

if __name__ == "__main__":
    process_dataset("dataset", "clean_dataset")
