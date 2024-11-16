import numpy as np
from PIL import Image
import os

def text_to_hex(text):
    # encode the text to bytes, then convert to hexadecimal
    hex_output = text.encode("utf-8").hex()
    return hex_output


def extract_transforming_name(obj):
    # Get the full class name
    full_name = str(obj)
    # Extract the class name
    class_name = full_name.split('.')[-1].replace("'>", '')
    return class_name



def save_image(image_array: np.ndarray, save_path: str):
    
    img = Image.fromarray(image_array)
    
    _, file_extension = os.path.splitext(save_path)
    
    file_extension = file_extension.lower()
    
    if file_extension == '.png':
        # PNG supports lossless compression by default
        img.save(save_path, format='PNG', compress_level=0)  # compress_level=0 for no compression
    
    elif file_extension in ['.tif', '.tiff']:
        # TIFF with LZW compression is lossless
        img.save(save_path, format='TIFF', compression='tiff_lzw')
    
    elif file_extension == '.bmp':
        # BMP is a lossless format by nature
        img.save(save_path, format='BMP')
    
    elif file_extension == '.webp':
        # WEBP can be lossless if specified
        img.save(save_path, format='WEBP', lossless=True)
    
    else:
        raise ValueError(f"Unsupported or non-lossless format for extension: .{file_extension}")