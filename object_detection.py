from pathlib import Path
from rembg import remove, new_session
from skimage import transform
from PIL import Image, ImageFilter
import numpy as np


session = new_session()

def remove_background():
    for file in Path('../DASC2-challenge3/').glob('*.png'):
        input_path = str(file)
        output_path = str(file.parent / (file.stem + "_out.png"))

        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input = i.read()
                output = remove(input, session=session)
                o.write(output)
        
            
def add_white_background(image_path):
    image = Image.open(image_path)
    new_image = Image.new('RGBA', image.size, 'WHITE') # Create a white rgba background
    new_image.paste(image, (0, 0), image)              # Paste the image on the background.
    new_image = new_image.resize((299, 299))
    new_image = new_image.filter(ImageFilter.GaussianBlur(radius=1))
    new_image.convert('RGB').save('test.png', 'PNG')  # Save as PNG

remove_background()
add_white_background('kiwi_out.png')