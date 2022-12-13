from pathlib import Path
from rembg import remove, new_session
from tensorflow import keras
from skimage import transform
from PIL import Image, ImageFilter
import numpy as np

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

model_resnet = keras.models.load_model('models/resnet50v2-1')

session = new_session()

for file in Path('../DASC2-challenge3/banana').glob('*.png'):
    input_path = str(file)
    output_path = str(file.parent / (file.stem + "_out.png"))

    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input, session=session)
            o.write(output)
            
def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (299, 299, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image


image = Image.open('banana_out.png')
new_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
new_image.paste(image, (0, 0), image)              # Paste the image on the background. Go to the links given below for details.
new_image = new_image.resize((299, 299))
new_image = new_image.filter(ImageFilter.GaussianBlur(radius=2))
new_image.convert('RGB').save('test.png', "PNG")  # Save as PNG