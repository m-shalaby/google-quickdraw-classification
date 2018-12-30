import numpy as np
from ctypes import cdll, create_string_buffer
lib = cdll.LoadLibrary('Noise Reduction Library\\reduce.dll')

def  denoise_image(image):
    """
    Assumes image is a numpy array of length 10000. Convert to a python bytes then returns a new numpy array with the denoised image.
    """
    image_bytes = np.ndarray.tobytes(image.astype(np.uint8))
    pointer = create_string_buffer(len(image))
    pointer.raw = image_bytes
    lib.reduceImageArray(pointer)
    image_bytes = pointer.raw
    new_image = np.frombuffer(image_bytes, dtype=np.uint8)

    return new_image
  