import numpy
from pathlib import Path
from cStringIO import StringIO
from PIL import Image
from tensorflow.python.lib.io import file_io


def get_folder(path):
    output_dir = Path(path)
    # output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_image_paths(directory):
    return ['{}/{}'.format(directory, x) for x in file_io.list_directory(directory) if x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png')]


def load_image(path):
    handler = StringIO()
    handler.write(file_io.read_file_to_string(path, binary_mode=True))
    image = numpy.array(Image.open(handler))
    handler.close()
    return image


def load_images(image_paths, convert=None):
    iter_all_images = (load_image(fn) for fn in image_paths)
    if convert:
        iter_all_images = (convert(img) for img in iter_all_images)
    for i, image in enumerate(iter_all_images):
        if i == 0:
            all_images = numpy.empty((len(image_paths), ) + image.shape, dtype=image.dtype)
        all_images[i] = image
    return all_images
