from __future__ import division, print_function, absolute_import, unicode_literals

import argparse
import os
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework.errors_impl import NotFoundError
from PIL import Image
import numpy

from lib.utils import get_image_paths, load_images
from lib.training_data import get_training_data, stack_images

from lib.model import autoencoder_A
from lib.model import autoencoder_B
from lib.model import encoder, decoder_A, decoder_B


def dispatch(src_dir,
             dst_dir,
             job_dir,
             batch_size=64,
             num_epochs=10):
    load_models(job_dir)
    print('load model weights')

    images_A = get_image_paths(src_dir)
    images_B = get_image_paths(dst_dir)
    images_A = load_images(images_A) / 255.0
    images_B = load_images(images_B) / 255.0

    images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

    for epoch in range(num_epochs):
        warped_A, target_A = get_training_data(images_A, batch_size)
        warped_B, target_B = get_training_data(images_B, batch_size)

        loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
        loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
        print( loss_A, loss_B )

    save_model_weights(job_dir, encoder, 'encoder.h5')
    save_model_weights(job_dir, decoder_A, 'decoder_A.h5')
    save_model_weights(job_dir, decoder_B, 'decoder_B.h5')
    print('save model weights')

    test_A = target_A[0:14]
    test_B = target_B[0:14]
    save_sample(job_dir, test_A, test_B)
    print('save sample outputs')


def load_models(job_dir):
    def load_model_weights(model, fname):
        if not job_dir.startswith("gs://"):
            try:
                model.load_weights(os.path.join(job_dir, "models/{}".format(fname)))
            except IOError as e:
                print('Existing models {} not found. New models will be created...'.format(fname))
        else:
            try:
                copy_file_from_gcs(job_dir + '/models', fname)
                model.load_weights(fname)
            except NotFoundError as e:
                print('Existing models {} not found. New models will be created...'.format(fname))

    load_model_weights(encoder, 'encoder.h5')
    load_model_weights(decoder_A, 'decoder_A.h5')
    load_model_weights(decoder_B, 'decoder_B.h5')


def save_model_weights(job_dir, model, fname):
    if not job_dir.startswith('gs://'):
        model.save_weights(os.path.join(job_dir, 'models/{}'.format(fname)))
    else:
        model.save_weights(fname)
        copy_file_to_gcs(job_dir + '/models', fname)


def save_sample(job_dir, test_A, test_B, fname='sample.jpg'):
    figure_A = numpy.stack([
        test_A,
        autoencoder_A.predict(test_A),
        autoencoder_B.predict(test_A),
    ], axis=1)
    figure_B = numpy.stack([
        test_B,
        autoencoder_B.predict(test_B),
        autoencoder_A.predict(test_B),
    ], axis=1)

    figure = numpy.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4, 7) + figure.shape[1:])
    figure = stack_images(figure)

    figure = numpy.clip(figure * 255, 0, 255).astype('uint8')
    image = Image.fromarray(figure)

    if not job_dir.startswith('gs://'):
        image.save(os.path.join(job_dir, 'outputs/{}'.format(fname)))
    else:
        image.save(fname)
        copy_file_to_gcs(job_dir + '/outputs', fname)


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def copy_file_from_gcs(job_dir, file_path):
    with file_io.FileIO(os.path.join(job_dir, file_path), mode='r') as input_f:
        with file_io.FileIO(file_path, mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir containing face images to transform to destination face')
    parser.add_argument('--dst-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir containing face images to be transformed from source face')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=10,
                        help='Maximum number of epochs on which to train')
    parse_args, unknown = parser.parse_known_args()

    dispatch(**parse_args.__dict__)

