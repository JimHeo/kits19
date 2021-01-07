from starter_code.utils import load_case
from starter_code.visualize import hu_to_grayscale
import numpy as np
import nibabel as nb
import argparse
import os
from pathlib import Path
from numpy.lib.twodim_base import vander
import tensorflow as tf
from convert_tfrecord import ioutil
from tqdm import trange

HU_MAX = 512
HU_MIN = -512

def convert_axial_nii_to_tfrecord(case: int, output_path):
    encoder = ioutil.TFRecordEncoder()
    options = ioutil.TFRecordOptions

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # with tf.python_io.TFRecordWriter(output_path, options) as writer:
    with tf.io.TFRecordWriter(output_path, options) as writer:
        volume, seg = load_case(case)
        affine = volume.affine
        volume = volume.get_data()
        seg = seg.get_data().astype(np.uint8)
        vol = hu_to_grayscale(volume, HU_MIN, HU_MAX)
        for i in trange(vol.shape[0]):
            writer.write(encoder.encode(case, vol[i], seg, affine, i, "axial").SerializeToString())

        tf.compat.v1.logging.info(f'Wrote {case} to {output_path}')

def main(args):
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    convert_axial_nii_to_tfrecord(args.case, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('build_data', description='''
    Converts NIFTI volumes to tfrecord.
    ''')
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--output-path', required=True)

    main(parser.parse_args())
