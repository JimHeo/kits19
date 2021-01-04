from starter_code.utils import load_case
from starter_code.visualize import hu_to_grayscale
import numpy as np
import nibabel as nb
import argparse
import os
from numpy.lib.twodim_base import vander
import tensorflow as tf

from convert_tfrecord import ioutil

def convert_nii_to_tfrecord(case: int, output_path):
    encoder = ioutil.TFRecordEncoder()
    options = ioutil.TFRecordOptions

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # with tf.python_io.TFRecordWriter(output_path, options) as writer:
    with tf.io.TFRecordWriter(output_path, options) as writer:
        volume, seg = load_case(case)
        affine = volume.affine
        volume = volume.get_data()
        seg = seg.get_data()
        volume = ioutil.voxel_to_tensor_space(volume)
        seg = ioutil.voxel_to_tensor_space(seg)
        writer.write(encoder.encode(case, volume, seg, affine).SerializeToString())

        tf.compat.v1.logging.info(f'Wrote {case} to {output_path}')
            
def convert_tfrecord_to_nii(input_path, output_path):
    decoder = ioutil.TFRecordDecoder()
  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
  
    tf_dataset = tf.data.TFRecordDataset(filenames=input_path, compression_type='GZIP')
    for serialized_example in tf_dataset:
        volume, volume_format, height, width, frame, channels, case, affine, seg, seg_format = decoder.decode(serialized_example)
        break
  
    affine_shape = (4, 4)
    sh = (height.numpy()[0], width.numpy()[0], frame.numpy()[0], channels.numpy()[0])
    s = np.reshape(seg.numpy(), sh)
    v = np.reshape(volume.numpy(), sh)
    a = np.reshape(affine.numpy(), affine_shape)
  
    volume = ioutil.tensor_to_voxel_space(v)
    seg = ioutil.tensor_to_voxel_space(s)
    vol_nii = nb.Nifti1Image(volume, affine=a)
    seg_nii = nb.Nifti1Image(seg, affine=a)
    nb.save(vol_nii, output_path)
    nb.save(seg_nii, output_path + '.seg')
    
    # tf.logging.info(f'Wrote {input_path} to {output_path}')
    tf.compat.v1.logging.info(f'Wrote {input_path} to {output_path}')

def main(args):
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    if args.convert_to == 'tfrecord':
        if args.case:
            convert_nii_to_tfrecord(args.case, args.output_path)
    elif args.convert_to == 'nifti' or args.convert_to == 'nii' :
        if args.input_path:
            convert_tfrecord_to_nii(args.input_path, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('build_data', description='''
    Converts NIfTI volumes to tfrecord.
    ''')
    parser.add_argument('--convert-to', default='tfrecord', choices=['tfrecord', 'nii', 'nifti'])
    parser.add_argument('--case', type=int)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--input-path')

    main(parser.parse_args())
