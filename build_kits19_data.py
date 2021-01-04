import tensorflow as tf
import nibabel as nib
import numpy as np
import collections
import sys
from starter_code.utils import load_case

def _int64_list_feature(values):
  """Returns a TF-Feature of int64_list.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _float_list_feature(values):
  """Returns a TF-Feature of float_list.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

# path to save the TFRecords file
train_filename = 'kits19_train.tfrecord'

# open the file
writer = tf.io.TFRecordWriter(train_filename)

cases = 210
# iterate through all .nii files:
for case in range(cases):

    # Load the image and label
    vol, seg = load_case(case)
    
    sys.stdout.write('\r>> Converting image %d/%d' % (case + 1, cases))
    sys.stdout.flush()

    # Create a feature
    feature = {'train/seg': _int64_list_feature(seg.get_data().ravel().astype(tf.int64)),
               'train/vol': _float_list_feature(vol.get_data().ravel())}
               
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()