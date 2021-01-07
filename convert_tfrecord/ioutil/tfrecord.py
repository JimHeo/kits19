import numpy as np
import tensorflow as tf
import collections
import six

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


def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  def norm2bytes(value):
    return value.encode() if isinstance(value, str) and six.PY3 else value

  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


Options = tf.io.TFRecordOptions(compression_type='GZIP')
# Options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

CString = Options.get_compression_type_string(Options)


class Encoder:
  """Encodes a numpy volume array as tfrecord."""

  def encode(self, case, volume, seg, affine):
    """Returns numpy volume array as tfrecord string.

    Args:
      volume: numpy array
    Returns:
      tfrecord: tfrecord serialized to string
    """
    with tf.name_scope('encode'):
      return tf.train.Example(features=tf.train.Features(feature={
          'image/encoded': _bytes_list_feature(volume.astype(np.float32).tobytes()),
          'image/format': _bytes_list_feature("nii.gz"),
          'image/height': _int64_list_feature(volume.shape[0]),
          'image/width': _int64_list_feature(volume.shape[1]),
          'image/channels': _int64_list_feature(volume.shape[3]),
          'image/frame': _int64_list_feature(volume.shape[2]),
          'image/case': _int64_list_feature(case),
          'image/affine': _bytes_list_feature(affine.astype(np.float32).tobytes()),
          'image/segmentation/class/encoded':  _bytes_list_feature(seg.astype(np.int32).tobytes()),
          'image/segmentation/class/format': _bytes_list_feature("nii.gz"),
      }))


class Decoder:
  """Decodes a tfrecord volume to tensor."""

  def __init__(self):
    self.features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.VarLenFeature(tf.string),
        'image/height': tf.io.VarLenFeature(tf.int64),
        'image/width': tf.io.VarLenFeature(tf.int64),
        'image/channels': tf.io.VarLenFeature(tf.int64),
        'image/frame': tf.io.VarLenFeature(tf.int64),
        'image/case': tf.io.VarLenFeature(tf.int64),
        'image/affine': tf.io.FixedLenFeature([], tf.string),
        'image/segmentation/class/encoded':  tf.io.FixedLenFeature([], tf.string),
        'image/segmentation/class/format': tf.io.VarLenFeature(tf.string),
    }

  def decode(self, example):
    """Returns tensor from tfrecord string.

    Args:
      example: tfrecord string
    Returns:
      volume: volume tensor
    """
    with tf.name_scope('decode'):
      features = tf.io.parse_single_example(example, features=self.features)

      volume = tf.io.decode_raw(features['image/encoded'], tf.float32)
      volume_format = features['image/format'].values
      height = features['image/height'].values
      width = features['image/width'].values
      frame = features['image/frame'].values
      channels = features['image/channels'].values
      case = features['image/case'].values
      affine = tf.io.decode_raw(features['image/affine'], tf.float32)
      seg = tf.io.decode_raw(features['image/segmentation/class/encoded'], tf.int32)
      seg_format = features['image/segmentation/class/format'].values
      
      return volume, volume_format, height, width, frame, channels, case, affine, seg, seg_format
