import tensorflow as tf
import matplotlib.pyplot as plt
import scipy, scipy.io, scipy.io.wavfile
import sounddevice as sd
import numpy as np
import pandas
import time

# Some parameters
WAV_RATE = 22050
WAV_LENGHT = 220500
N_CLASSES = 527


df_lbls_idx = pandas.read_csv("class_labels_indices.csv", quotechar='"')
df_lbls_idx = np.array(df_lbls_idx)
def _get_label_names(dense_labels):
    indexes = np.argwhere(dense_labels).reshape(-1)
    indexes = list(indexes)
    indexes = df_lbls_idx[indexes, 2]
    indexes = ", ".join(indexes)

    return indexes


def read_and_decode(filename_queue):
    """Read and decode my tf_record file containing the raw audioset
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Features expected on the tf_record
    features = tf.parse_single_example(
        serialized_example,
        features={'raw': tf.FixedLenFeature([], tf.string),
                  'labels': tf.VarLenFeature(tf.int64),
                  'name': tf.FixedLenFeature([], tf.string) })

    # Read and decode raw audio
    wav_raw = tf.decode_raw(features['raw'], tf.int16)    
    wav_raw = tf.cast(wav_raw, tf.float64) * (1./30000)
    wav_raw = tf.reshape(wav_raw, [WAV_LENGHT])

    # Read and decode labels and name
    wav_labels = features['labels']
    wav_labels = tf.sparse_to_dense(features["labels"].values, 
                                    (N_CLASSES,), 1,
                                    validate_indices=False)
    wav_labels = tf.cast(wav_labels, tf.bool)
    wav_name = features['name']
        
    return wav_raw, wav_labels


def inputs(tfrecords_filename, batch_size=30, num_epochs=None):
    """Reads input data num_epochs times.
    Args:
        dataset: select between 'train', 'valid' and 'test'
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
          train forever.
    Returns:
        A tuple (images, labels), where:
        * wavs is a float tensor with shape [batch_size, WAV_LENGHT]
          in the range [-1, 1].
        * labels is an int32 tensor with shape [batch_size] with the true label,
          a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [tfrecords_filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename queue.
    wav, lbls = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into <batch_size> batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    wavs, sparse_labels = tf.train.shuffle_batch(
        [wav, lbls], 
        batch_size=batch_size, 
        num_threads=2,
        capacity=batch_size * 6,
        min_after_dequeue=batch_size * 3)

    return wavs, sparse_labels



"""
Here begins the desired shit ;)
"""
def main():
    waves, labels = inputs("audioset_train.tfrecords", batch_size=3)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session()  as sess:
        sess.run(init_op)
        
        # Start batch threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        m_waves, m_labels = sess.run([waves, labels])
        for wav, lbl in zip(m_waves, m_labels):
            print _get_label_names(lbl)
            sd.play(wav, WAV_RATE)
            time.sleep(10)
            sd.stop()
        
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()