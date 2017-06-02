import os
import gzip
import numpy as np
import tensorflow as tf
import scipy, scipy.io, scipy.io.wavfile
import argparse
import pandas
from tqdm import tqdm
pandas.set_option('display.max_colwidth',1000)

# Some parameters
WAV_RATE = 22050
WAV_LENGHT = 220500
dl_dir = '../download/'

# Parse the arguments
parser = argparse.ArgumentParser(description='build the tfwriter')
parser.add_argument('dataset', nargs='*', type=str,
                    default=['train'],
                    help='dataset to build (train or validate)')
args = parser.parse_args()
record_file_name = 'audioset_' + args.dataset[0] + '.tfrecords'

# Open the desired CSVs
# 1) Label indes and names
df_lbls_idx = pandas.read_csv(dl_dir + 'class_labels_indices.csv', quotechar='"')

# 2) Datsaset videos with start time, end time and labels
if args.dataset[0] == "train":
    df_dataset = pandas.read_csv(dl_dir + 'balanced_train_segments.csv', 
                                 names=['f_id', 'start', 'end', 'lbls'],
                                 quotechar='"', 
                                 skipinitialspace=True, 
                                 skiprows=2)

if args.dataset[0] == "validate":
    df_dataset = pandas.read_csv(dl_dir + 'eval_segments.csv', 
                                 names=['f_id', 'start', 'end', 'lbls'],
                                 quotechar='"', 
                                 skipinitialspace=True, 
                                 skiprows=3)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _get_wav_lbls(wav_name):
    # Retrieve the labels corresponding to file <wav_name>
    labels = df_dataset[
        df_dataset.f_id.str.contains(wav_name)] # find the labels
    labels = labels.lbls.to_string(index=False) # /m/085jw,/m/0l14l2
    labels = labels.split(',') # [u'/m/085jw', u'/m/0l14l2']
    labels = map(str, labels)
    return labels


def _get_lbl_idx(lbl):
    # Retrieve index of given label
    label_idx = df_lbls_idx[
        df_lbls_idx.mid.str.contains(lbl)]
    label_idx = int(label_idx['index'])
    return label_idx


# Create a tfrecord file and list the wav's paths
writer = tf.python_io.TFRecordWriter(record_file_name)
wav_paths = [f for f in os.listdir(dl_dir + args.dataset[0] + '/') if '.wav.gz' in f]

for wav_path in tqdm(wav_paths):
    # Read the gzipped wav files
    wav_unzip = gzip.open(dl_dir + args.dataset[0] + '/' + wav_path)
    wav = scipy.io.wavfile.read(wav_unzip)[1]
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)
    wav_const = np.zeros((WAV_LENGHT), dtype='int16')
    wav_const[:wav.shape[0]] = wav
    
    # Retrieve the features to record
    wav_name = wav_path[:11]
    wav_labels = _get_wav_lbls(wav_name)
    wav_labels = map(_get_lbl_idx, wav_labels)
    wav_const = wav_const.tostring()
    
    # Features in an array
    feature={'raw': _bytes_feature(wav_const),
             'labels': _int64_feature(wav_labels),
             'name': _bytes_feature(wav_name) }
    sample = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Write to tfrecord
    writer.write(sample.SerializeToString())

writer.close()