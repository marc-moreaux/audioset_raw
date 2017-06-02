from Queue import Queue
from threading import Thread
import scipy.io.wavfile
import pandas as pd
import numpy as np
import random
import time
import gzip
import os

# Some parameters
N_CLASSES = 527
WAV_RATE = 22050
WAV_LENGHT = 220500
dl_dir = './download/'
ds_dir = dl_dir + 'train/'

label_indices = pd.read_csv(dl_dir + 'class_labels_indices.csv', quotechar='"')
label_indices = label_indices.values.tolist()
dataset = pd.read_csv(dl_dir + 'balanced_train_segments.csv', 
                      names=['f_id', 'start', 'end', 'lbls'],
                      quotechar='"', skiprows=2, skipinitialspace=True)
dataset = dataset.values.tolist()


def _get_wav_lbls(wav_name):
    # Retrieve the labels corresponding to file <wav_name>
    labels = [item[3] for item in dataset if item[0] == wav_name][0]
    labels = labels.split(',') # [u'/m/085jw', u'/m/0l14l2']
    labels = map(str, labels)
    return labels


def _get_lbl_idx(lbl):
    # Retrieve index of given label
    label_idx = [item[0] for item in label_indices if item[1] == lbl][0]
    return label_idx


def _get_sample(wav_path):
    # Extract WAV's name
    wav_name = wav_path.split('/')[-1][:11]

    # Open the wav and retrieve its labels
    wav = gzip.open(wav_path)
    wav = scipy.io.wavfile.read(wav)[1]
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)
    wav_const = np.zeros((WAV_LENGHT), dtype='int16')
    wav_const[:wav.shape[0]] = wav

    # Acess WAV's labels    
    wav_lbls_idx = _get_wav_lbls(wav_name)
    wav_lbls_idx = map(_get_lbl_idx, wav_lbls_idx)
    wav_lbls_dense = np.zeros((N_CLASSES))
    wav_lbls_dense[wav_lbls_idx] = 1
    
    return wav_const, wav_lbls_dense


def fill_batch_queue(paths_q, batch_q, batch_size):
    while paths_q.qsize() > batch_size:
        # Create the batch of <batch_size> samples
        wav_batch = np.ndarray((batch_size, WAV_LENGHT))
        lbls_batch = np.ndarray((batch_size, N_CLASSES))
        for idx in range(batch_size):
            wav_path = paths_q.get()
            paths_q.task_done()
            wav, lbls = _get_sample(wav_path)
            wav_batch[idx,:] = wav
            lbls_batch[idx,:] = lbls

        # Put the batch in another queue
        batch_q.put((wav_batch, lbls_batch))



def _get_queue_from_dir_path(dataset_dir):
    # Read wav_paths and put them into a queue
    paths_q = Queue(maxsize=0)
    wav_paths = [ds_dir + f for f in os.listdir(ds_dir) if '.wav.gz' in f]
    random.shuffle(wav_paths)
    for item in wav_paths:
        paths_q.put(item)

    return paths_q


def get_batchs(batch_size=10, n_epochs=1, num_threads=4):
    for _ in range(n_epochs):
        paths_q = _get_queue_from_dir_path(ds_dir)
        batch_q = Queue(maxsize=num_threads*2)

        # Start the Threads :
        # They'll stop when size of <paths_q> is less than a batch size.
        for i in range(num_threads):
            worker = Thread(target=fill_batch_queue, 
                            args=(paths_q, batch_q, batch_size))
            worker.setDaemon(True)
            worker.start()


        while paths_q.qsize() > batch_size:
            if batch_q.empty() == False:
                wav_batch, lbls_batch = batch_q.get()
                batch_q.task_done()
                print lbls_batch[0][:10]
                yield wav_batch, lbls_batch






get_batchs()
