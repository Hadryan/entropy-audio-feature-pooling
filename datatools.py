import os
import re
from glob import glob
from typing import List

import numpy as np
import tensorflow as tf

PATTERN_PREFIX_LABEL_UID = "(.+\/)?(\w+)\/([^_]+)_.+wav"

VALIDATION_LIST_TXT = 'train/validation_list.txt'
TEST_LIST_TXT = 'train/testing_list.txt'
SAMPLING_RATE = 16000


def get_filenames_of_train_val_test_sets(data_dir: str, labels):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    @type data_dir: object
    """
    # ids = {i: label for i, label in enumerate(labels)}
    # names = {label: i for i, label in ids.items()}
    label2id = {label: i for i, label in enumerate(labels)}

    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile(PATTERN_PREFIX_LABEL_UID)
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

    valset = get_file_set(data_dir, pattern, VALIDATION_LIST_TXT)
    testset = get_file_set(data_dir, pattern, TEST_LIST_TXT)

    labels_set = set(labels)
    train_files, train_labels, val_files, val_labels, test_files, test_labels = [], [], [], [], [], []
    for entry in all_files:
        entry = entry.replace('\\', '/')
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in labels_set:
                label = 'unknown'

            label_id = label2id[label]

            # sample = (str(label_id), uid, entry)
            # print(f"sample with {type(label_id)}, {type(uid)}, {type(entry)}")
            if uid in valset:
                val_files.append(entry)
                val_labels.append(label_id)
            elif uid in testset:
                test_files.append(entry)
                test_labels.append(label_id)
            else:
                train_files.append(entry)
                train_labels.append(label_id)

    print(f'There are {len(train_files)} train, {len(val_files)} val and {len(test_files)} test samples')
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


def get_file_set(data_dir: str, pattern, file_list_path: str):
    with open(os.path.join(data_dir, file_list_path), 'r') as f:
        files = f.readlines()
    file_set = set()
    for entry in files:
        r = re.match(pattern, entry)
        if r:
            file_set.add(r.group(3))
    return file_set


def loadGeneral(data_dir, mode, labels, filePath):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    ids = {i: name for i, name in enumerate(labels)}
    names = {name: i for i, name in ids.items()}
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, filePath))

    possible = set(labels)
    train, val = [], []
    for entry in all_files:
        entry = entry.replace('\\', '/')
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = names[label]

            sample = (label_id, uid, entry)

            train.append(sample)

    print('There are {} samples '.format(len(train)))
    return train


# array-the array to be padded
# actualSize- the size the array should be padded to.
# This is a placeholder padding,later this will have a mode to decide if its 0 padding or noise padding.
def padArray(array, actualSize):
    sizeofArray = len(array)
    asa = actualSize - sizeofArray
    for i in range(asa):
        array = np.append(array, 0.0001)
    array = np.float32(array)
    return array


# This dataset loads the tuplet data from load.
# Then reads the files and seperates them into 2 tables.
# 1-datas table-made up of the float values of the wav file-isolating the first second.
# If the file is too small-it pads the array.
# Otherwise if the file is too big it cuts the file information to 1 sec
# 2-labels   table-made up of the labels of the respective wav files.
def dataset(data, labels):
    datas = []
    labels = []
    # for every tuplet:
    for (label_id, uid, fname) in data:
        try:
            # read the file and turn the information into float
            # _, wav = wavfile.read(fname)
            # wav = wav.astype(np.float32) / np.iinfo(np.int16).max
            wav = path2audio(fname)
            # The period wanted is 1 second-16000
            # If the file is too big it keeps the first second of the file.
            # Otherwise if the file is too small it pads the array
            L = 16000
            if len(wav) > 16000:
                wav = wav[0:L]
            if len(wav) < 16000:
                wav = padArray(wav, 16000)

            # appends the information to the respective tables.
            labels.append(label_id)
            datas.append(wav)
            # throws error if necessary
        except Exception as err:
            print("Error while reading wav file")
            print(err, label_id, uid, fname)
    return datas, labels


def inputConfig(data):
    # Creates an empty ndarray for the new data.
    lenOf = len(data)
    datas = np.ndarray(shape=(lenOf, 2, 98, 257), dtype=np.float32)
    for i in range(lenOf):
        wav = data[i]
        # calculate the specgram.

        x = get_specgram(wav)
        #  print(x.shape)
        datas[i] = x.numpy()  # shape is [bs, time, freq_bins, 2]
    # x = tf.math.to_float(x)
    datas = datas.reshape(lenOf, 98, 2, 257)
    datas = datas.reshape(lenOf, 98, 257, 2)
    return datas


# This function configs the data of labels from a number output ex.4 to an array of [0 0 0 1 0 0 0....0]
def classConfig(data, num_of_classes):
    lenOf = len(data)
    datas = np.ndarray(shape=(lenOf, num_of_classes))
    for i in range(lenOf):
        # print(data[i])
        pos = data[i]
        for k in range(num_of_classes):
            if (pos == k):
                datas[i][k] = 1
            else:
                datas[i][k] = 0
    return datas


def convert_to_audio_dataset(files: List, labels: List, should_squeeze=True):
    """Constructs a dataset of audios and labels."""
    """Constructs a dataset of audios and labels."""
    # path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    # audio_ds = path_ds.map(lambda x: path_to_audio(x))
    # label_ds = tf.data.Dataset.from_tensor_slices(labels)
    # return tf.data.Dataset.zip((audio_ds, label_ds))
    print(f" audio paths type {type(files)}")
    print(f" audio paths type {type(files[0])}")

    # files = tf.convert_to_tensor(files)
    # labels = tf.convert_to_tensor(labels)
    file_dataset = tf.data.Dataset.from_tensor_slices(files)
    audio_dataset = file_dataset.map(lambda x: path2audio(x, should_squeeze))
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_dataset, label_dataset))


def path2audio(path, should_squeeze=True):
    """Reads and decodes an audio file."""
    print(f"path2audio audio path {path}")
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    print(f"path2audio audio shape {audio.shape}")
    if should_squeeze:
        audio = tf.squeeze(audio)
    print(f"path2audio audio shape {audio.shape}")
    return audio


def get_specgram(wav):
    # print(f"wav shape {wav.shape}")
    specgram = tf.signal.stft(
        wav,
        512,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
        128,  # 16000 * 0.010 -- default stride
    )
    # print(f"specgram shape {specgram.shape}")
    # specgram is a complex tensor, so split it into abs and phase parts:
    phase = tf.math.angle(specgram) / np.pi
    # print(f"phase shape {phase.shape}")

    # log(1 + abs) is a default transformation for energy units
    amp = tf.math.log1p(tf.abs(specgram))
    # print(f"amp shape {amp.shape}")
    x = tf.stack([amp, phase], axis=-1)
    # print(f"x shape {x.shape}")
    return x


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    print(f"wav shape {audio.shape}")
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


def shuffle_data(items: List, shuffle_seed: int = 43):
    rng = np.random.RandomState(shuffle_seed)
    rng.shuffle(items)
    return items


def mfccs(pcm, sample_rate=16000):
    # # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
    # pcm = tf.random.normal([batch_size, num_samples], dtype=tf.float32)

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(pcm, frame_length=512, frame_step=128,
                           fft_length=512)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :13]
    return mfccs
