# import tensorflow.python.keras as keras
import pickle

# from levy import fit_levy
from scipy import stats
# from tensorflow.python.keras.utils import plot_model, multi_gpu_model
from sklearn.metrics import classification_report
from tensorflow.python.keras import callbacks, regularizers
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dropout, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.vis_utils import plot_model

from components import EntropyPooling2D

"""
Title: Speaker Recognition
Author: [Fadi Badine](https://twitter.com/fadibadine)
Date created: 14/06/2020
Last modified: 03/07/2020
Description: Classify speakers using Fast Fourier Transform (FFT) and a 1D Convnet.
"""

"""
## Introduction
This example demonstrates how to create a model to classify speakers from the
frequency domain representation of speech recordings, obtained via Fast Fourier
Transform (FFT).
It shows the following:
- How to use `tf.data` to load, preprocess and feed audio streams into a model
- How to create a 1D convolutional network with residual
connections for audio classification.
Our process:
- We prepare a dataset of speech samples from different speakers, with the speaker as label.
- We add background noise to these samples to augment our data.
- We take the FFT of these samples.
- We train a 1D convnet to predict the correct speaker given a noisy FFT speech sample.
Note:
- This example should be run with TensorFlow 2.3 or higher, or `tf-nightly`.
- The noise samples in the dataset need to be resampled to a sampling rate of 16000 Hz
before using the code in this example. In order to do this, you will need to have
installed `ffmpg`.
"""

"""
## Setup
"""

import os
import shutil
import numpy as np

import tensorflow as tf

from pathlib import Path

# Get the data from https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
# and save it to the 'Downloads' folder in your HOME directory
DATASET_ROOT = os.path.join(os.path.expanduser("~"), "Downloads/16000_pcm_speeches")

# The folders in which we will put the audio samples and the noise samples
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

# Percentage of samples to use for validation
VALID_SPLIT = 0.1

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43

# The sampling rate to use.
# This is the one used in all of the audio samples.
# We will resample all of the noise to this sampling rate.
# This will also be the output size of the audio wave samples
# (since all samples are of 1 second long)
SAMPLING_RATE = 16000

# The factor to multiply the noise with according to:
#   noisy_sample = sample + noise * prop * scale
#      where prop = sample_amplitude / noise_amplitude
SCALE = 0.5

BATCH_SIZE = 32
EPOCHS = 100

# If folder `audio`, does not exist, create it, otherwise do nothing
if os.path.exists(DATASET_AUDIO_PATH) is False:
    os.makedirs(DATASET_AUDIO_PATH)

# If folder `noise`, does not exist, create it, otherwise do nothing
if os.path.exists(DATASET_NOISE_PATH) is False:
    os.makedirs(DATASET_NOISE_PATH)

for folder in os.listdir(DATASET_ROOT):
    if os.path.isdir(os.path.join(DATASET_ROOT, folder)):
        if folder in [AUDIO_SUBFOLDER, NOISE_SUBFOLDER]:
            # If folder is `audio` or `noise`, do nothing
            continue
        elif folder in ["other", "_background_noise_"]:
            # If folder is one of the folders that contains noise samples,
            # move it to the `noise` folder
            shutil.move(
                os.path.join(DATASET_ROOT, folder),
                os.path.join(DATASET_NOISE_PATH, folder),
            )
        else:
            # Otherwise, it should be a speaker folder, then move it to
            # `audio` folder
            shutil.move(
                os.path.join(DATASET_ROOT, folder),
                os.path.join(DATASET_AUDIO_PATH, folder),
            )


# Get the list of all noise files
noise_paths = []
for subdir in os.listdir(DATASET_NOISE_PATH):
    subdir_path = Path(DATASET_NOISE_PATH) / subdir
    if os.path.isdir(subdir_path):
        noise_paths += [
            os.path.join(subdir_path, filepath)
            for filepath in os.listdir(subdir_path)
            if filepath.endswith(".wav")
        ]

print(
    "Found {} files belonging to {} directories".format(
        len(noise_paths), len(os.listdir(DATASET_NOISE_PATH))
    )
)

"""
Resample all noise samples to 16000 Hz
"""

command = (
    "for dir in `ls -1 " + DATASET_NOISE_PATH + "`; do "
    "for file in `ls -1 " + DATASET_NOISE_PATH + "/$dir/*.wav`; do "
    "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
    "$file | grep sample_rate | cut -f2 -d=`; "
    "if [ $sample_rate -ne 16000 ]; then "
    "ffmpeg -hide_banner -loglevel panic -y "
    "-i $file -ar 16000 temp.wav; "
    "mv temp.wav $file; "
    "fi; done; done"
)
os.system(command)

# Split noise into chunks of 16,000 steps each
def load_noise_sample(path):
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == SAMPLING_RATE:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(sample.shape[0] / SAMPLING_RATE)
        sample = tf.split(sample[: slices * SAMPLING_RATE], slices)
        return sample
    else:
        print("Sampling rate for {} is incorrect. Ignoring it".format(path))
        return None


noises = []
for path in noise_paths:
    sample = load_noise_sample(path)
    if sample:
        noises.extend(sample)
noises = tf.stack(noises)

print(
    "{} noise files were split into {} noise samples where each is {} sec. long".format(
        len(noise_paths), noises.shape[0], noises.shape[1] // SAMPLING_RATE
    )
)

"""
## Dataset generation
"""


def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale

    return audio


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


# Get the list of audio file paths along with their corresponding labels

class_names = os.listdir(DATASET_AUDIO_PATH)
print("Our class names: {}".format(class_names,))

audio_paths = []
labels = []
for label, name in enumerate(class_names):
    print("Processing speaker {}".format(name,))
    dir_path = Path(DATASET_AUDIO_PATH) / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)

print(
    "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
)

# Shuffle
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

# Split into training and validation
num_val_samples = int(VALID_SPLIT * len(audio_paths))
print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]

print("Using {} files for validation.".format(num_val_samples))
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

# Create 2 datasets, one for training and the other for validation
train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
    BATCH_SIZE
)

valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)


# Add noise to the training set
train_ds = train_ds.map(lambda x, y: (add_noise(x, noises, scale=SCALE), y), num_parallel_calls=tf.data.experimental.AUTOTUNE,)

# Transform audio wave to the frequency domain using `audio_to_fft`
train_ds = train_ds.map(lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

valid_ds = valid_ds.map(lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

"""
## Model Definition
"""

def train(x_train, y_train, x_test, y_test, epochs):

    #  calculate classes
    if np.unique(y_train).shape[0] == np.unique(y_test).shape[0]:
        #
        num_classes = np.unique(y_train).shape[0]
    else:
        print('Error in class data...')
        return -2

    # set validation data
    '''val_size = int(0.1 * x_train.shape[0])
    r = np.random.randint(0, x_train.shape[0], size=val_size)
    x_val = x_train[r, :, :]
    y_val = y_train[r]
    x_train = np.delete(x_train, r, axis=0)
    y_train = np.delete(y_train, r, axis=0)'''
    step = int(x_train.shape[0]*0.005)
    length = int(x_train.shape[0]*0.1*0.005)
    r = []
    for i in range(0, x_train.shape[0]-length, step):   r.extend(range(i, i+length))
    x_val = x_train[r, :, :]
    y_val = y_train[r]
    x_train = np.delete(x_train, r, axis=0)
    y_train = np.delete(y_train, r, axis=0)

    print('\nInitializing CNN2D...')
    print('\nclasses:', num_classes)
    print('x train shape:', x_train.shape), print('x val shape:', x_val.shape), print('x test shape:', x_test.shape)
    print('y train shape:', y_train.shape), print('y val shape:', y_val.shape), print('y test shape:', y_test.shape)
    print("\nTrain split with mean|std {:.2f}|{:.2f}".format(np.mean(x_train), np.std(x_train)))
    print("Test split with mean|std {:.2f}|{:.2f}".format(np.mean(x_test), np.std(x_test)))

    # shape data
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # define the model
    activation = 'elu'
    regularizer = 0.0000
    dropout = 0.25

    # preprocessing
    '''
    offset = 1.0 * np.std(x_train)
    dc0 = (x)
    dc1 = GaussianNoise(offset*0.1)(x)
    dc2 = GaussianDropout(dropout)(x)
    dc3 = Lambda(lambda r: r + __import__('keras').backend.random_uniform((1,), -offset, offset))(x)
    dc4 = Lambda(lambda r: r + __import__('keras').backend.random_uniform((1,), -offset, offset))(x)
    m = Concatenate()([dc0, dc1, dc2, dc3, dc4])
    m = Lambda(lambda r: r - __import__('keras').backend.mean(r))(x)
    '''

    # sequential

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 1), activation='elu', kernel_regularizer=regularizers.l2(regularizer), input_shape=(x_train.shape[1], x_train.shape[2], 1)))
    model.add(EntropyPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='elu', kernel_regularizer=regularizers.l2(regularizer)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='elu', kernel_regularizer=regularizers.l2(regularizer)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    # model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='elu', kernel_regularizer=regularizers.l2(regularizer)))
    # model.add(MaxPooling2D(pool_size=(1, 2)))
    # model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(64, activation='elu', kernel_regularizer=regularizers.l2(regularizer)))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    # functional
    '''
    x = Input((x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    m = Conv2D(16, 3, activation=activation , kernel_regularizer=regularizers.l2(regularizer))(x)
    m = EntropyPooling2D((2, 2))(m)
    m = Dropout(dropout)(m)
    m = Conv2D(32, 3, activation=activation, kernel_regularizer=regularizers.l2(regularizer))(m)
    m = EntropyPooling2D((2, 2))(m)
    m = Dropout(dropout)(m)
    m = Conv2D(64, 3, activation=activation, kernel_regularizer=regularizers.l2(regularizer))(m)
    m = EntropyPooling2D((2, 2))(m)
    m = Dropout(dropout)(m)
    if x_train.shape[1] < 50:
        #
        m = Flatten()(m)
    else:
        m = Conv2D(128, 3, activation=activation, kernel_regularizer=regularizers.l2(regularizer))(m)
        m = GlobalAveragePooling2D()(m)
        m = Dropout(dropout)(m)
    m = (Dense(64, activation=activation, kernel_regularizer=regularizers.l2(regularizer)))(m)
    m = Dropout(dropout)(m)
    y = Dense(num_classes, activation='softmax')(m)
    model = Model(inputs=[x], outputs=[y])
    '''

    # summarize model
    for i in range(0, len(model.layers)):
        if i == 0:
            plot_model(model, to_file='Models\\model_cnn2d.png')
            # f = open('Models\\model_cnn2d.txt', 'w')
            # print(' ')
        # print('{}. Layer {} with input / output shapes: {} / {}'.format(i, model.layers[i].name, model.layers[i].input_shape, model.layers[i].output_shape))
        # f.write('{}. Layer {} with input / output shapes: {} / {} \n'.format(i, model.layers[i].name, model.layers[i].input_shape, model.layers[i].output_shape))
        if i == len(model.layers) - 1:
            # f.close()
            print(' ')
            model.summary()

    # compile, fit evaluate
    callback = [callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=10, restore_best_weights=True)]
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=256, epochs=epochs, verbose=2, validation_data=(x_val, y_val), callbacks=callback)
    score = model.evaluate(x_test, y_test, verbose=2)

    # evaluate on larger frames
    aggr_size = 5
    for i in range(0, y_test.shape[0] - aggr_size, aggr_size):
        if i == 0:
            y_pred = model.predict(x_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_test = np.argmax(y_test, axis=1)
            y_aggr_test = []
            y_aggr_pred = []
        if np.unique(y_test[i:i + aggr_size]).shape[0] == 1:
            y_aggr_test.append(stats.mode(y_test[i:i + aggr_size])[0][0])
            y_aggr_pred.append(stats.mode(y_pred[i:i + aggr_size])[0][0])
    # print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
    scipy_score = classification_report(y_aggr_test, y_aggr_pred, output_dict=True)['accuracy']
    print('short {:.2f} and aggr {:.2f}'.format(score[1], scipy_score))

    # save model
    open("Models\\model_cnn2d.json", "w").write(model.to_json())
    pickle.dump(model.get_config(), open("Models\\model_cnn2d.pickle", "wb"))
    model.save_weights("Models\\model_cnn2d.h5")

    # results
    return score[1]


for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
    train(x_batch_train, y_batch_train, x_batch_train, y_batch_train, 5)
