"""
Title: Speaker Recognition
Author: [Fadi Badine](https://twitter.com/fadibadine)
Date created: 14/06/2020
Last modified: 03/07/2020
Description: Classify speakers using Fast Fourier Transform (FFT) and a 1D Convnet.
"""

import lab
from models import ResidualModel, PoolingLayerFactory

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
- Other resources 
https://www.tensorflow.org/tutorials/audio/simple_audio
https://www.tensorflow.org/io/tutorials/audio
"""

"""
## Setup
"""

import os
import shutil
import numpy as np

import tensorflow as tf
import datatools as dt
import noisykit as noisy
from pathlib import Path
from IPython.display import display, Audio

# Get the data from https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
# and save it to the 'Downloads' folder in your HOME directory
DATASET_ROOT = os.path.join(os.path.expanduser("~"), "Downloads/16000_pcm_speeches")
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"
DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)
VALID_SPLIT = 0.1
SHUFFLE_SEED = 43
SAMPLING_RATE = 16000

SCALE = 0.25 #0.5
BATCH_SIZE = 32
EPOCHS = 100
TRAIN_NOISE = False
VAL_NOISE = False
TEST_NOISE = True
NOISE = 'uniform'
TASK_NAME = f"speaker_noise_{NOISE}_scale_{SCALE}_train_{str(TRAIN_NOISE)[0]}_val_{str(VAL_NOISE)[0]}_test_{str(TEST_NOISE)[0]}"

PATIENCE = 3


def run_exp(train_noise, val_noise, test_noise, last_pool, resblock_pool,
            batch_size=BATCH_SIZE, epochs=EPOCHS, task_name=TASK_NAME, patience=PATIENCE, scale=SCALE, noise_type=NOISE):
    global noises
    if os.path.exists(DATASET_AUDIO_PATH) is False:
        os.makedirs(DATASET_AUDIO_PATH)
    if os.path.exists(DATASET_NOISE_PATH) is False:
        os.makedirs(DATASET_NOISE_PATH)
    for folder in os.listdir(DATASET_ROOT):
        if os.path.isdir(os.path.join(DATASET_ROOT, folder)):
            if folder in [AUDIO_SUBFOLDER, NOISE_SUBFOLDER]:
                continue
            elif folder in ["other", "_background_noise_"]:
                shutil.move(
                    os.path.join(DATASET_ROOT, folder),
                    os.path.join(DATASET_NOISE_PATH, folder),
                )
            else:
                shutil.move(
                    os.path.join(DATASET_ROOT, folder),
                    os.path.join(DATASET_AUDIO_PATH, folder),
                )
    class_names = os.listdir(DATASET_AUDIO_PATH)
    print("Our class names: {}".format(class_names, ))
    audio_paths = []
    labels = []
    for label, name in enumerate(class_names):
        print("Processing speaker {}".format(name, ))
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
    noise_paths = noisy.get_list_of_noise_paths()
    noisy.resample_noise_samples()
    noises = noisy.get_noises(noise_paths)
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(audio_paths)
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(labels)
    num_val_samples = int(VALID_SPLIT * len(audio_paths))
    print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
    train_audio_paths = audio_paths[:-2 * num_val_samples]
    train_labels = labels[:-2 * num_val_samples]
    print("Using {} files for validation.".format(num_val_samples))
    valid_audio_paths = audio_paths[-2 * num_val_samples:-num_val_samples]
    valid_labels = labels[-2 * num_val_samples:-num_val_samples]
    test_audio_paths = audio_paths[-num_val_samples:]
    test_labels = labels[-num_val_samples:]
    train_ds = dt.convert_to_audio_dataset(train_audio_paths, train_labels, False)
    train_ds = train_ds.shuffle(buffer_size=batch_size * 8, seed=SHUFFLE_SEED).batch(batch_size)
    valid_ds = dt.convert_to_audio_dataset(valid_audio_paths, valid_labels, False)
    valid_ds = valid_ds.shuffle(buffer_size=batch_size * 8, seed=SHUFFLE_SEED).batch(batch_size)
    test_ds = dt.convert_to_audio_dataset(test_audio_paths, test_labels, False)
    test_ds = test_ds.shuffle(buffer_size=batch_size * 8, seed=SHUFFLE_SEED).batch(batch_size)
    if train_noise:
        train_ds = train_ds.map(lambda x, y: (noisy.add_noise(x, noises, scale=scale, should_squeeze=False), y),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE, )
    if val_noise:
        valid_ds = valid_ds.map(lambda x, y: (noisy.add_noise(x, noises, scale=scale, should_squeeze=False), y),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE, )
    if test_noise:
        test_ds = test_ds.map(lambda x, y: (noisy.add_noise(x, noises, scale=scale, dist=noise_type, should_squeeze=False), y),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE, )
    train_ds = train_ds.map(lambda x, y: (dt.audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.map(lambda x, y: (dt.audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (dt.audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    model = ResidualModel(class_names, last_pool=last_pool, resblock_pool=resblock_pool)
    d = SAMPLING_RATE // 2
    dim = (d, 1)
    model.build((batch_size, *dim))
    print(f"model name {model.name}")
    # model.summary()
    train_cross_entr_metric, val_cross_entr_metric, test_cross_entr_metric, acc_metric, loss_op, optimizer \
        = lab.setup_ops(from_logits=False)
    model_metadata, model_save_folder = lab.start_training_loop(epochs, model, train_ds, train_cross_entr_metric,
                                                                acc_metric,
                                                                loss_op, optimizer,
                                                                batch_size, valid_ds, val_cross_entr_metric,
                                                                task_name=task_name,
                                                                exp_descr=f"residual_model_"
                                                                          f"last_{last_pool}"
                                                                          f"_res_{resblock_pool}",
                                                                patience=patience)
    test_predictions = []
    labels_test = []
    model = ResidualModel(class_names, last_pool=last_pool, resblock_pool=resblock_pool)
    model.load_weights(model_metadata.model_save_path)
    lab.start_testing_loop(test_ds, model, test_cross_entr_metric, acc_metric, test_predictions, labels_test,
                           model_metadata, model_save_folder)
    lab.calculate_confusion_matrix(class_names, labels_test, test_predictions, model_save_folder)


poolings = [PoolingLayerFactory.MAX,PoolingLayerFactory.INFO]

for last in poolings:
    for res in poolings:
        for i in range(10):
            print(f"{TASK_NAME}")
            print(f"Running with last pool {last}, resblock pool {res}, time {i}.")
            run_exp(TRAIN_NOISE, VAL_NOISE, TEST_NOISE, last, res,
                    batch_size=BATCH_SIZE, epochs=EPOCHS, task_name=TASK_NAME,
                    patience=PATIENCE, scale=SCALE, noise_type=NOISE)

"""
We get ~ 98% validation accuracy.
"""

"""
## Demonstration
Let's take some samples and:
- Predict the speaker
- Compare the prediction with the real speaker
- Listen to the audio to see that despite the samples being noisy,
the model is still pretty accurate
"""

# SAMPLES_TO_DISPLAY = 10

# test_ds = dt.convert_to_audio_dataset(valid_audio_paths, valid_labels, False)
# test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
#     BATCH_SIZE
# )

# test_ds = test_ds.map(lambda x, y: (noisy.add_noise(x, noises, scale=SCALE, should_squeeze=False), y))
#
# for audios, labels in test_ds.take(1):
#     # Get the signal FFT
#     ffts = dt.audio_to_fft(audios)
#     # Predict
#     y_pred = model.predict(ffts)
#     # Take random samples
#     rnd = np.random.randint(0, BATCH_SIZE, SAMPLES_TO_DISPLAY)
#     audios = audios.numpy()[rnd, :, :]
#     labels = labels.numpy()[rnd]
#     y_pred = np.argmax(y_pred, axis=-1)[rnd]
#
#     for index in range(SAMPLES_TO_DISPLAY):
#         # For every sample, print the true and predicted label
#         # as well as run the voice with the noise
#         print(
#             "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
#                 "[92m" if labels[index] == y_pred[index] else "[91m",
#                 class_names[labels[index]],
#                 "[92m" if labels[index] == y_pred[index] else "[91m",
#                 class_names[y_pred[index]],
#             )
#         )
#         display(Audio(audios[index, :, :].squeeze(), rate=SAMPLING_RATE))
