import tensorflow as tf

import datatools as dt
import lab
import noisykit
from models import AlexNet, PoolingLayerFactory

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
BATCH_SIZE = 32
PREFETCH = 10 * BATCH_SIZE
SHUFFLE_SEED = 43
SCALE = 2
TRAIN_NOISE = False
VAL_NOISE = False
TEST_NOISE = False
TASK_NAME = f"commands_noise_train_{str(TRAIN_NOISE)[0]}_val_{str(VAL_NOISE)[0]}_test_{str(TEST_NOISE)[0]}"
POOLING_OPS = [PoolingLayerFactory.INFO,
               PoolingLayerFactory.INFO,
               PoolingLayerFactory.INFO,
               PoolingLayerFactory.INFO]
EPOCHS = 3


def prepare_data(batch_size, train_noise, val_noise, test_noise):
    labels = 'yes no up down left right on off stop go silence unknown'.split()
    train_files_labels, val_files_labels, test_files_labels = dt.get_filenames_of_train_val_test_sets('', labels)
    train_files = dt.shuffle_data(train_files_labels[0])
    train_labels = dt.shuffle_data(train_files_labels[1])
    train_dataset = dt.convert_to_audio_dataset(train_files, train_labels)
    train_dataset = train_dataset.shuffle(buffer_size=batch_size * 10, seed=SHUFFLE_SEED).batch(batch_size)
    val_dataset = dt.convert_to_audio_dataset(val_files_labels[0], val_files_labels[1])
    val_dataset = val_dataset.shuffle(buffer_size=batch_size * 10, seed=SHUFFLE_SEED).batch(batch_size)
    test_dataset = dt.convert_to_audio_dataset(test_files_labels[0], test_files_labels[1])
    test_dataset = test_dataset.shuffle(buffer_size=batch_size * 10, seed=SHUFFLE_SEED).batch(batch_size)
    noise_files = []
    if train_noise or val_noise or test_noise:
        noise_paths = noisykit.get_list_of_noise_paths()
        noisykit.resample_noise_samples()
        noise_files = noisykit.get_noises(noise_paths)
    if train_noise and noise_files:
        train_dataset = train_dataset.map(
            lambda x, y: (noisykit.add_noise(x, noise_files, scale=SCALE), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    if val_noise and noise_files:
        val_dataset = val_dataset.map(
            lambda x, y: (noisykit.add_noise(x, noise_files, scale=SCALE), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    if test_noise and noise_files:
        test_dataset = test_dataset.map(
            lambda x, y: (noisykit.add_noise(x, noise_files, scale=SCALE), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    # Transform audio wave to the frequency domain using `audio_to_fft`
    train_dataset = train_dataset.map(lambda x, y: (dt.get_specgram(x), y),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.prefetch(PREFETCH)
    val_dataset = val_dataset.map(lambda x, y: (dt.get_specgram(x), y),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_ds = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(lambda x, y: (dt.get_specgram(x), y),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return labels, train_dataset, val_dataset, test_dataset, noise_files


labels, train_dataset, val_dataset, test_dataset, noise_files = \
    prepare_data(BATCH_SIZE, TRAIN_NOISE, VAL_NOISE, TEST_NOISE)

# model parameters2
# leaky_relu_alpha = 0.2
# dropout_rate = 0.5


# def loss(pred, target):
#     return tf.losses.sparse_categorical_crossentropy(target, pred, from_logits=True)
#     # return tf.losses.kullback_leibler_divergence( target , pred )


train_cross_entr_metric, val_cross_entr_metric, test_cross_entr_metric, acc_metric, loss_op, optimizer \
    = lab.setup_ops()


model = AlexNet(labels, types_of_poolings=POOLING_OPS, ksizes=None)
dim = (122, 257, 2)
model.build((BATCH_SIZE, *dim))
print(f"Class name: {model.name}")
# tf.keras.utils.plot_model(model.build_graph(), to_file="alexnetstyle.png", show_shapes=True, show_layer_names=False)
# print("Model saved as png!")
model_metadata, model_save_folder = lab.start_training_loop(EPOCHS, model, train_dataset, train_cross_entr_metric,
                                                            acc_metric, loss_op,
                                                            optimizer,
                                                            BATCH_SIZE, val_dataset, val_cross_entr_metric,
                                                            task_name=TASK_NAME,
                                                            exp_descr=f"{model.name}_"
                                                                      f"{''.join([str(p[0]) for p in POOLING_OPS])}",
                                                            patience=3)

test_predictions = []
labels_test = []
model = AlexNet(labels, types_of_poolings=POOLING_OPS, ksizes=None)
model.load_weights(model_metadata.model_save_path)
lab.start_testing_loop(test_dataset, model, test_cross_entr_metric, acc_metric, test_predictions, labels_test,
                       model_metadata, model_save_folder)
lab.calculate_confusion_matrix(labels[:-1], labels_test, test_predictions, model_save_folder)
