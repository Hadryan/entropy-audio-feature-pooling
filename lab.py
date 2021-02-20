import datetime
import json
import os
import time
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import tensorflow as tf
import numpy as np
from informationPoolSupport import batch_average


information_pool_custom_loss_basic_keras=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def information_pool_custom_loss(y_true, y_pred):
            
     loss=information_pool_custom_loss_basic_keras(y_true,y_pred)
     kl_terms = [ batch_average(kl) for kl in tf.compat.v1.get_collection('kl_terms') ]
     kl_terms=tf.math.add_n(kl_terms)/(257*98*32)
     loss=loss + 0.5*kl_terms
     

        
     return loss
def setup_ops(from_logits=True):
    loss_op = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    #loss_op=information_pool_custom_loss 
    learning_rate = tf.optimizers.schedules.ExponentialDecay(1e-3, 10000, 0.99, staircase=from_logits)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, clipvalue=15)
    train_cross_entr_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)
    val_cross_entr_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)
    test_cross_entr_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    return train_cross_entr_metric, val_cross_entr_metric, test_cross_entr_metric, acc_metric, loss_op, optimizer


def train_step(training_model, inputs, y_batch_train, cross_entr, acc, loss_op, optimizer):
    with tf.GradientTape() as tape:
        logit = training_model(inputs, training=True)
        current_loss = loss_op(y_batch_train, logit)
        added_losses=training_model.losses
        if added_losses :
            current_loss=current_loss + tf.math.add_n(added_losses)
    grads = tape.gradient(current_loss, training_model.trainable_weights)
    grads = [grad if grad is not None else tf.zeros_like(var)
             for var, grad in zip(training_model.trainable_variables, grads)]
    optimizer.apply_gradients(zip(grads, training_model.trainable_weights))
    cross_entr.update_state(y_batch_train, logit)
    acc.update_state(y_batch_train, logit)
    return current_loss
    # print(f"Loss {tf.reduce_mean(current_loss).value}")


def train_epoch(training_model, train_data, cross_entr, acc, loss_op, optimizer, batch_size,
                train_summary_writer, global_step, log_steps=0, tfboard_logsteps=50):
    track_step = 0
    for step, (x_batch_train, y_batch_train) in enumerate(train_data):
        # print(f"x_batch_train {x_batch_train.shape}")
        # print(f"y_batch_train {y_batch_train.shape}")
        current_loss = train_step(training_model, x_batch_train, y_batch_train, cross_entr, acc, loss_op, optimizer)
        if log_steps and step % log_steps == 0:
            tf.print(f"Loss {tf.reduce_mean(current_loss)}")
            print(f"Step {step}. Seen so far: {((step + 1) * batch_size)} samples")

        if tfboard_logsteps and step % tfboard_logsteps == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', cross_entr.result(), step=global_step)
                tf.summary.scalar('accuracy', acc.result(), step=global_step)
        global_step += 1
    return training_model, global_step


class ModelMetadata:
    def __init__(self):
        self.model_save_path = None
        self.epoch = None
        self.global_step = None

        self.val_loss = None
        self.val_acc = None

        self.train_loss = None
        self.train_acc = None

        self.test_loss = None
        self.test_acc = None


def start_training_loop(num_epochs, model, train_dataset, train_cross_entr_metric, acc_metric, loss_op, optimizer,
                        batch_size, val_dataset, val_cross_entr_metric,
                        task_name="default",
                        exp_descr="exp",
                        patience=10,
                        log_pool_layers=False):
    base_logs = f'logs/{task_name}'
    best_val_loss = 10000000000
    current_patience = patience
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_folder = base_logs + f"/checkpoints/{exp_descr}/{current_time}/"
    model_save_path = model_save_folder + "model"
    meta_data_save_path = model_save_folder + "metadata.json"
    model_metadata = ModelMetadata()
    model_metadata.model_save_path = model_save_path

    train_log_dir = base_logs + f'/tensorboard/{exp_descr}/' + current_time + '/train'
    val_log_dir = base_logs + f'/tensorboard/{exp_descr}/' + current_time + '/val'
    metadata_log_dir = base_logs + f'/tensorboard/{exp_descr}/' + current_time + '/metadata'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    metadata_summary_writer = tf.summary.create_file_writer(metadata_log_dir)
    poolinfo_log_dir = base_logs + f'/tensorboard/{exp_descr}/' + current_time + '/poolinfo'
    poolinfo_summary_writer = None
    if log_pool_layers:
        poolinfo_summary_writer = tf.summary.create_file_writer(poolinfo_log_dir)
        model.log_pool_info = True

    global_step = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        _, global_step = train_epoch(model, train_dataset,
                                     train_cross_entr_metric, acc_metric,
                                     loss_op, optimizer,
                                     batch_size,
                                     train_summary_writer, global_step)

        # Display metrics at the end of each epoch.
        train_cross_entr = train_cross_entr_metric.result()
        train_accuracy = acc_metric.result()

        # Reset training metrics at the end of each epoch
        train_cross_entr_metric.reset_states()
        acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_cross_entr_metric.update_state(y_batch_val, val_logits)
            acc_metric.update_state(y_batch_val, val_logits)
        val_cross_entr = val_cross_entr_metric.result()
        val_cross_entr_metric.reset_states()
        val_acc = acc_metric.result()
        acc_metric.reset_states()

        if val_cross_entr < best_val_loss:
            best_val_loss = val_cross_entr
            model_metadata.epoch = epoch
            model_metadata.global_step = global_step
            model_metadata.val_loss = str(val_cross_entr.numpy())
            model_metadata.val_acc = str(val_acc.numpy())
            model_metadata.train_loss = str(train_cross_entr.numpy())
            model_metadata.train_acc = str(train_accuracy.numpy())
            variance = np.abs(float(model_metadata.train_loss) - float(model_metadata.val_loss))
            latest_model_path = model_save_path + f"_e_{epoch}_bias_{round(float(train_cross_entr), 3)}_" \
                                                  f"l_{round(float(val_cross_entr), 3)}_var_{round(variance, 3)}"
            model_metadata.model_save_path = latest_model_path

            current_patience = patience
            model.save_weights(latest_model_path)
        else:
            current_patience = current_patience - 1

        print(f"|Epoch {epoch}/{num_epochs} - duration: {round(float(time.time() - start_time), 3)}s| - "
              f"|Training cross entr: {round(float(train_cross_entr), 3)}, train_acc: {round(float(train_accuracy), 3)}| "
              f"|Validation cross entr: {round(float(val_cross_entr), 3)} val_acc: {round(float(val_acc), 3)}| "
              f"|patience:{current_patience}|")
        with metadata_summary_writer.as_default():
            tf.summary.scalar('epoch', epoch, step=global_step)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_cross_entr, step=global_step)
            tf.summary.scalar('accuracy', train_accuracy, step=global_step)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_cross_entr, step=global_step)
            tf.summary.scalar('accuracy', val_acc, step=global_step)

        if log_pool_layers:
            with poolinfo_summary_writer.as_default():
                i = 0
                for layer in model.pool_info:
                    i+=1
                    tf.summary.histogram(f"pool{i}_{layer[0]}_in", layer[1], step=epoch)
                    tf.summary.histogram(f"pool{i}_{layer[0]}_out", layer[2], step=epoch)

        if current_patience < 1:
            break
        with open(os.path.join(meta_data_save_path), 'w') as f:
            json.dump(model_metadata.__dict__, f)
    return model_metadata, model_save_folder


def start_testing_loop(test_dataset, model, test_cross_entr_metric, acc_metric, test_predictions, labels_test,
                       model_metadata=None, model_save_folder=None):
    for x_batch_test, y_batch_test in test_dataset:
        test_logits = model(x_batch_test, training=False)
        test_cross_entr_metric.update_state(y_batch_test, test_logits)
        acc_metric.update_state(y_batch_test, test_logits)
        y_pred = np.argmax(test_logits, axis=1)
        test_predictions.extend(y_pred)
        labels_test.extend(y_batch_test)

    test_cross_entr = test_cross_entr_metric.result()
    test_acc = acc_metric.result()

    if model_metadata and model_save_folder:
        metadata_save_path = model_save_folder + "metadata.json"
        model_metadata.test_loss = str(test_cross_entr.numpy())
        model_metadata.test_acc = str(test_acc.numpy())
        with open(os.path.join(metadata_save_path), 'w') as f:
            if isinstance(model_metadata, ModelMetadata):
                json.dump(model_metadata.__dict__, f)
            else:
                json.dump(model_metadata, f)

    print(f"Test cross entr: {round(float(test_cross_entr), 3)} and acc: {round(float(test_acc), 3)}")


def calculate_confusion_matrix(labels, labels_test, test_predictions, model_save_folder=None):
    print('Confusion Matrix')
    cf = confusion_matrix(labels_test, test_predictions)
    print(cf)
    print('Classification Report')
    print(classification_report(labels_test, test_predictions, target_names=labels))
    df_cm = pd.DataFrame(cf, labels, labels)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True)  # font size
    if model_save_folder:
        plt.savefig(model_save_folder + "confusion.png")
