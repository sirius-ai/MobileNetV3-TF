# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
MobileNetV3 training code.
Author: aiboy.wei@outlook.com .
'''

from MobileNetV3 import MobileNetV3
from datetime import datetime
import tensorflow as tf
import numpy as np
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--train_phase', type=bool, default=True, help='train phase, true or false!')
    parser.add_argument('--model_type', type=str, default="large", help='model type, choice large or small!')
    parser.add_argument('--max_epoch', default=4, help='max epoch to train the network！')
    parser.add_argument('--input_shape', default=(224, 224, 3), help='the input size！')
    parser.add_argument('--classes_number', type=int, required=True, help='class number depend on your training datasets！')
    parser.add_argument('--weight_decay', default=2e-4, help='L2 weight regularization.')
    parser.add_argument('--lr_schedule', help='Number of epochs for learning rate piecewise.', default=[1, 2, 3])
    parser.add_argument('--train_batch_size', default=32, help='batch size of training.')
    parser.add_argument('--test_batch_size', default=32, help='batch size of testing.')
    parser.add_argument('--train_tfrecords_file_path', default='./data/train.tfrecords', type=str,
                        help='path to the training datasets of tfrecords file path')
    parser.add_argument('--test_tfrecords_file_path', default='./data/test.tfrecords', type=str,
                        help='path to the testing datasets of tfrecords file path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--ckpt_best_path', default='./output/ckpt_best', help='the best ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the log file save path')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--ckpt_interval', default=1000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=1000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=50, help='intervals to save ckpt file')
    parser.add_argument('--pretrained_model', type=str, default='', help='Load a pretrained model before training starts.')
    parser.add_argument('--dropout_rate', type=float, help='dropout rate', default=0.2)

    args = parser.parse_args()

    return args


def evaluation(log_dir, datasets, model):
    count = 0
    total_predict = []
    def batch_evaluation(pred, labels):
        correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
        return list(correct_prediction.numpy())

    for i, (images, labels) in enumerate(datasets):
        logits = model(images, training=False)
        pred = tf.nn.softmax(logits)
        batch_correct_prediction = batch_evaluation(pred, labels)
        total_predict.extend(batch_correct_prediction)
        count += len(labels)

    total_predict = np.asarray(total_predict)
    Accuracy = tf.reduce_mean(total_predict)
    print(f'test total images {count}, Accuracy is {Accuracy}!')

    with open(os.path.join(log_dir, 'result.txt'), 'at') as f:
        f.write(f'test total images {count}, Accuracy is {Accuracy}!\n')


@tf.function
def train_parse_function(example_proto):
    features = {'rgb_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['rgb_raw'])
    img = tf.reshape(img, (224, 224, 3))
    h, w, c = img.shape
    if h != 224 or w != 224 or c != 3:
        assert 0, "Assert! Input image shape should be (224, 224, 3)!!!"
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img,  0.0078125)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int64)
    return img, label

@tf.function
def test_parse_function(example_proto):
    features = {'rgb_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['rgb_raw'])
    img = tf.reshape(img, (224, 224, 3))
    h, w, c = img.shape
    if h != 224 or w != 224 or c != 3:
        assert 0, "Assert! Input image shape should be (224, 224, 3)!!!"
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img,  0.0078125)
    label = tf.cast(features['label'], tf.int64)
    return img, label


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = get_parser()

    # create log dir
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.log_file_path), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    with open(os.path.join(log_dir, 'result.txt'), 'at') as f:
        f.write('%s\t%s\t%s\n' % ("step", "loss", "Accuracy"))

    # fix cudnn error, if you use gpu device
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.compat.v2.config.experimental.set_memory_growth(gpu, True)\

    # training datasets pipe
    train_tfrecords_f = os.path.join(args.train_tfrecords_file_path)
    train_dataset = tf.data.TFRecordDataset(train_tfrecords_f)
    train_dataset = train_dataset.map(train_parse_function)
    # dataset = dataset.shuffle(buffer_size=args.buffer_size)
    train_dataset = train_dataset.batch(args.train_batch_size)
    # testing datasets pipe
    test_tfrecords_f = os.path.join(args.test_tfrecords_file_path)
    test_dataset = tf.data.TFRecordDataset(test_tfrecords_f)
    test_dataset = test_dataset.map(test_parse_function)
    test_dataset = test_dataset.batch(args.test_batch_size)

    # learning rate schedule
    epoch_var = tf.Variable(1, trainable=False)
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=args.lr_schedule,
                                                                            values=[0.005, 0.002, 0.001, 0.0005],
                                                                            name='lr_schedule')
    lr = learning_rate_fn(epoch_var)

    # Instantiate an optimizer, special change optimizers in here if need.
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    # Instantiate a loss function， customer loss function can insert in here.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model = MobileNetV3(type=args.model_type, input_shape=args.input_shape, classes_number=args.classes_number,
                        l2_reg=args.weight_decay, dropout_rate=args.dropout_rate, name="MobileNetV3")
    tf.keras.backend.set_learning_phase(True)
    # model architecture write to file
    fd = open(f'./misc/MobileNetV3_{args.model_type}.txt', "w")
    for var in model.variables:
        info = f'<tf.Variable \'{var.name}\'' + f' shape={var.shape}' + f' dtype={var.numpy().dtype}>'
        fd.write(info + "\n")
    fd.close()

    # load model weights
    if args.pretrained_model:
        model.load(args.pretrained_model)
        # model.load_weights(args.pretrained_model)
        print(f'Successful to load pretrained model!')

    step = 0
    for e in range(args.max_epoch):
        epoch_var.assign(e)
        lr = learning_rate_fn(epoch_var)
        optimizer.learning_rate = lr # update learning rate
        for i, (images, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(images, training=args.train_phase)
                regularization_loss = tf.math.add_n(model.losses)
                # logits = tf.nn.l2_normalize(logits, 1, 1e-10, name='logits')
                pred = tf.nn.softmax(logits)
                pred_loss = loss_fn(labels, pred)
                loss_value = pred_loss + regularization_loss

            trainable_variables = model.trainable_variables
            grads = tape.gradient(loss_value, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))

            step += 1
            if step % args.show_info_interval == 0:
                # calculate accuracy
                pred = tf.nn.softmax(logits)
                correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
                Accuracy = tf.reduce_mean(correct_prediction)
                print(f'epoch {e}, lr {lr}, total_step {step}, loss {loss_value}, Accuracy {Accuracy}')
                with open(os.path.join(log_dir, 'result.txt'), 'at') as f:
                    f.write('%d\t%2.4f\t%2.4f\n' % (step, loss_value, Accuracy))

            if step % args.ckpt_interval == 0:
                # ckpt_path = os.path.join(args.ckpt_path, f'./checkpoints/MobileNetV3_{args.model_type}_{step}')
                # model.save_weights(ckpt_path, save_format='tf')
                ckpt_path = os.path.join(args.ckpt_path, f'MobileNetV3_{args.model_type}_{step}.h5')
                model.save(ckpt_path)

            if step % args.validate_interval == 0:
                evaluation(log_dir, test_dataset, model)

    # save finnal parameters
    ckpt_path = os.path.join(args.ckpt_path, f'MobileNetV3_final.h5')
    model.save(ckpt_path)
    # final test
    evaluation(log_dir, test_dataset, model)
