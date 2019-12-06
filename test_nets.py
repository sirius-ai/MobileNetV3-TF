# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
MobileNetV3 testing code.
Author: aiboy.wei@outlook.com .
'''

from MobileNetV3 import MobileNetV3, custom_objects
import tensorflow as tf
import numpy as np
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to test net')
    parser.add_argument('--train_phase', type=bool, default=False, help='train phase, true or false!')
    parser.add_argument('--model_type', type=str, default="large", help='model type, choice large or small!')
    parser.add_argument('--input_shape', default=(224, 224, 3), help='the input size！')
    parser.add_argument('--classes_number', type=int, help='class number depend on your training datasets！')
    parser.add_argument('--weight_decay', default=0., help='L2 weight regularization.')
    parser.add_argument('--test_batch_size', default=32, help='batch size of testing.')
    parser.add_argument('--test_tfrecords_file_path', default='./data/test.tfrecords', type=str,
                        help='path to the testing datasets of tfrecords file path')
    parser.add_argument('--pretrained_model', type=str, default='', help='Load a pretrained model before training starts.')
    parser.add_argument('--dropout_rate', type=float, help='dropout rate', default=0.)

    args = parser.parse_args()

    return args

def batch_evaluation(pred, labels):
    correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
    return list(correct_prediction.numpy())

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

    # fix cudnn error, if you use gpu device
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.compat.v2.config.experimental.set_memory_growth(gpu, True)

    # testing datasets pipe
    test_tfrecords_f = os.path.join(args.test_tfrecords_file_path)
    test_dataset = tf.data.TFRecordDataset(test_tfrecords_f)
    test_dataset = test_dataset.map(test_parse_function)
    test_dataset = test_dataset.batch(args.test_batch_size)

    # load pretrained model or pretrained weights
    _custom_objects = custom_objects
    model = tf.keras.models.load_model(args.pretrained_model, custom_objects=_custom_objects)
    # model = MobileNetV3(type=args.model_type, input_shape=args.input_shape, classes_number=args.classes_number,
    #                     l2_reg=args.weight_decay, dropout_rate=args.dropout_rate, name="MobileNetV3")
    # model.load_weights(args.pretrained_model)
    tf.keras.backend.set_learning_phase(False)

    total_predict = []
    count = 0
    for i, (images, labels) in enumerate(test_dataset):
        logits = model(images, training=args.train_phase)
        pred = tf.nn.softmax(logits)
        batch_correct_prediction = batch_evaluation(pred, labels)
        total_predict.extend(batch_correct_prediction)
        count += len(labels)
        if count % 100 == 0:
            print(f'Successful to processed {count}')
    print(f'Successful to processed {count}')

    total_predict = np.asarray(total_predict)
    Accuracy = tf.reduce_mean(total_predict)
    print(f'test total images {count}, Accuracy is {Accuracy}!')
