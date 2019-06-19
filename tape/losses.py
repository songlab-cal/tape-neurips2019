import tensorflow as tf
import rinokeras as rk


def classification_loss_and_accuracy(labels, logits, mask=None):
    mask = 1 if mask is None else tf.cast(mask, logits.dtype)

    predictions = tf.argmax(logits, -1, output_type=labels.dtype)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels, logits, mask)
    accuracy = rk.utils.accuracy(labels, predictions, mask)
    return loss, accuracy


def classification_loss_and_top1_top5_top10_accuracies(labels, logits, mask=None):
    mask = 1 if mask is None else tf.cast(mask, logits.dtype)

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits, mask)

    in_top_1 = tf.nn.in_top_k(logits, labels, 1)
    in_top_5 = tf.nn.in_top_k(logits, labels, 5)
    in_top_10 = tf.nn.in_top_k(logits, labels, 10)

    in_top_1 = tf.cast(in_top_1, logits.dtype)
    in_top_5 = tf.cast(in_top_5, logits.dtype)
    in_top_10 = tf.cast(in_top_10, logits.dtype)

    weights = tf.ones_like(in_top_1) * mask
    denominator = tf.reduce_sum(weights) + 1e-10
    top_1_accuracy = tf.reduce_sum(in_top_1 * weights) / denominator
    top_5_accuracy = tf.reduce_sum(in_top_5 * weights) / denominator
    top_10_accuracy = tf.reduce_sum(in_top_10 * weights) / denominator

    return loss, top_1_accuracy, top_5_accuracy, top_10_accuracy


def binary_loss_and_accuracy(labels, logits, mask=None):
    mask = 1 if mask is None else tf.cast(mask, logits.dtype)

    predictions = tf.cast(logits > 0, labels.dtype)
    loss = tf.losses.sigmoid_cross_entropy(labels, logits, mask)
    accuracy = rk.utils.accuracy(labels, predictions, mask)
    return loss, accuracy
