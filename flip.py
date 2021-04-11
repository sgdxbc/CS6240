import tensorflow as tf
from common import *

adv_sample_number = flipping_settings["perturbation_length"]
adv_path = flipping_settings["output_path"]
adv = tf.Variable(tf.zeros([adv_sample_number]))


def underlay_mask(delays, count):
    group = tf.stack(
        [
            tf.roll(
                tf.concat([adv, tf.zeros([SAMPLE_RATE - adv_sample_number])], axis=0),
                shift=delay,
                axis=0,
            )
            for delay in delays
        ]
    )
    return tf.repeat(group, repeats=count, axis=0)


last_loss = tf.zeros([])


def loss_fn(delays):
    xx, yy = xx_mat + underlay_mask(delays, x_mat.shape[0]), yy_mat
    dist = tf.keras.losses.mse(yy, pred(xx))
    norm = tf.keras.losses.mse(tf.zeros([adv_sample_number]), adv)
    global last_loss
    last_loss = tf.math.reduce_mean(dist) + alpha * norm
    return last_loss


def accuracy(x_mat, yi):
    accuracy_list = []
    for delay in range(0, SAMPLE_RATE - adv_sample_number, sample_interval):
        xx = x_mat + underlay_mask([delay], x_mat.shape[0])
        p = tf.argmax(pred(xx), axis=1)
        accuracy_list.append(tf.math.count_nonzero(p == yi) / len(yi))
    return tf.math.reduce_mean(accuracy_list)


opt_loop(loss_fn, adv, accuracy, lambda: last_loss, SAMPLE_RATE - adv_sample_number)

tf.io.write_file(
    str(adv_path), tf.audio.encode_wav(tf.expand_dims(adv, -1), SAMPLE_RATE)
)
