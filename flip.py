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
    dist_list_list = [[] for _ in range(len(target_map))]
    for i in range(dist.shape[0] // train_count):
        dist_list_list[i % len(target_map)].append(
            tf.math.reduce_mean(dist[i * train_count : (i + 1) * train_count])
        )
    dist_list = [tf.math.reduce_mean(sublist) for sublist in dist_list_list]
    dist_sum = tf.math.reduce_mean(dist_list)
    dist_diff = tf.math.reduce_std(dist_list)
    norm = tf.keras.losses.mse(tf.zeros([adv_sample_number]), adv)
    global last_loss
    last_loss = dist_sum + beta * dist_diff + alpha * norm
    return last_loss


def accuracy(x_mat, yi):
    accuracy_list = []
    biacclist = ([], [])
    for delay in range(0, SAMPLE_RATE - adv_sample_number, sample_interval):
        xx = x_mat + underlay_mask([delay], x_mat.shape[0])
        p = tf.argmax(pred(xx), axis=1)
        a = p == yi
        accuracy_list.append(tf.math.count_nonzero(p == yi) / len(yi))
        biacclist[0].append(tf.math.count_nonzero(a[:a.shape[0] // 2]) / len(yi) * 2)
        biacclist[1].append(tf.math.count_nonzero(a[a.shape[0] // 2:]) / len(yi) * 2)
    print(tf.math.reduce_mean(biacclist[0]).numpy(), tf.math.reduce_mean(biacclist[1]).numpy())
    return tf.math.reduce_mean(accuracy_list)


opt_loop(loss_fn, adv, accuracy, lambda: last_loss, SAMPLE_RATE - adv_sample_number, max_epoch=200)

tf.io.write_file(
    str(adv_path), tf.audio.encode_wav(tf.expand_dims(adv, -1), SAMPLE_RATE)
)
