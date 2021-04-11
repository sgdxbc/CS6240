import tensorflow as tf
from common import *


music_path = music_settings["music_path"]
perturbated_path = music_settings["output_path"]
adv_chunk_length = music_settings["perturbation_chunk_length"]
chunk_count = music_settings["chunk_count"]
adv_delay_interval = adv_chunk_length
adv_delay_interval2 = sample_interval
delay_per_batch = batch_size
stop_when_no_progress_in = 20


# |<------------------last chuck------------------>|<-SAMPLE_RATE - adv_chuck_length->|
# |<-SAMPLE_RATE - adv_chuck_length->|<-adv_chuck->|
music = decode_audio(tf.io.read_file(str(music_path)))[
    : chunk_count * SAMPLE_RATE + (SAMPLE_RATE - adv_chunk_length)
]

# adv = tf.Variable(tf.random.normal([adv_chunk_length * chunk_count]))
adv = tf.Variable(tf.zeros([adv_chunk_length * chunk_count]))

zero_length = SAMPLE_RATE - adv_chunk_length
chuck_length = zero_length + adv_chunk_length + zero_length


def underlay_mask(chuck_index, delays, count):
    music_chunk = music[
        chuck_index * SAMPLE_RATE : chuck_index * SAMPLE_RATE + chuck_length
    ]
    adv_chunk = tf.concat(
        [
            tf.zeros([zero_length]),
            adv[chuck_index * adv_chunk_length : (chuck_index + 1) * adv_chunk_length],
            tf.zeros([zero_length]),
        ],
        axis=0,
    )
    whole_chunk = music_chunk + adv_chunk
    mask_group = tf.stack(
        [whole_chunk[delay : delay + SAMPLE_RATE] for delay in delays]
    )
    return tf.repeat(mask_group, repeats=count, axis=0)


def underlay_total():
    return music + tf.concat(
        [
            *[
                tf.concat(
                    [
                        tf.zeros([zero_length]),
                        adv[i * adv_chunk_length : (i + 1) * adv_chunk_length],
                    ],
                    axis=0,
                )
                for i in range(chunk_count)
            ],
            tf.zeros([zero_length]),
        ],
        axis=0,
    )


last_loss = tf.zeros([])


def loss_fn(delays):
    dist_mean = []
    for chunk_index in range(chunk_count):
        xx, yy = xx_mat + underlay_mask(chunk_index, delays, x_mat.shape[0]), yy_mat
        dist_mean.append(tf.math.reduce_mean(tf.keras.losses.mse(yy, pred(xx))))
    norm = tf.keras.losses.mse(tf.zeros([adv.shape[0]]), adv)
    global last_loss
    last_loss = tf.math.reduce_mean(dist_mean) + alpha * norm
    return last_loss


def accuracy(x_mat, yi, interval=adv_delay_interval):
    accuracy_list = []
    underlay = underlay_total()
    for delay in range(0, underlay.shape[0] - SAMPLE_RATE, interval):
        xx = (
            tf.tile(
                [underlay[delay : delay + SAMPLE_RATE]], multiples=[x_mat.shape[0], 1]
            )
            + x_mat
        )
        p = tf.argmax(pred(xx), axis=1)
        accuracy_list.append(tf.math.count_nonzero(p == yi) / len(yi))
    return tf.math.reduce_mean(accuracy_list)


opt_loop(
    loss_fn,
    adv,
    accuracy,
    lambda: last_loss,
    SAMPLE_RATE - adv_chunk_length,
    adv_delay_interval,
    [adv_delay_interval, sample_interval],
)


tf.io.write_file(
    str(perturbated_path),
    tf.audio.encode_wav(tf.expand_dims(underlay_total(), -1), SAMPLE_RATE),
)
