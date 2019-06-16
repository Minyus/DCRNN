import tensorflow as tf
import numpy as np


def cut_to_zeros(a_tensor, pct_tensor, threshold):
    bool_tensor = tf.greater_equal(pct_tensor, threshold)
    cut_tensor = tf.where(bool_tensor, a_tensor, tf.zeros_like(a_tensor))
    return cut_tensor


def dense_to_sparse(dense_tensor):
    indices = tf.where(tf.not_equal(dense_tensor, tf.constant(0, dense_tensor.dtype)))
    values = tf.gather_nd(dense_tensor, indices)
    shape = dense_tensor.get_shape()
    sparse_tensor = tf.SparseTensor(indices, values, shape)
    return sparse_tensor


def thresholded_dense_to_sparse(a_tensor, pct_tensor, threshold):
    cut_tensor = cut_to_zeros(a_tensor, pct_tensor, threshold)
    sparse_tensor = dense_to_sparse(cut_tensor)
    sparse_tensor = tf.sparse_reorder(sparse_tensor)
    return sparse_tensor


def linear_cosine_decay_start_end(start, end, global_step, decay_steps):
    norm_decay = \
        tf.train.linear_cosine_decay(learning_rate=1.0,
                                     alpha=0.0,
                                     beta=0.0,
                                     global_step=global_step,
                                     decay_steps=decay_steps,
                                     )
    output = ((start-end) * norm_decay + end)
    return output

if __name__ == '__main__':
    a = np.reshape(np.arange(24), (3, 4, 2))
    with tf.Session() as sess:
        a_t = tf.constant(a)
        sparse = dense_to_sparse(a_t)
        dense = tf.sparse_tensor_to_dense(sparse)
        b = sess.run(dense)
    print(np.all(a == b))