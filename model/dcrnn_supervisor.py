from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf
import time

from lib import utils, metrics
from lib.AMSGrad import AMSGrad
from lib.metrics import masked_mae_loss, masked_rmse_loss, masked_mse_loss

from model.dcrnn_model import DCRNNModel


class DCRNNSupervisor(object):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, sess, adj_mx, dataloaders, kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self._paths_kwargs = kwargs.get('paths')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        # log_level = self._kwargs.get('log_level', 'INFO')
        # self._logger = utils.get_logger(self._log_dir, __name__, level=log_level)
        self._writer = tf.summary.FileWriter(self._log_dir)

        # Data preparation
        self._data = dataloaders

        # for k, v in self._data.items():
        #     if hasattr(v, 'shape'):
        #         self._kwargs.logger.info((k, v.shape))

        # Build models.
        scaler = self._data['scaler']
        with tf.name_scope('Train'):
            with tf.variable_scope('DCRNN', reuse=False):
                train_batch_size = dataloaders['train_loader'].batch_size
                self._train_model = DCRNNModel(is_training=True, scaler=scaler,
                                               batch_size=train_batch_size,
                                               adj_mx=adj_mx, **self._model_kwargs)

        with tf.name_scope('Val'):
            with tf.variable_scope('DCRNN', reuse=True):
                val_batch_size = dataloaders['val_loader'].batch_size
                self._val_model = DCRNNModel(is_training=False, scaler=scaler,
                                              batch_size=val_batch_size,
                                              adj_mx=adj_mx, **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('DCRNN', reuse=True):
                test_batch_size = dataloaders['test_loader'].batch_size
                self._test_model = DCRNNModel(is_training=False, scaler=scaler,
                                              batch_size=test_batch_size,
                                              adj_mx=adj_mx, **self._model_kwargs)

        # Learning rate.
        self._lr = tf.get_variable('learning_rate', shape=(),
                                   initializer=tf.constant_initializer(0.01),
                                   trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr, name='lr_update')

        # Configure optimizer
        optimizer_name = self._train_kwargs.get('optimizer', 'adam').lower()
        epsilon = float(self._train_kwargs.get('epsilon', 1e-3))
        optimizer = tf.train.AdamOptimizer(self._lr, epsilon=epsilon)
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr, )
        elif optimizer_name == 'amsgrad':
            optimizer = AMSGrad(self._lr, epsilon=epsilon)

        # Calculate loss
        output_dim = self._model_kwargs.get('output_dim')
        preds = self._train_model.outputs
        labels = self._train_model.labels[..., :output_dim]

        null_val = 0. if kwargs['model'].get('exclude_zeros_in_metric', True) else np.nan

        loss_func_dict = {'mae': masked_mae_loss(scaler, null_val),
                          'rmse': masked_rmse_loss(scaler, null_val),
                          'mse': masked_mse_loss(scaler, null_val)}
        self._loss_fn = loss_func_dict.get(kwargs['train'].get('loss_func', 'mae'))
        self._metric_fn = loss_func_dict.get(kwargs['train'].get('metric_func', 'mae'))

        self._train_loss = self._loss_fn(preds=preds, labels=labels)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self._train_loss, tvars)
        max_grad_norm = kwargs['train'].get('max_grad_norm', 1.)
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=tf.train.get_or_create_global_step(),
                                                   name='train_op')

        max_to_keep = self._train_kwargs.get('max_to_keep', 100)
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)

        # load model
        model_filename = self._paths_kwargs.get('model_filename')
        if model_filename is not None:
            self._saver.restore(sess, model_filename)
            self._kwargs.logger.info('Pretrained model was loaded from : {}'.format(model_filename))
        else:
            sess.run(tf.global_variables_initializer())

        # Log model statistics.
        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        self._kwargs.logger.info('Total number of trainable parameters: {:d}'.\
                          format(total_trainable_parameter))
        for var in tf.global_variables():
            self._kwargs.logger.debug('{}, {}'.format(var.name, var.get_shape()))

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['paths']['model_dir']
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def run_epoch_generator(self, sess, model, data_generator,
                            return_output=False,
                            training=False, writer=None,
                            return_ground_truth=False):
        losses = []
        metrics = []
        outputs = []
        ground_truths = []

        output_dim = self._model_kwargs.get('output_dim')
        preds = model.outputs
        labels = model.labels[..., :output_dim]
        loss = self._loss_fn(preds=preds, labels=labels)
        metric = self._metric_fn(preds=preds, labels=labels)
        fetches = {
            'loss': loss,
            'metric': metric,
            'global_step': tf.train.get_or_create_global_step()
        }
        if training:
            fetches.update({
                'train_op': self._train_op
            })
            merged = model.merged
            if merged is not None:
                fetches.update({'merged': merged})

        if return_output:
            fetches.update({
                'outputs': model.outputs
            })

        for step_i, (x, y) in enumerate(data_generator):
            feed_dict = {
                model.inputs: x,
                model.labels: y,
            }

            v = sess.run(fetches, feed_dict=feed_dict)

            losses.append(v['loss'])
            metrics.append(v['metric'])
            if writer is not None and 'merged' in v:
                writer.add_summary(v['merged'], global_step=v['global_step'])
            if return_output:
                outputs.append(v['outputs'])
            if return_ground_truth:
                ground_truths.append(y)

            train_steps_per_epoch = self._data_kwargs.get('train_steps_per_epoch', 1)
            target_train_steps = self._data_kwargs.get('target_train_steps', 1000000)
            self._kwargs.logger.debug('Completed {} step: {:06d}/{:06d}|global step {:06d}/{:06d}|loss:{}|metric:{}'.\
                               format('training' if training else 'validation',
                                      step_i + 1, train_steps_per_epoch,
                                      v['global_step'], target_train_steps, v['loss'], v['metric']))

            if training:
                if ((step_i + 1) >= train_steps_per_epoch) or (v['global_step'] >= target_train_steps):
                    break

        results = {
            'loss': np.mean(losses),
            'metric': np.mean(metrics)
        }
        if return_output:
            results['outputs'] = outputs
        if return_ground_truth:
            results['ground_truths'] = ground_truths

        return results

    def get_lr(self, sess):
        return np.asscalar(sess.run(self._lr))

    def set_lr(self, sess, lr):
        sess.run(self._lr_update, feed_dict={
            self._new_lr: lr
        })

    def train(self, sess, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(sess, **kwargs)

    def _train(self, sess, base_lr, lr_decay_steps, patience=50,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, save_model=1,
               test_every_n_epochs=10, cosine_decay_steps=None, **kwargs):
        history = []
        min_val_metric = float('inf')
        wait = 0
        target_train_steps = self._data_kwargs.get('target_train_steps', 1000000)

        while True:

            self._global_step = sess.run(tf.train.get_or_create_global_step())
            self._epoch = self._global_step // self._data_kwargs['train_steps_per_epoch']

            self._kwargs.logger.info('Global step: {} / {} | epoch: {}'.\
                              format(self._global_step, target_train_steps, self._epoch))

            # Learning rate schedule.
            new_lr = base_lr * (lr_decay_ratio ** np.sum(self._global_step >= np.array(lr_decay_steps))) \
                if cosine_decay_steps is None \
                else sess.run(tf.train.cosine_decay(learning_rate=base_lr, global_step=self._global_step,
                                           decay_steps=cosine_decay_steps))
            new_lr = max(min_learning_rate, new_lr)
            self.set_lr(sess=sess, lr=new_lr)

            start_time = time.time()
            train_results = self.run_epoch_generator(sess, self._train_model,
                                                     self._data['train_loader'].get_iterator(),
                                                     training=True,
                                                     writer=self._writer)
            train_loss = train_results['loss']
            train_metric = train_results['metric']
            if train_loss > 1e5:
                self._kwargs.logger.warning('Gradient explosion detected. Ending...')
                break

            global_step = sess.run(tf.train.get_or_create_global_step())
            # Compute validation error.
            val_results = self.run_epoch_generator(sess, self._val_model,
                                                   self._data['val_loader'].get_iterator(),
                                                   training=False)
            # val_loss = np.asscalar(val_results['loss'])
            # val_metric = np.asscalar(val_results['metric'])
            val_loss = val_results['loss']
            val_metric = val_results['metric']

            utils.add_simple_summary(self._writer,
                                     ['loss/train_loss',
                                      'metric/train_metric',
                                      'loss/val_loss',
                                      'metric/val_metric',
                                      'learning_rate',
                                      ],
                                     [train_loss,
                                      train_metric,
                                      val_loss,
                                      val_metric,
                                      new_lr,
                                      ],
                                     global_step=global_step)
            end_time = time.time()

            message = \
                ('Epoch: {:6d}|global step: {:6d}/{:6d}|' +
                 'train_loss: {:8.5f}|train_metric: {:8.5f}|val_loss: {:8.5f}|val_metric: {:8.5f}|' +
                 'lr:{:.6f}|{:.0f}s/epoch').\
                format(self._epoch, global_step, target_train_steps,
                       train_loss, train_metric, val_loss, val_metric,
                       new_lr, (end_time - start_time))
            self._kwargs.logger.info(message)
            if self._epoch % test_every_n_epochs == test_every_n_epochs - 1:
                self.evaluate(sess)
            if val_metric <= min_val_metric:
                wait = 0
                if save_model > 0:
                    model_filename = self.save(sess, val_metric)
                self._kwargs.logger.info(
                    'Val metric decrease from {:8.5f} to {:8.5f}. The model was saved at: {}'.\
                    format(min_val_metric, val_metric, model_filename))
                min_val_metric = val_metric
            else:
                wait += 1
                if wait > patience:
                    self._kwargs.logger.warning('Early stopping at epoch: {}'.format(self._epoch))
                    break

            history.append(val_loss)
            # Increases epoch.
            self._epoch += 1

            sys.stdout.flush()
            if global_step >= self._data_kwargs.get('target_train_steps', 1000000):
                self._kwargs.logger.info('Finish training since the global step reached: {}'.\
                                  format(global_step))
                break
        return np.min(history)

    def evaluate(self, sess, **kwargs):
        global_step = sess.run(tf.train.get_or_create_global_step())
        test_results = self.run_epoch_generator(sess, self._test_model,
                                                self._data['test_loader'].get_iterator(),
                                                return_output=True,
                                                training=False,
                                                return_ground_truth=True)

        test_loss = test_results['loss']
        utils.add_simple_summary(self._writer, ['loss/test_loss'], [test_loss],
                                 global_step=global_step)

        # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
        y_preds = np.concatenate(test_results['outputs'], axis=0)

        scaler = self._data['scaler']

        y_truths = np.concatenate(test_results['ground_truths'], axis=0)

        assert y_preds.shape[:3] == y_truths.shape[:3], \
            'NOT {} == {}'.format(y_preds.shape[:3], y_truths.shape[:3])

        y_preds_original = []
        y_truths_original = []

        min_output_value = self._data_kwargs.get('min_output_value')
        max_output_value = self._data_kwargs.get('max_output_value')
        clip = (min_output_value is not None) or (max_output_value is not None)
        if clip:
            self._kwargs.logger.info('The output values are clipped to range: [{}, {}]'.\
                              format(min_output_value, max_output_value))

        for horizon_i in range(y_truths.shape[1]):
            y_pred_original = scaler.inverse_transform(y_preds[:, horizon_i, :, 0])
            if clip:
                y_pred_original = \
                    np.clip(y_pred_original, min_output_value, max_output_value,
                            out=y_pred_original)
            y_preds_original.append(y_pred_original)
            self._kwargs.logger.info(stat_str(y_pred_original, horizon_i, 'pred'))

            y_truth_original = scaler.inverse_transform(y_truths[:, horizon_i, :, 0])
            y_truths_original.append(y_truth_original)

            if not np.all(np.isnan(y_truth_original)):
                self._kwargs.logger.info(stat_str(y_truth_original, horizon_i, 'true'))

                for null_val in [0, np.nan]:
                    desc = r'0 excl.' if null_val == 0 else 'any    '
                    rmse = metrics.masked_rmse_np(y_pred_original, y_truth_original, null_val=null_val)
                    r2 = metrics.masked_r2_np(y_pred_original, y_truth_original, null_val=null_val)
                    mae = metrics.masked_mae_np(y_pred_original, y_truth_original, null_val=null_val)
                    mape = metrics.masked_mape_np(y_pred_original, y_truth_original, null_val=null_val) \
                        if null_val == 0 else np.nan
                    self._kwargs.logger.info(
                        "T+{:02d}|{}|RMSE: {:8.5f}|R2: {:8.4f}|MAE: {:5.2f}|MAPE: {:5.3f}|".\
                            format(horizon_i + 1, desc, rmse, r2, mae, mape,)
                        )

                utils.add_simple_summary(self._writer,
                                         ['%s_%d' % (item, horizon_i + 1) for item in
                                          ['metric/rmse', 'metric/mape', 'metric/mae']],
                                         [rmse, mape, mae],
                                         global_step=global_step)
        outputs = {
            'predictions': y_preds_original,
            'groundtruth': y_truths_original
        }
        return outputs

    def save(self, sess, metric):
        global_step = sess.run(tf.train.get_or_create_global_step())
        metric_func = self._kwargs['train'].get('metric_func', 'mae')
        prefix = os.path.join(self._log_dir, 'model_{}{:09.7f}'.format(metric_func, metric))
        model_filename = \
            self._saver.save(sess, prefix, global_step=global_step, write_meta_graph=False)
        return model_filename


def stat_str(arr, horizon_i, desc=''):
    stat = 'T+{:02d}|{}|min: {:8.5f}|max: {:8.5f}|mean: {:8.5f}|std dev: {:8.5f}'. \
        format(horizon_i + 1, desc, np.min(arr), np.max(arr), np.mean(arr), np.std(arr))
    return stat
