from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf
import time
import yaml

from lib import utils, metrics
from lib.AMSGrad import AMSGrad
from lib.metrics import masked_mae_loss

from model.dcrnn_model import DCRNNModel


class DCRNNSupervisor(object):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, adj_mx, dataloaders, **kwargs):

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self._paths_kwargs = kwargs.get('paths')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, level=log_level)
        self._writer = tf.summary.FileWriter(self._log_dir)
        self._logger.info(kwargs)

        # Data preparation
        STDATALOADER = True
        if STDATALOADER:
            self._data = dataloaders
        if not STDATALOADER:
            self._data = utils.load_dataset(**self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # Build models.
        scaler = self._data['scaler']
        with tf.name_scope('Train'):
            with tf.variable_scope('DCRNN', reuse=False):
                train_batch_size = dataloaders['train_loader'].batch_size if STDATALOADER \
                    else self._data_kwargs['batch_size']
                self._train_model = DCRNNModel(is_training=True, scaler=scaler,
                                               batch_size=train_batch_size,
                                               adj_mx=adj_mx, **self._model_kwargs)

        with tf.name_scope('Val'):
            with tf.variable_scope('DCRNN', reuse=True):
                val_batch_size = dataloaders['val_loader'].batch_size if STDATALOADER \
                    else self._data_kwargs['val_batch_size']
                self._val_model = DCRNNModel(is_training=False, scaler=scaler,
                                              batch_size=val_batch_size,
                                              adj_mx=adj_mx, **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('DCRNN', reuse=True):
                test_batch_size = dataloaders['test_loader'].batch_size if STDATALOADER \
                    else self._data_kwargs['test_batch_size']
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
        self._loss_fn = masked_mae_loss(scaler, null_val)
        self._train_loss = self._loss_fn(preds=preds, labels=labels)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self._train_loss, tvars)
        max_grad_norm = kwargs['train'].get('max_grad_norm', 1.)
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        global_step = tf.train.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step,
                                                   name='train_op')

        self._epoch = 0

        max_to_keep = self._train_kwargs.get('max_to_keep', 100)
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)

        # Log model statistics.
        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        self._logger.info('Total number of trainable parameters: {:d}'.\
                          format(total_trainable_parameter))
        for var in tf.global_variables():
            self._logger.debug('{}, {}'.format(var.name, var.get_shape()))

    @staticmethod
    def _get_log_dir(kwargs):
        # log_dir = kwargs['train'].get('log_dir')
        log_dir = kwargs['paths']['model_dir']

        if 0: # TODO: remove
        # if log_dir is None:
            batch_size = kwargs['data'].get('train_batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def run_epoch_generator(self, sess, model, data_generator,
                            return_output=False,
                            training=False, writer=None,
                            return_ground_truth=False):
        losses = []
        maes = []
        outputs = []
        ground_truths = []

        output_dim = self._model_kwargs.get('output_dim')
        preds = model.outputs
        labels = model.labels[..., :output_dim]
        loss = self._loss_fn(preds=preds, labels=labels)
        fetches = {
            'loss': loss,
            'mae': loss, # TODO:
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

            vals = sess.run(fetches, feed_dict=feed_dict)

            losses.append(vals['loss'])
            maes.append(vals['mae'])
            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])
            if return_output:
                outputs.append(vals['outputs'])
            if return_ground_truth:
                ground_truths.append(y)
            self._logger.debug('Completed {} step #: {:06d} | global step: {:06d}'.\
                               format('training' if training else 'evaluation', step_i + 1,
                                      vals['global_step']))
            if step_i + 1 >= self._train_kwargs.get('steps_per_epoch', 1000000):
                break
            if vals['global_step'] >= self._train_kwargs.get('target_steps', 1000000):
                break

        results = {
            'loss': np.mean(losses),
            'mae': np.mean(maes)
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

    def _train(self, sess, base_lr, epoch, steps, patience=50, epochs=100,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, save_model=1,
               test_every_n_epochs=10, **train_kwargs):
        history = []
        min_val_loss = float('inf')
        wait = 0

        max_to_keep = train_kwargs.get('max_to_keep', 100)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        model_filename = self._paths_kwargs.get('model_filename')
        if model_filename is not None:
            saver.restore(sess, model_filename)
        else:
            sess.run(tf.global_variables_initializer())

        self._epoch = epoch + 1

        while self._epoch <= epochs:
            self._logger.info('Started epoch: {} / {}'.format(self._epoch, epochs))
            # Learning rate schedule.
            new_lr = max(min_learning_rate,
                         base_lr * (lr_decay_ratio ** np.sum(self._epoch >= np.array(steps))))
            self.set_lr(sess=sess, lr=new_lr)

            start_time = time.time()
            train_results = self.run_epoch_generator(sess, self._train_model,
                                                     self._data['train_loader'].get_iterator(),
                                                     training=True,
                                                     writer=self._writer)
            train_loss, train_mae = train_results['loss'], train_results['mae']
            if train_loss > 1e5:
                self._logger.warning('Gradient explosion detected. Ending...')
                break

            global_step = sess.run(tf.train.get_or_create_global_step())
            # Compute validation error.
            val_results = self.run_epoch_generator(sess, self._val_model,
                                                   self._data['val_loader'].get_iterator(),
                                                   training=False)
            val_loss, val_mae = np.asscalar(val_results['loss']), np.asscalar(val_results['mae'])

            utils.add_simple_summary(self._writer,
                                     ['loss/train_loss', 'metric/train_mae', 'loss/val_loss',
                                      'metric/val_mae'],
                                     [train_loss, train_mae, val_loss, val_mae],
                                     global_step=global_step)
            end_time = time.time()
            message = \
                'Epoch: {}/{}|Global Step: {}|train_mae: {:.4f}|val_mae: {:.4f}|lr:{:.6f}|{:.1f}s'.\
                format(self._epoch, epochs, global_step, train_mae, val_mae, new_lr,
                       (end_time - start_time))
            self._logger.info(message)
            if self._epoch % test_every_n_epochs == test_every_n_epochs - 1:
                self.evaluate(sess)
            if val_loss <= min_val_loss:
                wait = 0
                if save_model > 0:
                    model_filename = self.save(sess, val_loss)
                self._logger.info(
                    'Val loss decrease from %.4f to %.4f, saved to %s' % \
                    (min_val_loss, val_loss, model_filename))
                min_val_loss = val_loss
            else:
                wait += 1
                if wait > patience:
                    self._logger.warning('Early stopping at epoch: %d' % self._epoch)
                    break

            history.append(val_mae)
            # Increases epoch.
            self._epoch += 1

            sys.stdout.flush()
            if global_step >= self._train_kwargs.get('target_steps', 1000000):
                self._logger.info('Finish training since the global step reached: {}'.\
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

        STDATALOADER = True
        if STDATALOADER:
            y_truths = np.concatenate(test_results['ground_truths'], axis=0)
        if not STDATALOADER:
            y_truths = self._data['y_test']

        assert y_preds.shape[:3] == y_truths.shape[:3], \
            'NOT {} == {}'.format(y_preds.shape[:3], y_truths.shape[:3])

        y_preds_original = []
        y_truths_original = []

        for horizon_i in range(y_truths.shape[1]):
            y_pred_original = scaler.inverse_transform(y_preds[:, horizon_i, :, 0])
            y_preds_original.append(y_pred_original)
            self._logger.info(stat_str(y_pred_original, horizon_i, 'pred'))

            y_truth_original = scaler.inverse_transform(y_truths[:, horizon_i, :, 0])
            y_truths_original.append(y_truth_original)

            if not np.all(np.isnan(y_truth_original)):
                self._logger.info(stat_str(y_truth_original, horizon_i, 'true'))

                for null_val in [0, np.nan]:
                    desc = r'0 excl.' if null_val == 0 else 'any    '
                    rmse = metrics.masked_rmse_np(y_pred_original, y_truth_original, null_val=null_val)
                    r2 = metrics.masked_r2_np(y_pred_original, y_truth_original, null_val=null_val)
                    mae = metrics.masked_mae_np(y_pred_original, y_truth_original, null_val=null_val)
                    mape = metrics.masked_mape_np(y_pred_original, y_truth_original, null_val=null_val) \
                        if null_val == 0 else np.nan
                    self._logger.info(
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

    def load(self, sess, model_filename):
        """
        Restore from saved model.
        :param sess:
        :param model_filename:
        :return:
        """
        self._saver.restore(sess, model_filename)

    def save(self, sess, val_loss):
        config = dict(self._kwargs)
        global_step = np.asscalar(sess.run(tf.train.get_or_create_global_step()))
        prefix = os.path.join(self._log_dir, 'model-{:09.6f}'.format(val_loss))
        config['train']['epoch'] = self._epoch
        config['train']['global_step'] = global_step
        config['train']['log_dir'] = self._log_dir
        config['train']['model_filename'] = self._saver.save(sess, prefix,
                                                             global_step=global_step,
                                                             write_meta_graph=False)
        config_filename = 'dcrnn_config_global_step_{}.yaml'.format(global_step)
        with open(os.path.join(self._log_dir, config_filename), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config['train']['model_filename']


def stat_str(arr, horizon_i, desc=''):
    stat = 'T+{:02d}|{}|min: {:8.5f}|max: {:8.5f}|mean: {:8.5f}|std dev: {:8.5f}'. \
        format(horizon_i + 1, desc, np.min(arr), np.max(arr), np.mean(arr), np.std(arr))
    return stat
