from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import cPickle

from tfutils import base, data, optimizer

import json
import copy
import argparse

import cate_network_builder
import h5py
import time

host = os.uname()[1]

DATA_PATH = {}
DATA_PATH['train/Data_force'] = '/media/data3/chengxuz/whisker/tfrecords/Data_force/'
DATA_PATH['train/Data_torque'] = '/media/data3/chengxuz/whisker/tfrecords/Data_torque/'
DATA_PATH['train/category'] = '/media/data3/chengxuz/whisker/tfrecords/category/'
#DATA_PATH['val/images'] = '/media/data2/one_world_dataset/tfvaldata/images/'
#DATA_PATH['val/normals'] = '/media/data2/one_world_dataset/tfvaldata/normals/'

train_data_path_prefix = '/mnt/fs0/chengxuz/Data/whisker/tfrecs_all/tfrecords'
val_data_path_prefix = '/mnt/fs0/chengxuz/Data/whisker/val_tfrecs/tfrecords_val'
#train_data_path_prefix = '/data/chengxuz/whisker/tfrecs_all/tfrecords'
#val_data_path_prefix = '/data/chengxuz/whisker/val_tfrecs/tfrecords_val'

save_num_now = None

if 'neuroaicluster' in host:
    DATA_PATH['train/Data_force'] = train_data_path_prefix + '/Data_force/'
    DATA_PATH['train/Data_torque'] = train_data_path_prefix + '/Data_torque/'
    DATA_PATH['train/category'] = train_data_path_prefix + '/category/'
    DATA_PATH['val/Data_force'] = val_data_path_prefix + '/Data_force/'
    DATA_PATH['val/Data_torque'] = val_data_path_prefix + '/Data_torque/'
    DATA_PATH['val/category'] = val_data_path_prefix + '/category/'
    DATA_PATH['Data_force_stat'] = train_data_path_prefix + '/Data_force/Data_force_combined.pkl'
    DATA_PATH['Data_torque_stat'] = train_data_path_prefix + '/Data_torque/Data_torque_combined.pkl'

NEW_DATA_PATH = {}
new_data_path_prefix = '/mnt/fs2/chengxuz/Data/whisker2/tfrecords2'

NEW_DATA_PATH['train/Data_force'] = new_data_path_prefix + '/Data_force/'
NEW_DATA_PATH['train/Data_torque'] = new_data_path_prefix + '/Data_torque/'
NEW_DATA_PATH['train/category'] = new_data_path_prefix + '/category/'
NEW_DATA_PATH['val/Data_force'] = new_data_path_prefix + '/Data_force/'
NEW_DATA_PATH['val/Data_torque'] = new_data_path_prefix + '/Data_torque/'
NEW_DATA_PATH['val/category'] = new_data_path_prefix + '/category/'

#OTHER_LABELS_LIST = ['speed', 'orn', 'scale', 'position']
#OTHER_LABELS_LIST = ['speed', 'orn', 'scale', 'position', 'objid']
OTHER_LABELS_LIST = ['speed', 'orn', 'scale', 'position', 'objid']

for other_label in OTHER_LABELS_LIST:
    for group in ['train', 'val']:
        NEW_DATA_PATH['%s/%s' % (group, other_label)] = '%s/%s/' % (new_data_path_prefix, other_label)

NEW_DATA_PATH['strain'] = '*_strain.tfrecords'
NEW_DATA_PATH['sval'] = '*_sval.tfrecords'

NEW_DATA_PATH['ctrain'] = 'ctrain_*.tfrecords'
NEW_DATA_PATH['cval'] = 'cval_*.tfrecords'


def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res

def online_agg_genfeautre(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(1)
    return agg_res

def in_top_k(inputs, outputs, target):
    return {'top1': tf.nn.in_top_k(outputs, inputs[target], 1),
            'top5': tf.nn.in_top_k(outputs, inputs[target], 5)}

def cmu_in_top_k(inputs, outputs, target):
    top1_list = []
    top5_list = []
    for output in outputs:
        tmp_ret = in_top_k(inputs, output, target)

        top1_list.append(tmp_ret['top1'])
        top5_list.append(tmp_ret['top5'])
    return {'top1': tf.concat(top1_list, 0), 'top5': tf.concat(top5_list, 0)}

def cmu_parallel_in_top_k(inputs, outputs, target):
    new_inputs = tf.split(inputs[target], axis = 0, num_or_size_splits = len(outputs))
    top1_list = []
    top5_list = []
    for new_input, new_output in zip(new_inputs, outputs):
        tmp_ret = cmu_in_top_k({target: new_input}, new_output, target)
        top1_list.append(tmp_ret['top1'])
        top5_list.append(tmp_ret['top5'])

    return {'top1': tf.concat(top1_list, 0), 'top5': tf.concat(top5_list, 0)}

def parallel_in_top_k(inputs, outputs, target):
    return in_top_k(inputs, tf.concat(outputs, 0), target)

def cmu_softmax_cross_entropy_loss(labels, logits, **kwargs):
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        n_time = len(logits)
        losses = []
        for i, logit in enumerate(logits):
            losses.append(
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=labels, logits=logit, **kwargs)))
        return tf.reduce_mean(losses)

def parallel_softmax_cross_entropy_loss(labels, logits, gpu_offset = 0, **kwargs):
    #labels = tf.Print(labels, [labels], message = "Labels", summarize = 30)
    #y, idx = tf.unique(labels)
    #labels = tf.Print(labels, [tf.reduce_max(idx)], message = "Labels", summarize = 1)

    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        n_gpus = len(logits)
        print('Loss n gpus: %i' % n_gpus)
        if n_gpus>1:
            labels = tf.split(labels, axis=0, num_or_size_splits=n_gpus)
        else:
            labels = [labels]
        losses = []
        for i, (label, logit) in enumerate(zip(labels, logits)):
            with tf.device('/gpu:%d' % (i + gpu_offset)):
                with tf.name_scope('gpu_' + str(i)) as gpu_scope:
                    label = tf.squeeze(label)
                    losses.append(
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=label, logits=logit)))
                    tf.get_variable_scope().reuse_variables()
        return losses

def cmu_parallel_softmax_cross_entropy_loss(labels, logits, **kwargs):
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        n_gpus = len(logits)
        labels = tf.split(labels, axis=0, num_or_size_splits=n_gpus)
        losses = []
        for i, (label, logit) in enumerate(zip(labels, logits)):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('gpu_' + str(i)) as gpu_scope:
                    label = tf.squeeze(label)
                    losses.append(
                        tf.reduce_mean(cmu_softmax_cross_entropy_loss(
                                    labels=label, logits=logit)))
                    tf.get_variable_scope().reuse_variables()
        return losses

def parallel_reduce_mean(losses, **kwargs):
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        for i, loss in enumerate(losses):
            losses[i] = tf.reduce_mean(loss)
        return losses

class ParallelClipOptimizer(object):

    def __init__(self, optimizer_class, gpu_offset = 0, clip=True, *optimizer_args, **optimizer_kwargs):
        self._optimizer = optimizer_class(*optimizer_args, **optimizer_kwargs)
        self.clip = clip
        self.gpu_offset = gpu_offset

    def compute_gradients(self, *args, **kwargs):
        gvs = self._optimizer.compute_gradients(*args, **kwargs)
        if self.clip:
            # gradient clipping. Some gradients returned are 'None' because
            # no relation between the variable and loss; so we skip those.
            gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                   for grad, var in gvs if grad is not None]
        return gvs

    def minimize(self, losses, global_step):
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            grads_and_vars = []
            if not isinstance(losses, list):
                losses = [losses]

            for i, loss in enumerate(losses):
                with tf.device('/gpu:%d' % (i + self.gpu_offset)):
                    with tf.name_scope('gpu_' + str(i)) as gpu_scope:
                        grads_and_vars.append(self.compute_gradients(loss))

            if len(losses)==1:
                grads_and_vars = self.average_gradients(grads_and_vars)
            else:
                #with tf.device('/cpu:0'):
                grads_and_vars = self.average_gradients(grads_and_vars)

            return self._optimizer.apply_gradients(grads_and_vars,
                                               global_step=global_step)

    def average_gradients(self, all_grads_and_vars):
        average_grads_and_vars = []
        for grads_and_vars in zip(*all_grads_and_vars):
            grads = []
            for g, _ in grads_and_vars:
                grads.append(tf.expand_dims(g, axis=0))
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)
            # all variables are the same so we just use the first gpu variables
            var = grads_and_vars[0][1]
            grad_and_var = (grad, var)
            average_grads_and_vars.append(grad_and_var)
        return average_grads_and_vars

class WhiskerWorld(data.TFRecordsParallelByFileProvider):

    def __init__(self,
                 data_path,
                 group='train',
                 batch_size=1,
                 n_threads=4,
                 expand_spatial=False,
                 norm_flag = False,
                 split_12 = False,
                 norm_std = 1,
                 num_fake = 0,
                 patprefix = None,
                 otherlabels = False,
                 *args,
                 **kwargs):

        self.group = group
        self.force = 'Data_force'
        self.torque = 'Data_torque'
        self.label = 'category'
        self.batch_size = batch_size
        self.expand_spatial = expand_spatial
        self.norm_flag = norm_flag
        self.norm_std = norm_std
        self.split_12 = split_12
        self.num_fake = num_fake
        self.otherlabels = otherlabels
        if norm_flag:
            self.stat_path = {}
            self.stat_path['Data_force'] = data_path['Data_force_stat']
            self.stat_path['Data_torque'] = data_path['Data_torque_stat']
        postprocess = {self.force: [(self.postprocess_images, (), {})], self.torque: [(self.postprocess_images, (), {})]}

        file_pattern = '*.tfrecords'
        if not patprefix is None:
            file_pattern = data_path[patprefix + group]

        source_dirs = [data_path["%s/%s" % (group, self.force)] , data_path["%s/%s" % (group, self.torque)] , data_path["%s/%s" % (group, self.label)]]

        if self.otherlabels:
            self.otherlabel_list = OTHER_LABELS_LIST

            for other_label in self.otherlabel_list:
                source_dirs.append(data_path["%s/%s" % (group, other_label)])
                if not other_label=='objid':
                    if other_label=='speed':
                        postprocess[other_label] = [(self.postprocess_arrs_3, (), {})]
                    else:
                        postprocess[other_label] = [(self.postprocess_arrs_4, (), {})]

        super(WhiskerWorld, self).__init__(
            source_dirs = source_dirs,
            postprocess = postprocess,
            batch_size=batch_size,
            n_threads=n_threads,
            file_pattern=file_pattern,
            shuffle = True,
            *args, **kwargs)

    def set_data_shapes(self, data):
        for i in range(len(data)):
            for k in data[i]:
                # set shape[0] to batch size for all entries
                shape = data[i][k].get_shape().as_list()
                shape[0] = self.batch_size
                data[i][k].set_shape(shape)
        return data

    def slice_concat(self, data, curr_key, new_key):
        #print(data[curr_key].get_shape().as_list())
        slice0 = tf.strided_slice( data[curr_key], [0,0,0,0,0], [0,0,0,0,0], [3, 1, 1, 1, 1], end_mask = 31)
        slice1 = tf.strided_slice( data[curr_key], [1,0,0,0,0], [0,0,0,0,0], [3, 1, 1, 1, 1], end_mask = 31)
        slice2 = tf.strided_slice( data[curr_key], [2,0,0,0,0], [0,0,0,0,0], [3, 1, 1, 1, 1], end_mask = 31)

        #slice1 = tf.strided_slice( data[curr_key], [1], [2], [3])
        #slice2 = tf.strided_slice( data[curr_key], [2], [3], [3])
        #print(slice0.get_shape().as_list())
        data[new_key] = tf.concat([slice0, slice1, slice2], 1)
        #print(data[new_key].get_shape().as_list())

        if self.num_fake>0:
            shape_list = data[new_key].get_shape().as_list()
            pad_zeros = tf.zeros([shape_list[0]*self.num_fake] + shape_list[1:])
            data[new_key] = tf.concat([data[new_key], pad_zeros], 0)


        return data

    def slice_concat_12(self, data, curr_key, new_key):
        slice0 = tf.strided_slice( data[curr_key], [0,0,0,0,0], [0,0,0,0,0], [3, 1, 1, 1, 1], end_mask = 31)
        slice1 = tf.strided_slice( data[curr_key], [1,0,0,0,0], [0,0,0,0,0], [3, 1, 1, 1, 1], end_mask = 31)
        slice2 = tf.strided_slice( data[curr_key], [2,0,0,0,0], [0,0,0,0,0], [3, 1, 1, 1, 1], end_mask = 31)
        data[new_key] = tf.concat([slice0, slice1, slice2], 1)

        slice0_ = tf.strided_slice( data[curr_key], [0,0,0,0,0], [0,0,0,0,0], [4, 1, 1, 1, 1], end_mask = 31)
        slice1_ = tf.strided_slice( data[curr_key], [1,0,0,0,0], [0,0,0,0,0], [4, 1, 1, 1, 1], end_mask = 31)
        slice2_ = tf.strided_slice( data[curr_key], [2,0,0,0,0], [0,0,0,0,0], [4, 1, 1, 1, 1], end_mask = 31)
        slice3_ = tf.strided_slice( data[curr_key], [3,0,0,0,0], [0,0,0,0,0], [4, 1, 1, 1, 1], end_mask = 31)
        data[new_key] = tf.concat([slice0_, slice1_, slice2_, slice3_], 1) 
        return data

    def slice_concat_ol(self, data, curr_key, new_key):
        slice0 = tf.strided_slice( data[curr_key], [0,0], [0,0], [3, 1], end_mask = 3)
        slice1 = tf.strided_slice( data[curr_key], [1,0], [0,0], [3, 1], end_mask = 3)
        slice2 = tf.strided_slice( data[curr_key], [2,0], [0,0], [3, 1], end_mask = 3)

        data[new_key] = tf.concat([slice0, slice1, slice2], 1)

        if self.num_fake>0:
            shape_list = data[new_key].get_shape().as_list()
            pad_zeros = tf.zeros([shape_list[0]*self.num_fake] + shape_list[1:])
            data[new_key] = tf.concat([data[new_key], pad_zeros], 0)

        return data

    def slice_concat_12_ol(self, data, curr_key, new_key):
        slice0 = tf.strided_slice( data[curr_key], [0,0], [0,0], [3, 1], end_mask = 3)
        slice1 = tf.strided_slice( data[curr_key], [1,0], [0,0], [3, 1], end_mask = 3)
        slice2 = tf.strided_slice( data[curr_key], [2,0], [0,0], [3, 1], end_mask = 3)
        data[new_key] = tf.concat([slice0, slice1, slice2], 1)

        slice0_ = tf.strided_slice( data[curr_key], [0,0], [0,0], [4, 1], end_mask = 3)
        slice1_ = tf.strided_slice( data[curr_key], [1,0], [0,0], [4, 1], end_mask = 3)
        slice2_ = tf.strided_slice( data[curr_key], [2,0], [0,0], [4, 1], end_mask = 3)
        slice3_ = tf.strided_slice( data[curr_key], [3,0], [0,0], [4, 1], end_mask = 3)
        data[new_key] = tf.concat([slice0_, slice1_, slice2_, slice3_], 1) 
        return data

    def spatial_slice_concat(self, data, curr_key, new_key):
        shape_now = data[curr_key].get_shape().as_list()
        slice0 = tf.slice(data[curr_key], [0, 0, 0, 0, 0], [-1, -1, 5, -1, -1])
        slice1 = tf.slice(data[curr_key], [0, 0, 5, 0, 0], [-1, -1, 6, -1, -1])
        slice2 = tf.slice(data[curr_key], [0, 0, 11, 0, 0], [-1, -1, 14, -1, -1])
        slice3 = tf.slice(data[curr_key], [0, 0, 25, 0, 0], [-1, -1, 6, -1, -1])

        pad_ten0 = tf.zeros([shape_now[0], shape_now[1], 1, shape_now[3], shape_now[4]])
        pad_ten1 = tf.zeros([shape_now[0], shape_now[1], 1, shape_now[3], shape_now[4]])
        pad_ten2 = tf.zeros([shape_now[0], shape_now[1], 1, shape_now[3], shape_now[4]])
        pad_ten3 = tf.zeros([shape_now[0], shape_now[1], 1, shape_now[3], shape_now[4]])

        data[new_key] = tf.concat([slice0, pad_ten0, pad_ten1, slice1, pad_ten2, slice2, pad_ten3, slice3], 2)
        #data[new_key] = tf.concat([slice0, slice1, slice2, slice3], 2)

        return data

    def slice_label(self, data, curr_key, new_key):
        #print(data[curr_key].get_shape().as_list())
        slice0 = tf.strided_slice( data[curr_key], [0], [0], [3], end_mask = 1)

        if self.num_fake>0:
            shape_list = slice0.get_shape().as_list()
            pad_zeros = tf.zeros([shape_list[0]*self.num_fake] + shape_list[1:], dtype = slice0.dtype)
            slice0 = tf.concat([slice0, pad_zeros], 0)

        data[new_key] = slice0

        return data

    def slice_label_12(self, data, curr_key, new_key):
        #print(data[curr_key].get_shape().as_list())
        slice0 = tf.strided_slice( data[curr_key], [0], [0], [12], end_mask = 1)
        data[new_key] = slice0

        return data

    def normalize_data(self, data, curr_key):
        stat_dict = cPickle.load(open(self.stat_path[curr_key], 'r'))
        mean_tf = tf.constant(stat_dict['mean'], dtype = data[curr_key].dtype)
        var_tf = tf.constant(stat_dict['std'], dtype = data[curr_key].dtype)

        data[curr_key] = tf.multiply(tf.divide(tf.subtract(data[curr_key], mean_tf), var_tf), tf.constant(self.norm_std, dtype = data[curr_key].dtype))

        return data

    def init_ops(self):
        self.input_ops = super(WhiskerWorld, self).init_ops()

        # make sure batch size shapes of tensors are set
        self.input_ops = self.set_data_shapes(self.input_ops)

        for i in range(len(self.input_ops)):
            if not self.split_12:
                self.input_ops[i] = self.slice_concat(self.input_ops[i], 'Data_force', 'Data_force')
            else:
                self.input_ops[i] = self.slice_concat_12(self.input_ops[i], 'Data_force', 'Data_force')

            if self.expand_spatial:
                self.input_ops[i] = self.spatial_slice_concat(self.input_ops[i], 'Data_force', 'Data_force')

            if self.norm_flag:
                self.input_ops[i] = self.normalize_data(self.input_ops[i], 'Data_force')

            if not self.split_12:
                self.input_ops[i] = self.slice_concat(self.input_ops[i], 'Data_torque', 'Data_torque')
            else:
                self.input_ops[i] = self.slice_concat_12(self.input_ops[i], 'Data_torque', 'Data_torque')
            if self.expand_spatial:
                self.input_ops[i] = self.spatial_slice_concat(self.input_ops[i], 'Data_torque', 'Data_torque')
            if self.norm_flag:
                self.input_ops[i] = self.normalize_data(self.input_ops[i], 'Data_torque')

            if not self.split_12:
                self.input_ops[i] = self.slice_label(self.input_ops[i], 'category', 'category')
            else:
                self.input_ops[i] = self.slice_label_12(self.input_ops[i], 'category', 'category')

            if self.otherlabels:
                for other_label in self.otherlabel_list:
                    if not other_label=='objid':
                        if not self.split_12:
                            self.input_ops[i] = self.slice_concat_ol(self.input_ops[i], other_label, other_label)
                        else:
                            self.input_ops[i] = self.slice_concat_12_ol(self.input_ops[i], other_label, other_label)
                    else:
                        if not self.split_12:
                            self.input_ops[i] = self.slice_label(self.input_ops[i], other_label, other_label)
                        else:
                            self.input_ops[i] = self.slice_label_12(self.input_ops[i], other_label, other_label)

        return self.input_ops

    def postprocess_images(self, ims):
        def _postprocess_images(im):
            im = tf.decode_raw(im, np.float32)
            im = tf.reshape(im, [110, 31, 3, 3])
            return im
        return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)

    def postprocess_arrs_3(self, ims):
        def _postprocess_arrs(im):
            im = tf.decode_raw(im, np.float32)
            im = tf.reshape(im, [3])
            return im
        return tf.map_fn(lambda im: _postprocess_arrs(im), ims, dtype=tf.float32)

    def postprocess_arrs_4(self, ims):
        def _postprocess_arrs(im):
            im = tf.decode_raw(im, np.float32)
            im = tf.reshape(im, [4])
            return im
        return tf.map_fn(lambda im: _postprocess_arrs(im), ims, dtype=tf.float32)

# Change to a combined version
#key_list = ['fc_add', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']
key_list_default = ['fc_add', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']

def add_layer_list(subnet, now_list, layer_offset = 0):
    len_key = len(subnet.keys())
    for indx_tmp in xrange(len_key):
        now_layer = subnet['l%i' % indx_tmp]
        if 'conv' in now_layer:
            now_list.append('conv%i' % (indx_tmp + layer_offset + 1))
        else:
            now_list.append('fc%i' % (indx_tmp + layer_offset + 1))

    return now_list

def add_fc_add_list(addnet, now_list):
    for indx_add in xrange(len(addnet.keys()) - 1):
        now_list.append('fc_add%i' % (indx_add + 1))
    return now_list

def has_fc(now_net):
    for layer_now in now_net:
        if 'fc' in now_net[layer_now]:
            return True
    return False

def get_layer_list(cfg_now):
    if 'nodes' in cfg_now:
        default_list = ['fc_add', 'fc8', 'fc7', 'conv6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']
        if not 'addnet' in cfg_now:
            return default_list, True
        else:
            return add_fc_add_list(cfg_now['addnet'], default_list), True
    else:
        new_list = []
        if 'subnet' in cfg_now:
            new_list = add_layer_list(cfg_now['subnet'], new_list, len(new_list))
        else:
            net_list = ['spanet', 'tempnet']
            if has_fc(cfg_now['spanet']):
                net_list = ['tempnet', 'spanet']

            for net_now_name in net_list:
                new_list =  add_layer_list(cfg_now[net_now_name], new_list, len(new_list))

        new_list.append('fc_add')
        if 'addnet' in cfg_now:
            new_list = add_fc_add_list(cfg_now['addnet'], new_list)

        return new_list, False

def get_size(curr_tensor):
    shape_list = curr_tensor.get_shape().as_list()

    curr_size = 1
    for now_indx, now_shape in enumerate(shape_list[1:]):
        curr_size = now_shape*curr_size

    return curr_size

def save_features(inputs, outputs, key_list = key_list_default, special_set = False, only_labels = False, just_count = False, firstswipe = 0):

    ret_dict = {}
    ret_dict['label'] = inputs['category']

    if only_labels:
        for other_label in OTHER_LABELS_LIST + ['Data_force', 'Data_torque']:
            if other_label in inputs:
                ret_dict[other_label] = inputs[other_label]
        print(ret_dict)
        return ret_dict

    # Output test code
    #all_name_list = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #all_name_list = filter(lambda name_now: 'validation/topn' in name_now, all_name_list)
    #all_name_list = filter(lambda name_now: 'fc8' in name_now, all_name_list)
    #all_name_list = filter(lambda name_now: 'output' in name_now, all_name_list)

    #print(all_name_list)
    #print(len(all_name_list))

    if just_count:
        num_units = 0

    for target in key_list:

        divide_num = 1

        all_name_list = [n.name for n in tf.get_default_graph().as_graph_def().node]

        all_name_list = filter(lambda name_now: 'validation/topn' in name_now, all_name_list)
        all_name_list = filter(lambda name_now: (('%s/' % target) in name_now) or (('%s_' % target) in name_now), all_name_list)
        if (not special_set) or (target.startswith('fc_add')):
            #print(target, all_name_list)
            tmp_name_list = filter(lambda name_now: 'pool' in name_now, all_name_list)


            if len(tmp_name_list) > 0 and not just_count:
                all_name_list = tmp_name_list

            else:
                if just_count and len(tmp_name_list) > 0:
                    tensor_now = tf.get_default_graph().get_tensor_by_name("%s:0" % tmp_name_list[0])
                    print(target, tensor_now.name, tensor_now.get_shape().as_list())
                    num_units  = num_units + get_size(tensor_now)

                tmp_name_list = filter(lambda name_now: 'relu' in name_now, all_name_list)
                if (len(tmp_name_list) > 0) and (not target=='fc_add'):
                    all_name_list = tmp_name_list
                else:
                    tmp_name_list = filter(lambda name_now: name_now.endswith('/fc'), all_name_list)
                    all_name_list = tmp_name_list
        else:
            tmp_name_list = filter(lambda name_now: 'output' in name_now, all_name_list)
            all_name_list = tmp_name_list

        output_now_tmp = [tf.get_default_graph().get_tensor_by_name("%s:0" % tmp_name) for tmp_name in all_name_list]

        if target=='fc_add':
            output_now_tmp2 = []
            for v in output_now_tmp:
                #print(v.get_shape().as_list())
                if (v.get_shape().as_list()[1]==117):
                    output_now_tmp2.append(v)
            output_now_tmp = output_now_tmp2

        for len_rep in xrange(3):
            gfs_key = '%s_%i' % (target, len_rep)
            ret_dict[gfs_key] = []

        vector_len = len(output_now_tmp)
        vector_indx_iter = xrange(vector_len)
        if firstswipe>0 and 'fc_add' not in target:
            assert vector_len%3==0, "Must have three swipes for %s!" % target
            vector_indx_iter = xrange(vector_len//3*firstswipe)

        for len_have in vector_indx_iter:
            gfs_key = '%s_%i' % (target, len_have)
            ret_dict[gfs_key] = output_now_tmp[len_have]

        if len(output_now_tmp)>0 and just_count:
            print(target, output_now_tmp[0].name, output_now_tmp[0].get_shape().as_list())
            num_units = num_units + get_size(output_now_tmp[0])

    if just_count:
        print(num_units)
        exit()

    if special_set:
        print(len(ret_dict))
        #print([(v, ret_dict[v]) for v in ret_dict if v.startswith('fc_add')])

        total_parameters = 0
        for variable in ret_dict:
            if isinstance(ret_dict[variable], list):
                continue
            shape = ret_dict[variable].get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print(total_parameters)
    else:
        print(ret_dict)
    return ret_dict

def mean_losses_keep_rest(step_results):
    retval = {}
    keys = step_results[0].keys()
    for k in keys:
        plucked = [d[k] for d in step_results]
        if 'loss' in k:
            retval[k] = np.mean(plucked)
        else:
            retval[k] = plucked
    return retval

def get_params_from_arg(args):

    pathconfig = args.pathconfig
    if not os.path.isfile(pathconfig):
        pathconfig = os.path.join('network_cfgs', pathconfig)

    assert os.path.isfile(pathconfig), "%s not existing!" % args.pathconfig

    cfg_initial = json.load(open(pathconfig))
    #print(cfg_initial)

    exp_id  = args.expId
    cache_dir = os.path.join(args.cacheDirPrefix, '.tfutils', 'localhost:'+ str(args.nport), 'normalnet-test', 'normalnet', exp_id)

    if "BATCH_SIZE" in cfg_initial:
        BATCH_SIZE  = cfg_initial["BATCH_SIZE"]
    else:
        BATCH_SIZE  = 384

    if "QUEUE_CAP" in cfg_initial:
        queue_capa  = cfg_initial["QUEUE_CAP"]
    else:
        queue_capa  = 3840

    if args.test_queuecap==0:
        queue_capa = queue_capa*50;

    n_threads   = 4

    func_net = getattr(cate_network_builder, args.namefunc)

    train_data_param = {
                'func': WhiskerWorld,
                'data_path': DATA_PATH,
                'group': 'train',
                'n_threads': n_threads,
                #'batch_size': BATCH_SIZE,
                'batch_size': 12,
                'num_fake': args.num_fake,
            }
    val_data_param = {
                    'func': WhiskerWorld,
                    'data_path': DATA_PATH,
                    'group': 'val',
                    'n_threads': n_threads,
                    #'batch_size': BATCH_SIZE,
                    'batch_size': 12,
                }
    
    if args.expand==1 and args.parallel==0:
        train_data_param['expand_spatial'] = True
        val_data_param['expand_spatial'] = True

    if args.norm==1:
        train_data_param['norm_flag'] = True
        val_data_param['norm_flag'] = True

        train_data_param['norm_std'] = args.norm_std
        val_data_param['norm_std'] = args.norm_std

    train_queue_params = {
            'queue_type': 'random',
            'batch_size': BATCH_SIZE//3,
            'seed': 0,
            'capacity': queue_capa//3,
        }
    val_queue_params = {
                'queue_type': 'fifo',
                'batch_size': BATCH_SIZE//3,
                'seed': 0,
                'capacity': BATCH_SIZE*10//3,
            }
    val_target          = 'category'

    val_step_num = 12*2*9981//BATCH_SIZE
    NUM_BATCHES_PER_EPOCH = 12*24*9981//BATCH_SIZE

    if args.valinum>-1:
        val_step_num = args.valinum

    loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits
    learning_rate_params = {
            'func': tf.train.exponential_decay,
            'learning_rate': args.initlr,
            #'decay_rate': .5,
            'decay_rate': 1,
            'decay_steps': NUM_BATCHES_PER_EPOCH*10,  # exponential decay each epoch
            'staircase': True
        }

    optimizer_class = tf.train.MomentumOptimizer

    model_params = {
            'func': func_net,
            'seed': args.seed,
            'cfg_initial': cfg_initial
        }

    if args.split12==1:
        model_params['split_12'] = True
        train_data_param['split_12'] = True
        val_data_param['split_12'] = True
        train_queue_params['batch_size'] = BATCH_SIZE//12
        val_queue_params['batch_size'] = BATCH_SIZE//12
        train_queue_params['capacity'] = queue_capa//12
        val_queue_params['capacity'] = BATCH_SIZE*10//12

    if args.newdata==1:
        train_data_param['data_path'] = NEW_DATA_PATH
        val_data_param['data_path'] = NEW_DATA_PATH

        if args.valbycat==0:
            train_data_param['patprefix'] = 's'
            val_data_param['patprefix'] = 's'
        else:
            train_data_param['patprefix'] = 'c'
            val_data_param['patprefix'] = 'c'

        if args.otherlabels==1:
            train_data_param['otherlabels'] = True
            val_data_param['otherlabels'] = True

    if args.tnn==1:
        model_params['cfg_path'] = pathconfig
        model_params['tnndecay'] = args.tnndecay
        model_params['decaytrain'] = args.decaytrain
        model_params['cmu'] = args.cmu

    if args.parallel==1:
        model_params['model_func'] = model_params['func']
        model_params['func'] = cate_network_builder.parallel_net_builder
        model_params['n_gpus'] = len(args.gpu.split(','))
        if args.inputthre>0:
            model_params['inputthre'] = args.inputthre

        if args.expand==1:
            model_params['expand'] = 1

    if args.fixweights==1:
        model_params['fixweights'] = True

    optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': optimizer_class,
            'clip': True,
            'momentum': .9
        }

    if args.whichopt==1:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdamOptimizer,
            'clip': True,
        }

    if args.whichopt==2:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdagradOptimizer,
            'clip': True,
        }

    if args.whichopt==3:
        optimizer_params = {
                'func': optimizer.ClipOptimizer,
                'optimizer_class': optimizer_class,
                'clip': True,
                'momentum': .9,
                'use_nesterov': True
            }

    if args.whichopt==4:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdadeltaOptimizer,
            'clip': True,
        }

    if args.whichopt==5:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.RMSPropOptimizer,
            'clip': True,
        }

    if args.parallel==1:
        optimizer_params['func'] = ParallelClipOptimizer

    load_query = None
    load_params = {
            'host': 'localhost',
            'port': args.nport,
            'dbname': 'whisker_net',
            'collname': 'catenet',
            'exp_id': exp_id,
            'do_restore': True,
            'query': load_query 
    }

    if args.loadque==1:
        load_query = {'saved_filters': True, 'step': 70000}
        load_params = {
                'host': 'localhost',
                'port': args.nport,
                'dbname': 'whisker_net',
                'collname': 'catenet',
                'exp_id': 'catenet_adag_flv_slac_2',
                'do_restore': True,
                'query': load_query 
        }
        #print(load_query)

    if args.loadque==2:
        load_query = {'saved_filters': True, 'step': 59880}
        load_params = {
                'host': 'localhost',
                'port': args.nport,
                'dbname': 'whisker_net',
                'collname': 'catenet',
                'exp_id': 'catenet_tnn_decay_train_sep22_np',
                'do_restore': True,
                'query': load_query 
        }

    if args.loadque==3:
        if args.loadname is None:
            args.loadname = args.expId

        load_params = {
                'host': 'localhost',
                'port': args.loadport,
                'dbname': 'whisker_net',
                'collname': 'catenet',
                'exp_id': args.loadname,
                'do_restore': True,
                'query': load_query 
        }

    save_params = {
            'host': 'localhost',
            'port': args.nport,
            'dbname': 'whisker_net',
            'collname': 'catenet',
            'exp_id': exp_id,

            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': 2000,  # keeps loss from every SAVE_LOSS_FREQ steps.
            #'save_valid_freq': 5000,
            #'save_filters_freq': 5000,
            #'cache_filters_freq': 5000,
            'save_valid_freq':  NUM_BATCHES_PER_EPOCH*args.valid_time,
            'save_filters_freq': NUM_BATCHES_PER_EPOCH,
            'cache_filters_freq': NUM_BATCHES_PER_EPOCH,
            'cache_dir': cache_dir,
        }

    train_params = {
            'validate_first': False,
            'data_params': train_data_param,
            'queue_params': train_queue_params,
            #'thres_loss': 1000000000,
            'thres_loss': np.finfo(np.float32).max,
            #'num_steps': 90 * NUM_BATCHES_PER_EPOCH  # number of steps to train
            'num_steps': 120 * NUM_BATCHES_PER_EPOCH  # number of steps to train
        }

    loss_params = {
            'targets': val_target,
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_func,
        }

    if args.parallel==1 and args.cmu==0:
        loss_params['agg_func'] = parallel_reduce_mean
        loss_params['loss_per_case_func'] = parallel_softmax_cross_entropy_loss

    if args.parallel==1 and args.cmu==1:
        loss_params['agg_func'] = parallel_reduce_mean
        loss_params['loss_per_case_func'] = cmu_parallel_softmax_cross_entropy_loss

    if args.parallel==0 and args.cmu==1:
        loss_params['loss_per_case_func'] = cmu_softmax_cross_entropy_loss

    validation_params = {
            'topn': {
                'data_params': val_data_param,
                'queue_params': val_queue_params,
                'targets': {
                    'func': in_top_k,
                    'target': val_target,
                },
                'num_steps': val_step_num,
                'agg_func': lambda x: {k:np.mean(v) for k,v in x.items()},
                'online_agg_func': online_agg
            }
        }

    if args.parallel==1 and args.cmu==0:
        validation_params['topn']['targets']['func'] = parallel_in_top_k

    if args.parallel==1 and args.cmu==1:
        validation_params['topn']['targets']['func'] = cmu_parallel_in_top_k

    if args.parallel==0 and args.cmu==1:
        validation_params['topn']['targets']['func'] = cmu_in_top_k

    if args.no_valid==1:
        validation_params = {}

    if args.gen_feature==1:
        train_params['validate_first'] = True
        train_params['num_steps'] = 305005
        val_data_param['n_threads'] = 1

        key_list = key_list_default
        special_set = False

        if args.whichtype==1:
            key_list = ['fc_add', 'fc_add1', 'fc12', 'fc11', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10', 
                    'conv5', 'conv4', 'conv3', 'conv2', 'conv1']

        if args.whichtype==-1:
            key_list = ['fc_add']

        if args.whichtype==2:
            key_list = ['fc_add', 'fc_add1', 'fc_add2', 'fc8', 'fc7', 'conv6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']
            special_set = True

        if args.whichtype==-2:
            key_list, special_set = get_layer_list(cfg_initial)

        if args.whichpart>0:
            new_key_list = []
            for indx_which, item in enumerate(key_list):
                if indx_which % args.partnum==args.whichpart % args.partnum:
                    new_key_list.append(key_list[indx_which])

            key_list = new_key_list

        onlylabels = False

        if args.onlylabels==1:
            onlylabels = True

        just_count = False
        if args.justcount==1:
            just_count = True

        validation_params['topn']['targets'] = {
                'func': save_features,
                'key_list': key_list,
                'special_set': special_set,
                'only_labels': onlylabels,
                'just_count': just_count,
                'firstswipe': args.firstswipe,
            }
        validation_params['topn']['online_agg_func'] = online_agg_genfeautre
        validation_params['topn']['num_steps'] = 10
        val_queue_params['capacity'] = val_queue_params['batch_size'] 

        save_to_gfs = ['label']
        for key_now in key_list_default:
            for len_rep in xrange(3):
                gfs_key = '%s_%i' % (key_now, len_rep)
                save_to_gfs.append(gfs_key)

        print(save_to_gfs)
        save_params['save_to_gfs'] = save_to_gfs
        save_params['save_valid_freq'] = 305004
        save_params['save_intermediate_freq'] = 1
        save_params['port'] = 27017
        save_params = {'exp_id': exp_id,
                       'save_intermediate_freq': 1,
                       'save_to_gfs': save_to_gfs}

        if not args.loadstep is None:
            load_query = {'saved_filters': True, 'step': args.loadstep}
            load_params['query'] = load_query
            if not args.loadname is None:
                load_params['query']['exp_id'] = args.loadname

        '''
        #load_query = {'saved_filters': True, 'step': 305000}
        load_params = {
                'host': 'localhost',
                'port': args.nport,
                'dbname': 'whisker_net',
                'collname': 'catenet',
                'exp_id': 'catenet_adag_flv_slac_3',
                'do_restore': True,
                'query': load_query 
        }
        '''
        pass

    if args.gen_feature==0:
        if args.test_mult==0:
            params = {
                'save_params': save_params,

                'load_params': load_params,

                'model_params': model_params,

                'train_params': train_params,

                'loss_params': loss_params,

                'learning_rate_params': learning_rate_params,

                'optimizer_params': optimizer_params,

                'log_device_placement': False,  # if variable placement has to be logged
                'validation_params': validation_params,
            }
        else:
            save_params_2 = copy.deepcopy(save_params)
            save_params_2['exp_id'] = exp_id + '_2'
            save_params = [save_params, save_params_2]

            load_params_2 = copy.deepcopy(load_params)
            load_params_2['exp_id'] = exp_id + '_2'
            load_params = [load_params, load_params_2]

            model_params['n_gpus'] = 1
            model_params_2 = copy.deepcopy(model_params)
            model_params_2['gpu_offset'] = 1
            model_params = [model_params, model_params_2]

            loss_params_2 = copy.deepcopy(loss_params)
            loss_params_2['loss_func_kwargs'] = {'gpu_offset': 1}
            loss_params = [loss_params, loss_params_2]

            learning_rate_params_2 = copy.deepcopy(learning_rate_params)
            learning_rate_params = [learning_rate_params, learning_rate_params_2]

            optimizer_params_2 = copy.deepcopy(optimizer_params)
            optimizer_params_2['gpu_offset'] = 1
            optimizer_params = [optimizer_params, optimizer_params_2]

            validation_params_2 = copy.deepcopy(validation_params)
            validation_params = [validation_params, validation_params_2]
            validation_params = [{},{}]

            params = {
                'save_params': save_params,

                'load_params': load_params,

                'model_params': model_params,

                'train_params': train_params,

                'loss_params': loss_params,

                'learning_rate_params': learning_rate_params,

                'optimizer_params': optimizer_params,

                'log_device_placement': False,  # if variable placement has to be logged
                'validation_params': validation_params,
            }
        #base.train_from_params(**params)

        return params

    else:
        if args.parallel==1:
            if args.loadname is None:
                model_params['with_modelprefix'] = args.expId
            else:
                model_params['with_modelprefix'] = args.loadname
            pass

        params = {
            'load_params': load_params,
            'model_params': model_params,
            'validation_params': validation_params,
            'log_device_placement': False,  # if variable placement has to be logged
            'save_params': save_params,
            'dont_run': True,
        }
        #base.test_from_params(**params)
        sess, queues, dbinterface, valid_targets_dict = base.test_from_params(**params)
        print(valid_targets_dict.keys())
        coord, threads = base.start_queues(sess)

        fout = h5py.File(args.hdf5path, 'w')
        over_num = 9981*2*4
        #over_num = 256
        #over_num = 1280
        now_num = 0
        for indx_tmp in xrange(val_step_num + 1):
            start_time = time.time()
            res = sess.run(valid_targets_dict['topn']['targets'])
            #print(res.keys())
            end_num = min(now_num + res['label'].size, over_num)

            for key_tmp in res:
                #print(res[key_tmp].shape)
                now_data = res[key_tmp]
                if isinstance(now_data, list):
                    continue
                if key_tmp not in fout:
                    new_shape = list(now_data.shape)
                    new_shape[0] = over_num
                    dataset_tmp = fout.create_dataset(key_tmp, new_shape, dtype='f')
                else:
                    dataset_tmp = fout[key_tmp]
                dataset_tmp[now_num:end_num] = now_data[:(end_num - now_num)]
                
            #print(res['label'])
            now_num = end_num
            end_time = time.time()
            print('Batch %i takes time %f' % (indx_tmp, end_time - start_time))

        base.stop_queues(sess, queues, coord, threads)
        fout.close()
        sess.close()
    return None

def main():
    parser = argparse.ArgumentParser(description='The script to train the catenet for barrel')
    # System setting
    parser.add_argument('--gpu', default = '0', type = str, action = 'store', help = 'Index of gpu, currently only one gpu is allowed')

    # General setting
    parser.add_argument('--nport', default = 29101, type = int, action = 'store', help = 'Port number of mongodb')
    parser.add_argument('--pathconfig', default = "catenet_config.cfg", type = str, action = 'store', help = 'Path to config file')
    parser.add_argument('--expId', default = "catenet", type = str, action = 'store', help = 'Name of experiment id')
    #parser.add_argument('--expId', default = [], type = str, action = 'append', help = 'Name of experiment id')
    parser.add_argument('--seed', default = 0, type = int, action = 'store', help = 'Random seed for model')
    parser.add_argument('--cacheDirPrefix', default = "/media/data2/chengxuz", type = str, action = 'store', help = 'Prefix of cache directory')
    parser.add_argument('--namefunc', default = "catenet_tfutils", type = str, action = 'store', help = 'Name of function to build the network')
    parser.add_argument('--valinum', default = -1, type = int, action = 'store', help = 'Number of validation steps, default is -1, which means all the validation')
    parser.add_argument('--whichopt', default = 0, type = int, action = 'store', help = 'Choice of the optimizer, 0 means momentum, 1 means Adam')
    parser.add_argument('--initlr', default = 0.0001, type = float, action = 'store', help = 'Initial learning rate')
    parser.add_argument('--expand', default = 0, type = int, action = 'store', help = 'Whether do the spatial padding')
    parser.add_argument('--split12', default = 0, type = int, action = 'store', help = 'Whether do the 12 swipes spliting, default is no')
    parser.add_argument('--norm', default = 0, type = int, action = 'store', help = 'Whether do the normalization, default is no')
    parser.add_argument('--norm_std', default = 1, type = float, action = 'store', help = 'Std of new input, default is 1')
    parser.add_argument('--parallel', default = 0, type = int, action = 'store', help = 'Whether to do parallel across gpus, default is no (0)')
    parser.add_argument('--inputthre', default = 0, type = float, action = 'store', help = 'Threshold to control the input')
    parser.add_argument('--fixweights', default = 0, type = float, action = 'store', help = 'Threshold to control the input')

    # TNN related parameters
    parser.add_argument('--tnn', default = 0, type = int, action = 'store', help = 'Whether to use the tnn, default is no')
    parser.add_argument('--tnndecay', default = 0.1, type = float, action = 'store', help = 'Memory decay for tnn each layer')
    parser.add_argument('--decaytrain', default = 0, type = int, action = 'store', help = 'Whether the decay is trainable')
    parser.add_argument('--cmu', default = 0, type = int, action = 'store', help = 'Whether do cumulative loss')

    # Feature extraction related parameters
    parser.add_argument('--gen_feature', default = 0, type = int, action = 'store', help = 'Whether to generate features, default is 0, None')
    parser.add_argument('--hdf5path', default = "/mnt/fs1/chengxuz/barrel_response/response.hdf5", type = str, action = 'store', help = 'Where to save the output')
    parser.add_argument('--whichtype', default = 0, type = int, action = 'store', help = 'Which type of network, this will decide the key_list used')
    parser.add_argument('--whichpart', default = 0, type = int, action = 'store', help = 'Which part of the list, this will decide which part of the key_list used')
    parser.add_argument('--partnum', default = 2, type = int, action = 'store', help = 'Number of parts of the list, used when whichpart is larger than 0')
    parser.add_argument('--onlylabels', default = 0, type = int, action = 'store', help = 'Default is 0, not reporting only labels')
    parser.add_argument('--justcount', default = 0, type = int, action = 'store', help = 'Default is 0, not just counting number of units')
    parser.add_argument('--firstswipe', default = 0, type = int, action = 'store', help = 'Default is 0, not just extracting the first swipe')
    parser.add_argument('--loadstep', default = None, type = int, action = 'store', help = 'Specify the step to load')

    # Test parameters
    parser.add_argument('--num_fake', default = 0, type = int, action = 'store', help = 'Default is 0, no fake')
    parser.add_argument('--test_mult', default = 0, type = int, action = 'store', help = 'Default is 0, no multi')
    parser.add_argument('--test_queuecap', default = 0, type = int, action = 'store', help = 'Default is 0, using default queue capacity')
    parser.add_argument('--no_valid', default = 0, type = int, action = 'store', help = 'Default is 0, with validation')
    parser.add_argument('--valid_time', default = 1, type = int, action = 'store', help = 'Default is 1, every epoch')

    # Parameters for Multiple networks
    parser.add_argument('--innerargs', default = [], type = str, action = 'append', help = 'Arguments for every network')
    parser.add_argument('--gpu_offset', default = [], type = int, action = 'append', help = 'GPU offset for every network')
    parser.add_argument('--ngpus', default = [], type = int, action = 'append', help = 'Number of gpus for every network')

    # New dataset related parameters
    parser.add_argument('--newdata', default = 0, type = int, action = 'store', help = 'Default is 0, use old dataset')
    parser.add_argument('--valbycat', default = 0, type = int, action = 'store', help = 'Default is 0, use original validation splitting')
    parser.add_argument('--otherlabels', default = 0, type = int, action = 'store', help = 'Default is 0, not including other labels')

    # Load old results related parameters
    parser.add_argument('--loadque', default = 0, type = int, action = 'store', help = 'Special setting for load query')
    parser.add_argument('--loadname', default = None, type = str, action = 'store', help = 'While loadque is 3, will use this as load experiment id name')
    parser.add_argument('--loadport', default = 29101, type = int, action = 'store', help = 'While loadque is 3, will use this as load port number')

    args    = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if len(args.innerargs)==0:
        params = get_params_from_arg(args)

        if not params is None:
            base.train_from_params(**params)
    else:
        params = {
                'save_params': [],

                'load_params': [],

                'model_params': [],

                'train_params': None,

                'loss_params': [],

                'learning_rate_params': [],

                'optimizer_params': [],

                'log_device_placement': False,  # if variable placement has to be logged
                'validation_params': [],
            }

        list_names = ["save_params", "load_params", "model_params", "validation_params", "loss_params", "learning_rate_params", "optimizer_params"]
        
        assert len(args.innerargs)==len(args.gpu_offset)==len(args.ngpus), "Three lists must be the same length"
        for curr_arg, curr_gpu_offset, curr_ngpus in zip(args.innerargs, args.gpu_offset, args.ngpus):
            args = parser.parse_args(curr_arg.split())
            curr_params = get_params_from_arg(args)

            curr_params['model_params']['n_gpus'] = curr_ngpus
            curr_params['model_params']['gpu_offset'] = curr_gpu_offset 
            curr_params['loss_params']['loss_func_kwargs'] = {'gpu_offset': curr_gpu_offset }
            curr_params['optimizer_params']['gpu_offset'] = curr_gpu_offset
            
            for tmp_key in list_names:
                params[tmp_key].append(curr_params[tmp_key])

            params['train_params'] = curr_params['train_params']

        base.train_from_params(**params)

if __name__ == '__main__':
    main()
