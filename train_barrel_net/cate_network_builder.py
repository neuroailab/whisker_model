from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tfutils import model as model_tfutils
import model

from tnn import main
from tnn.cell import *

from collections import OrderedDict
from tensorflow.contrib import rnn

def getWhetherConv(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]
    ret_val = "conv" in tmp_dict
    return ret_val

def getConvFilterSize(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["conv"]
    if "filter_size1" in tmp_dict:
        return (tmp_dict["filter_size1"], tmp_dict["filter_size2"])
    else:
        return tmp_dict["filter_size"]

def getConvFilterSize3D(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["conv"]
    if "filter_size1" in tmp_dict:
        return (tmp_dict["filter_size1"], tmp_dict["filter_size2"], tmp_dict["filter_size3"])
    else:
        return tmp_dict["filter_size"]

def getConvStride(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["conv"]
    if "stride1" in tmp_dict:
        return (tmp_dict["stride1"], tmp_dict["stride2"])
    else:
        return tmp_dict["stride"]

def getConvStride3D(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["conv"]
    if "stride1" in tmp_dict:
        return (tmp_dict["stride1"], tmp_dict["stride2"], tmp_dict["stride3"])
    else:
        return tmp_dict["stride"]

def getConvNumFilters(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["conv"]
    return tmp_dict["num_filters"]

def getWhetherPool(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]
    ret_val = "pool" in tmp_dict
    return ret_val

def getPoolFilterSize(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["pool"]
    if "filter_size1" in tmp_dict:
        return (tmp_dict["filter_size1"], tmp_dict["filter_size2"])
    else:
        return tmp_dict["filter_size"]

def getPoolFilterSize3D(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["pool"]
    if "filter_size1" in tmp_dict:
        return (tmp_dict["filter_size1"], tmp_dict["filter_size2"], tmp_dict["filter_size3"])
    else:
        return tmp_dict["filter_size"]

def getPoolStride(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["pool"]
    if "stride1" in tmp_dict:
        return (tmp_dict["stride1"], tmp_dict["stride2"])
    else:
        return tmp_dict["stride"]

def getPoolStride3D(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["pool"]
    if "stride1" in tmp_dict:
        return (tmp_dict["stride1"], tmp_dict["stride2"], tmp_dict["stride3"])
    else:
        return tmp_dict["stride"]

def getFcNumFilters(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["fc"]
    return tmp_dict["num_features"]

def getWhetherLrn(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]
    return 'lrn' in tmp_dict

def getLrnDepth(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]
    return tmp_dict['lrn']

def getWhetherBn(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]
    return 'bn' in tmp_dict

def getBnMode(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]
    return tmp_dict['bn']

def group_in2d(small_inputs, padding_len, row_num = 11, col_num = 10):
    shape_now = small_inputs[0].get_shape().as_list()

    if padding_len>0:
        pad_right = tf.zeros([shape_now[0], padding_len, shape_now[2], shape_now[3]])
        pad_up = tf.zeros([shape_now[0], shape_now[1] + padding_len, padding_len, shape_now[3]])

    indx_now = 0

    tensor_rows = []

    for indx_row in xrange(row_num):

        tensor_cols = []
        for indx_col in xrange(col_num):
            small_input = small_inputs[indx_now]
            small_input = tf.concat([small_input, pad_right], 1)
            small_input = tf.concat([small_input, pad_up], 2)
            tensor_cols.append(small_input)
            indx_now = indx_now + 1
        tensor_curr_row = tf.concat(tensor_cols, 2)
        tensor_rows.append(tensor_curr_row)

    final_input = tf.concat(tensor_rows, 1)

    return final_input

def degroup_from2d(whole_output, padding_len, row_num = 11, col_num = 10):
    shape_now = whole_output.get_shape().as_list()

    assert shape_now[1]%row_num==0, 'Use "SAME" for padding!'
    assert shape_now[2]%col_num==0, 'Use "SAME" for padding!'
    small_shape = [shape_now[1]//row_num, shape_now[2]//col_num]

    small_inputs = []
    all_rows = tf.split(whole_output, row_num, 1)

    for indx_row in xrange(row_num):
        curr_row_tensor = all_rows[indx_row]
        all_cols_curr_row = tf.split(curr_row_tensor, col_num, 2)

        for indx_col in xrange(col_num):
            curr_col_row_tensor = all_cols_curr_row[indx_col]

            small_input = tf.slice(curr_col_row_tensor, [0, 0, 0, 0], [-1, small_shape[0] - padding_len, small_shape[1] - padding_len, -1])

            small_inputs.append(small_input)

    return small_inputs

class tnn_LSTMCell(rnn.RNNCell):

    def __init__(self,
                 harbor_shape,
                 harbor=(harbor, None),
                 pre_memory=None,
                 memory=(memory, None),
                 post_memory=None,
                 input_init=(tf.zeros, None),
                 state_init=(tf.zeros, None),
                 dtype=tf.float32,
                 name=None
                 ):

        self.harbor_shape = harbor_shape
        self.harbor = harbor if harbor[1] is not None else (harbor[0], {})
        self.pre_memory = pre_memory
        self.memory = memory if memory[1] is not None else (memory[0], {})
        self.post_memory = post_memory

        self.input_init = input_init if input_init[1] is not None else (input_init[0], {})
        self.state_init = state_init if state_init[1] is not None else (state_init[0], {})

        self.dtype = dtype
        self.name = name

        self._reuse = None

        #self.lstm_cell = rnn.LSTMCell(100, state_is_tuple=False)
        cell_type = 'LSTMCell'
        if 'type' in memory[1]:
            cell_type = memory[1]['type']
        #self.lstm_cell = rnn.LSTMCell(memory[1]['nunits'])
        self.lstm_cell = getattr(rnn, cell_type)(memory[1]['nunits'])

    def __call__(self, inputs=None, state=None):
        """
        Produce outputs given inputs

        If inputs or state are None, they are initialized from scratch.

        :Kwargs:
            - inputs (list)
                A list of inputs. Inputs are combined using the harbor function
            - state

        :Returns:
            (output, state)
        """

        with tf.variable_scope(self.name, reuse=self._reuse):

            if inputs is None:
                inputs = [self.input_init[0](shape=self.harbor_shape,
                                             **self.input_init[1])]
            output = self.harbor[0](inputs, self.harbor_shape, reuse=self._reuse, **self.harbor[1])
       
            pre_name_counter = 0
            for function, kwargs in self.pre_memory:
                with tf.variable_scope("pre_" + str(pre_name_counter), reuse=self._reuse):
                    output = function(output, **kwargs)
                pre_name_counter += 1

            if state is None:
                state = self.lstm_cell.zero_state(output.get_shape().as_list()[0], dtype = self.dtype)

            output, state = self.lstm_cell(output, state)
            self.state = tf.identity(state, name='state')

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    output = function(output, **kwargs)
                post_name_counter += 1
            self.output = tf.identity(tf.cast(output, self.dtype), name='output')

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_shape = self.output.shape
        return self.output, state

    @property
    def state_size(self):
        """
        Size(s) of state(s) used by this cell.

        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        # if self.state is not None:
        return self.state_shape
        # else:
        #     raise ValueError('State not initialized yet')

    @property
    def output_size(self):
        """
        Integer or TensorShape: size of outputs produced by this cell.
        """
        # if self.output is not None:
        return self.output_shape
        # else:
        #     raise ValueError('Output not initialized yet')


def build_partnet_3din2d(m, cfg, key_layernum, key_subnet, inputs=None, layer_offset=0, dropout=None):
    layernum_sub = cfg[ key_layernum ]

    shape_list = inputs.get_shape().as_list()
    small_inputs = tf.split(inputs, shape_list[1], 1)
    
    for indx_input in xrange(len(small_inputs)):
        small_inputs[indx_input] = tf.reshape(small_inputs[indx_input], [shape_list[0], 5, 7, -1])

    for indx_layer in xrange(layernum_sub):
        curr_size = getConvFilterSize(indx_layer, cfg, key_want = key_subnet) 
        input_whole = group_in2d(small_inputs, curr_size - 1)
        m.output = input_whole

        do_conv = getWhetherConv(indx_layer, cfg, key_want = key_subnet)
        if do_conv:
            layer_name = "conv%i" % (1 + indx_layer + layer_offset)
            with tf.variable_scope(layer_name):
                m.conv(getConvNumFilters(indx_layer, cfg, key_want = key_subnet), 
                        getConvFilterSize(indx_layer, cfg, key_want = key_subnet), 
                        getConvStride(indx_layer, cfg, key_want = key_subnet))

                if getWhetherBn(indx_layer, cfg, key_want = key_subnet):
                    m.batchnorm_corr(train)

                do_pool = getWhetherPool(indx_layer, cfg, key_want = key_subnet)
                if do_pool:
                    m.pool(getPoolFilterSize(indx_layer, cfg, key_want = key_subnet), 
                            getPoolStride(indx_layer, cfg, key_want = key_subnet))

        else:
            layer_name = "fc%i" % (1 + indx_layer + layer_offset)
            with tf.variable_scope(layer_name):
                m.fc(getFcNumFilters(indx_layer, cfg, key_want = key_subnet), 
                        init='trunc_norm', dropout=dropout, bias=.1)

                if getWhetherBn(indx_layer, cfg, key_want = key_subnet):
                    m.batchnorm_corr(train)

        small_inputs = degroup_from2d(m.output, curr_size - 1)

    for indx_input in xrange(len(small_inputs)):
        small_inputs[indx_input] = tf.reshape(small_inputs[indx_input], [shape_list[0], 1, 1, -1])
    
    m.output = tf.concat(small_inputs, 1)
    return m

def build_partnet_all3d(m, cfg, key_layernum, key_subnet, inputs=None, layer_offset=0, dropout=None):
    layernum_sub = cfg[ key_layernum ]
    for indx_layer in xrange(layernum_sub):
        do_conv = getWhetherConv(indx_layer, cfg, key_want = key_subnet)
        if do_conv:
            layer_name = "conv%i" % (1 + indx_layer + layer_offset)
            with tf.variable_scope(layer_name):
                curr_size_list = getConvFilterSize3D(indx_layer, cfg, key_want = key_subnet) 

                if indx_layer==0 and (not inputs==None):
                    m.conv3(getConvNumFilters(indx_layer, cfg, key_want = key_subnet), 
                            curr_size_list,
                            getConvStride3D(indx_layer, cfg, key_want = key_subnet),
                           padding='VALID', in_layer = inputs)
                else:
                    m.conv3(getConvNumFilters(indx_layer, cfg, key_want = key_subnet), 
                            curr_size_list, 
                            getConvStride3D(indx_layer, cfg, key_want = key_subnet))

                if getWhetherBn(indx_layer, cfg, key_want = key_subnet):
                    m.batchnorm_corr(train)

                do_pool = getWhetherPool(indx_layer, cfg, key_want = key_subnet)
                if do_pool:
                    m.pool3(getPoolFilterSize3D(indx_layer, cfg, key_want = key_subnet), 
                            getPoolStride3D(indx_layer, cfg, key_want = key_subnet))

        else:
            layer_name = "fc%i" % (1 + indx_layer + layer_offset)
            with tf.variable_scope(layer_name):
                m.fc(getFcNumFilters(indx_layer, cfg, key_want = key_subnet), 
                        init='trunc_norm', dropout=dropout, bias=.1)

                if getWhetherBn(indx_layer, cfg, key_want = key_subnet):
                    m.batchnorm_corr(train)

    return m

def build_partnet_3d(m, cfg, key_layernum, key_subnet, inputs=None, layer_offset=0, dropout=None):
    layernum_sub = cfg[ key_layernum ]
    for indx_layer in xrange(layernum_sub):
        do_conv = getWhetherConv(indx_layer, cfg, key_want = key_subnet)
        if do_conv:
            layer_name = "conv%i" % (1 + indx_layer + layer_offset)
            with tf.variable_scope(layer_name):
                curr_size = getConvFilterSize(indx_layer, cfg, key_want = key_subnet) 
                curr_size_list = [1, curr_size, curr_size]
                if indx_layer==0 and (not inputs==None):
                    m.conv3(getConvNumFilters(indx_layer, cfg, key_want = key_subnet), 
                            curr_size_list,
                            getConvStride(indx_layer, cfg, key_want = key_subnet),
                           padding='VALID', in_layer = inputs)
                else:
                    m.conv3(getConvNumFilters(indx_layer, cfg, key_want = key_subnet), 
                            curr_size_list, 
                            getConvStride(indx_layer, cfg, key_want = key_subnet))

                if getWhetherBn(indx_layer, cfg, key_want = key_subnet):
                    m.batchnorm_corr(train)

                do_pool = getWhetherPool(indx_layer, cfg, key_want = key_subnet)
                if do_pool:
                    m.pool(getPoolFilterSize(indx_layer, cfg, key_want = key_subnet), 
                            getPoolStride(indx_layer, cfg, key_want = key_subnet))

        else:
            layer_name = "fc%i" % (1 + indx_layer + layer_offset)
            with tf.variable_scope(layer_name):
                m.fc(getFcNumFilters(indx_layer, cfg, key_want = key_subnet), 
                        init='trunc_norm', dropout=dropout, bias=.1)

                if getWhetherBn(indx_layer, cfg, key_want = key_subnet):
                    m.batchnorm_corr(train)

    return m

def build_partnet(m, cfg, key_layernum, key_subnet, inputs=None, layer_offset=0, dropout=None):
    layernum_sub = cfg[ key_layernum ]
    for indx_layer in xrange(layernum_sub):
        do_conv = getWhetherConv(indx_layer, cfg, key_want = key_subnet)
        if do_conv:
            layer_name = "conv%i" % (1 + indx_layer + layer_offset)
            with tf.variable_scope(layer_name):
                if indx_layer==0 and (not inputs==None):
                    m.conv(getConvNumFilters(indx_layer, cfg, key_want = key_subnet), 
                            getConvFilterSize(indx_layer, cfg, key_want = key_subnet), 
                            getConvStride(indx_layer, cfg, key_want = key_subnet),
                           padding='VALID', in_layer = inputs)
                else:
                    m.conv(getConvNumFilters(indx_layer, cfg, key_want = key_subnet), 
                            getConvFilterSize(indx_layer, cfg, key_want = key_subnet), 
                            getConvStride(indx_layer, cfg, key_want = key_subnet))

                if getWhetherBn(indx_layer, cfg, key_want = key_subnet):
                    m.batchnorm_corr(train)

                do_pool = getWhetherPool(indx_layer, cfg, key_want = key_subnet)
                if do_pool:
                    m.pool(getPoolFilterSize(indx_layer, cfg, key_want = key_subnet), 
                            getPoolStride(indx_layer, cfg, key_want = key_subnet))

        else:
            layer_name = "fc%i" % (1 + indx_layer + layer_offset)
            with tf.variable_scope(layer_name):
                m.fc(getFcNumFilters(indx_layer, cfg, key_want = key_subnet), 
                        init='trunc_norm', dropout=dropout, bias=.1)

                if getWhetherBn(indx_layer, cfg, key_want = key_subnet):
                    m.batchnorm_corr(train)

    return m

def catenet_spa_temp_3din2d(inputs, cfg_initial, train = True, **kwargs):

    m = model.ConvNet(**kwargs)

    cfg = cfg_initial

    dropout_default = 0.5
    if 'dropout' in cfg:
        dropout_default = cfg['dropout']

    dropout = dropout_default if train else None

    if dropout==0:
        dropout = None

    shape_list = inputs.get_shape().as_list()
    curr_layer = 0

    assert shape_list[2]==35, 'Must set expand==1'

    m = build_partnet_3din2d(m, cfg, "layernum_spa", "spanet", inputs = inputs, layer_offset = curr_layer, dropout = dropout)

    curr_layer = curr_layer + cfg["layernum_spa"]
    m = build_partnet(m, cfg, "layernum_temp", "tempnet", layer_offset = curr_layer, dropout = dropout)

    return m

def catenet_spa_temp_3d(inputs, cfg_initial, train = True, **kwargs):

    m = model.ConvNet(**kwargs)

    cfg = cfg_initial

    dropout_default = 0.5
    if 'dropout' in cfg:
        dropout_default = cfg['dropout']

    dropout = dropout_default if train else None

    if dropout==0:
        dropout = None

    shape_list = inputs.get_shape().as_list()
    curr_layer = 0

    assert shape_list[2]==35, 'Must set expand==1'

    inputs = tf.reshape(inputs, [shape_list[0], shape_list[1], 5, 7, -1])
    m = build_partnet_3d(m, cfg, "layernum_spa", "spanet", inputs = inputs, layer_offset = curr_layer, dropout = dropout)
    new_input = m.output
    shape_list_tmp = new_input.get_shape().as_list()
    new_input = tf.reshape(new_input, [shape_list_tmp[0], shape_list_tmp[1], 1, -1])

    m.output = new_input
    curr_layer = curr_layer + cfg["layernum_spa"]
    m = build_partnet(m, cfg, "layernum_temp", "tempnet", layer_offset = curr_layer, dropout = dropout)

    return m

def catenet_spa_temp(inputs, cfg_initial, train = True, **kwargs):

    m = model.ConvNet(**kwargs)

    cfg = cfg_initial

    dropout_default = 0.5
    if 'dropout' in cfg:
        dropout_default = cfg['dropout']

    dropout = dropout_default if train else None

    if dropout==0:
        dropout = None

    inputs = tf.transpose(inputs, [0, 2, 1, 3])
    shape_list = inputs.get_shape().as_list()
    split_num = shape_list[2]
    if 'split_num' in cfg:
        split_num = cfg['split_num']
    small_inputs = tf.split(inputs, split_num, 2)
    small_outputs = []
    curr_layer = 0

    assert shape_list[1]==35, 'Must set expand==1'

    first_flag = True

    for small_input in small_inputs:
        small_input = tf.reshape(small_input, [shape_list[0], 5, 7, -1])
        if first_flag:
            with tf.variable_scope("small"):
                m = build_partnet(m, cfg, "layernum_spa", "spanet", inputs = small_input, layer_offset = curr_layer, dropout = dropout)
            first_flag = False
        else:
            with tf.variable_scope("small", reuse=True):
                m = build_partnet(m, cfg, "layernum_spa", "spanet", inputs = small_input, layer_offset = curr_layer, dropout = dropout)
        small_output = m.output
        shape_list_tmp = small_output.get_shape().as_list()
        small_output = tf.reshape(small_output, [shape_list_tmp[0], 1, 1, -1])
        small_outputs.append(small_output)

    new_input = tf.concat(small_outputs, 1)
    m.output = new_input
    curr_layer = curr_layer + cfg["layernum_spa"]
    m = build_partnet(m, cfg, "layernum_temp", "tempnet", layer_offset = curr_layer, dropout = dropout)

    return m

def catenet_temp_spa_sep(inputs, cfg_initial, train = True, **kwargs):

    m = model.ConvNet(**kwargs)

    cfg = cfg_initial

    dropout_default = 0.5
    if 'dropout' in cfg:
        dropout_default = cfg['dropout']

    dropout = dropout_default if train else None

    if dropout==0:
        dropout = None

    curr_layer = 0
    m = build_partnet(m, cfg, "layernum_temp", "tempnet", inputs = inputs, layer_offset = curr_layer, dropout = dropout)
    curr_layer = curr_layer + cfg["layernum_temp"]

    tensor_tmp = m.output
    tensor_tmp = tf.transpose(tensor_tmp, perm = [0, 2, 1, 3])

    shape_list = tensor_tmp.get_shape().as_list()
    #print(shape_list)
    sep_num = 9
    if "sep_num" in cfg:
        sep_num = cfg["sep_num"]

    small_inputs = tf.split(tensor_tmp, sep_num, 2)
    small_outputs = []

    first_flag = True

    for small_input in small_inputs:
        tensor_tmp = tf.reshape(small_input, [shape_list[0], shape_list[1], -1])

        shape_now = tensor_tmp.get_shape().as_list()
        slice0 = tf.slice(tensor_tmp, [0, 0, 0], [-1, 5, -1])
        slice1 = tf.slice(tensor_tmp, [0, 5, 0], [-1, 6, -1])
        slice2 = tf.slice(tensor_tmp, [0, 11, 0], [-1, 14, -1])
        slice3 = tf.slice(tensor_tmp, [0, 25, 0], [-1, 6, -1])

        pad_ten0 = tf.zeros([shape_now[0], 1, shape_now[2]])
        pad_ten1 = tf.zeros([shape_now[0], 1, shape_now[2]])
        pad_ten2 = tf.zeros([shape_now[0], 1, shape_now[2]])
        pad_ten3 = tf.zeros([shape_now[0], 1, shape_now[2]])

        tensor_tmp = tf.concat([slice0, pad_ten0, pad_ten1, slice1, pad_ten2, slice2, pad_ten3, slice3], 1)

        tensor_tmp = tf.reshape(tensor_tmp, [shape_list[0], 5, 7, -1])

        m.output = tensor_tmp
        if first_flag:
            with tf.variable_scope("small"):
                m = build_partnet(m, cfg, "layernum_spa", "spanet", layer_offset = curr_layer, dropout = dropout)
            first_flag = False
        else:
            with tf.variable_scope("small", reuse=True):
                m = build_partnet(m, cfg, "layernum_spa", "spanet", layer_offset = curr_layer, dropout = dropout)

        small_outputs.append( m.output )

    new_inputs = tf.concat(small_outputs, 1)
    m.output = new_inputs
    curr_layer = curr_layer + cfg["layernum_spa"]

    m = build_partnet(m, cfg, "layernum_spatemp", "spatempnet", layer_offset = curr_layer, dropout = dropout)

    return m

def catenet_temp_spa(inputs, cfg_initial, train = True, **kwargs):

    m = model.ConvNet(**kwargs)

    cfg = cfg_initial

    dropout_default = 0.5
    if 'dropout' in cfg:
        dropout_default = cfg['dropout']

    dropout = dropout_default if train else None

    if dropout==0:
        dropout = None

    curr_layer = 0
    m = build_partnet(m, cfg, "layernum_temp", "tempnet", inputs = inputs, layer_offset = curr_layer, dropout = dropout)
    curr_layer = curr_layer + cfg["layernum_temp"]

    tensor_tmp = m.output
    tensor_tmp = tf.transpose(tensor_tmp, perm = [0, 2, 1, 3])

    shape_list = tensor_tmp.get_shape().as_list()
    tensor_tmp = tf.reshape(tensor_tmp, [shape_list[0], shape_list[1], -1])

    shape_now = tensor_tmp.get_shape().as_list()
    slice0 = tf.slice(tensor_tmp, [0, 0, 0], [-1, 5, -1])
    slice1 = tf.slice(tensor_tmp, [0, 5, 0], [-1, 6, -1])
    slice2 = tf.slice(tensor_tmp, [0, 11, 0], [-1, 14, -1])
    slice3 = tf.slice(tensor_tmp, [0, 25, 0], [-1, 6, -1])

    pad_ten0 = tf.zeros([shape_now[0], 1, shape_now[2]])
    pad_ten1 = tf.zeros([shape_now[0], 1, shape_now[2]])
    pad_ten2 = tf.zeros([shape_now[0], 1, shape_now[2]])
    pad_ten3 = tf.zeros([shape_now[0], 1, shape_now[2]])

    tensor_tmp = tf.concat([slice0, pad_ten0, pad_ten1, slice1, pad_ten2, slice2, pad_ten3, slice3], 1)

    tensor_tmp = tf.reshape(tensor_tmp, [shape_list[0], 5, 7, -1])

    m.output = tensor_tmp
    m = build_partnet(m, cfg, "layernum_spa", "spanet", layer_offset = curr_layer, dropout = dropout)

    return m

def catenet_all3d(inputs, cfg_initial = None, train=True, **kwargs):
    m = model.ConvNet(**kwargs)

    cfg = cfg_initial

    dropout_default = 0.5
    if 'dropout' in cfg:
        dropout_default = cfg['dropout']

    dropout = dropout_default if train else None

    if dropout==0:
        dropout = None

    shape_list = inputs.get_shape().as_list()
    assert shape_list[2]==35, 'Must set expand==1'
    inputs = tf.reshape(inputs, [shape_list[0], shape_list[1], 5, 7, -1])
    m = build_partnet_all3d(m, cfg, "layernum_sub", "subnet", inputs = inputs, layer_offset = 0, dropout = dropout)

    return m

def catenet_tnn(inputs, cfg_path, train = True, tnndecay = 0.1, decaytrain = 0, cfg_initial = None, cmu = 0, fixweights = False, seed = 0, **kwargs):
    m = model.ConvNet(fixweights = fixweights, seed = seed, **kwargs)

    params = {'input': inputs.name,
               'type': 'fc'
            }

    dropout = 0.5 if train else None

    # Get inputs
    shape_list = inputs.get_shape().as_list()
    assert shape_list[2]==35, 'Must set expand==1'
    sep_num = shape_list[1]
    if not cfg_initial is None and 'sep_num' in cfg_initial:
        sep_num = cfg_initial['sep_num']
    small_inputs = tf.split(inputs, sep_num, 1)
    for indx_time in xrange(len(small_inputs)):
        small_inputs[indx_time] = tf.transpose(small_inputs[indx_time], [0, 2, 1, 3])
        small_inputs[indx_time] = tf.reshape(small_inputs[indx_time], [shape_list[0], 5, 7, -1])

    G = main.graph_from_json(cfg_path)

    if 'all_conn' in cfg_initial:
        node_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6']

	MASTER_EDGES = []
	for i in range(len(node_list)):
	    for j in range(len(node_list)):
		if (j > i+1 or i > j) and (j>0): #since (i, j) w/ j > i is already an edge
		    MASTER_EDGES.append((node_list[i], node_list[j]))

        print(MASTER_EDGES)
        G.add_edges_from(MASTER_EDGES)


    for node, attr in G.nodes(data=True):

        memory_func, memory_param = attr['kwargs']['memory']
        if 'nunits' in memory_param:
            attr['cell'] = tnn_LSTMCell
        else:
            memory_param['memory_decay'] = tnndecay
            memory_param['trainable'] = decaytrain==1
            attr['kwargs']['memory'] = (memory_func, memory_param)

        if fixweights:
            if node.startswith('conv'):
                _, prememory_param = attr['kwargs']['pre_memory'][0]
                attr['kwargs']['pre_memory'][0] = (model.conv_fix, prememory_param)

            if node.startswith('fc'):
                _, prememory_param = attr['kwargs']['pre_memory'][0]
                attr['kwargs']['pre_memory'][0] = (model.fc_fix, prememory_param)

        if not seed==0:
            for sub_prememory in attr['kwargs']['pre_memory']:
                prememory_func, prememory_param = sub_prememory
                if 'kernel_init_kwargs' in prememory_param:
                    prememory_param['kernel_init_kwargs']['seed'] = seed

	if node in ['fc7', 'fc8']:
            attr['kwargs']['pre_memory'][0][1]['dropout'] = dropout

    main.init_nodes(G, batch_size=shape_list[0])
    main.unroll(G, input_seq={'conv1': small_inputs}, ntimes = len(small_inputs))

    if not 'retres' in cfg_initial:
        if cmu==0:
            m.output = G.node['fc8']['outputs'][-1]
        else:
            m.output = tf.transpose(tf.stack(G.node['fc8']['outputs']), [1,2,0])
    else:
        m.output = tf.concat([G.node['fc8']['outputs'][x] for x in cfg_initial['retres']], 1)

    print(len(G.node['fc8']['outputs']))
    m.params = params

    return m

def catenet(inputs, cfg_initial = None, train=True, **kwargs):
    m = model.ConvNet(**kwargs)

    cfg = cfg_initial

    dropout_default = 0.5
    if 'dropout' in cfg:
        dropout_default = cfg['dropout']

    dropout = dropout_default if train else None

    if dropout==0:
        dropout = None

    layernum_sub = cfg['layernum_sub']
    for indx_layer in xrange(layernum_sub):
        do_conv = getWhetherConv(indx_layer, cfg)
        if do_conv:
            layer_name = "conv%i" % (1 + indx_layer)
            with tf.variable_scope(layer_name):
                if indx_layer==0:
                    m.conv(getConvNumFilters(indx_layer, cfg), getConvFilterSize(indx_layer, cfg), getConvStride(indx_layer, cfg),
                           padding='VALID', in_layer = inputs)
                else:
                    m.conv(getConvNumFilters(indx_layer, cfg), getConvFilterSize(indx_layer, cfg), getConvStride(indx_layer, cfg))

                if getWhetherLrn(indx_layer, cfg):
                    print('Lrn used!')
                    m.norm(getLrnDepth(indx_layer, cfg))

                if getWhetherBn(indx_layer, cfg):
                    m.batchnorm_corr(train)

                do_pool = getWhetherPool(indx_layer, cfg)
                if do_pool:
                    m.pool(getPoolFilterSize(indx_layer, cfg), getPoolStride(indx_layer, cfg))

        else:
            layer_name = "fc%i" % (1 + indx_layer)
            with tf.variable_scope(layer_name):
                m.fc(getFcNumFilters(indx_layer, cfg), init='trunc_norm', dropout=dropout, bias=.1)

                if getWhetherBn(indx_layer, cfg):
                    m.batchnorm_corr(train)

    if layernum_sub==0:
        m.output = inputs
    return m

def catenet_add(inputs, cfg_initial = None, train=True, **kwargs):

    m_add = model.ConvNet(**kwargs)
    cfg = cfg_initial

    dropout_default = 0.5
    if 'dropout' in cfg:
        dropout_default = cfg['dropout']

    dropout = dropout_default if train else None

    if dropout==0:
        dropout = None

    if 'layernum_add' in cfg:
        layernum_add = cfg['layernum_add']
    else:
        layernum_add = 1

    m_add.output = inputs

    for indx_layer in xrange(layernum_add - 1):
        layer_name = "fc_add%i" % (1 + indx_layer)
        with tf.variable_scope(layer_name):
            m_add.fc(getFcNumFilters(indx_layer, cfg, key_want = "addnet"), init='trunc_norm', dropout=dropout, bias=.1)

            if getWhetherBn(indx_layer, cfg, key_want = "addnet"):
                m.batchnorm_corr(train)

    with tf.variable_scope('fc_add'):
        #m_add.fc(117, init='trunc_norm', activation=None, dropout=None, bias=0)
        m_add.fc(117, init='trunc_norm', activation=None, dropout=None, bias=0, trainable = True)

    total_parameters = 0
    for variable in tf.trainable_variables():
	# shape is an array of tf.Dimension
	shape = variable.get_shape()
	#print(shape)
	#print(len(shape))
	variable_parametes = 1
	for dim in shape:
	    #print(dim)
	    variable_parametes *= dim.value
	#print(variable_parametes)
	total_parameters += variable_parametes
    print(total_parameters)

    return m_add

def catenet_from_3s(input_con, func_each = catenet, **kwargs):
    input1,input2,input3 = tf.split(input_con, 3, 1)

    with tf.variable_scope("cate_root"):
        # Building the network

        with tf.variable_scope("create"):
            m1 = func_each(input1, **kwargs)
            output_1 = m1.output

        #tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("create", reuse=True):

            m2 = func_each(input2, **kwargs)
            output_2 = m2.output

            m3 = func_each(input3, **kwargs)
            output_3 = m3.output

        input_t = tf.concat([output_1, output_2, output_3], 1)

        return input_t

def catenet_from_12s(input_con, func_each = catenet, **kwargs):
    input0,input1,input2,input3 = tf.split(input_con, 4, 1)

    with tf.variable_scope("create_big"):
        input_t0 = catenet_from_3s(input0, func_each = func_each, **kwargs)

    with tf.variable_scope("create_big", reuse=True):
        input_t1 = catenet_from_3s(input1, func_each = func_each, **kwargs)
        input_t2 = catenet_from_3s(input2, func_each = func_each, **kwargs)
        input_t3 = catenet_from_3s(input3, func_each = func_each, **kwargs)

    input_t = tf.concat([input_t0, input_t1, input_t2, input_t3], 1)

    return input_t

def deal_with_inputs(inputs):
    # Deal with inputs
    input_force, input_torque = (inputs['Data_force'], inputs['Data_torque'])
    shape_list = input_force.get_shape().as_list()
    input_force_rs = tf.reshape(input_force, [shape_list[0], shape_list[1], shape_list[2], -1])
    input_torque_rs = tf.reshape(input_torque, [shape_list[0], shape_list[1], shape_list[2], -1])
    input_con = tf.concat([input_force_rs, input_torque_rs], 3)

    return input_con

def catenet_3d_tfutils(inputs, split_12 = False, **kwargs):

    input_con = deal_with_inputs(inputs)
    #print(input_con.get_shape().as_list())

    if not split_12:
        input_t = catenet_from_3s(input_con, func_each = catenet_all3d, **kwargs)
    else:
        input_t = catenet_from_12s(input_con, func_each = catenet_all3d, **kwargs)

    m_final = catenet_add(input_t, **kwargs)
    return m_final.output, m_final.params

def catenet_tfutils(inputs, split_12 = False, **kwargs):

    input_con = deal_with_inputs(inputs)
    #print(input_con.get_shape().as_list())

    if not split_12:
        input_t = catenet_from_3s(input_con, **kwargs)
    else:
        input_t = catenet_from_12s(input_con, **kwargs)

    m_final = catenet_add(input_t, **kwargs)
    return m_final.output, m_final.params

def catenet_temp_spa_sep_tfutils(inputs, split_12 = False, **kwargs):

    input_con = deal_with_inputs(inputs)
    #print(input_con.get_shape().as_list())

    if not split_12:
        input_t = catenet_from_3s(input_con, func_each = catenet_temp_spa_sep, **kwargs)
    else:
        input_t = catenet_from_12s(input_con, func_each = catenet_temp_spa_sep, **kwargs)

    m_final = catenet_add(input_t, **kwargs)
    return m_final.output, m_final.params

def catenet_temp_spa_tfutils(inputs, split_12 = False, **kwargs):

    input_con = deal_with_inputs(inputs)
    #print(input_con.get_shape().as_list())

    if not split_12:
        input_t = catenet_from_3s(input_con, func_each = catenet_temp_spa, **kwargs)
    else:
        input_t = catenet_from_12s(input_con, func_each = catenet_temp_spa, **kwargs)

    m_final = catenet_add(input_t, **kwargs)
    return m_final.output, m_final.params

def catenet_spa_temp_3d_tfutils(inputs, split_12 = False, **kwargs):

    input_con = deal_with_inputs(inputs)
    #print(input_con.get_shape().as_list())

    if not split_12:
        input_t = catenet_from_3s(input_con, func_each = catenet_spa_temp_3d, **kwargs)
    else:
        input_t = catenet_from_12s(input_con, func_each = catenet_spa_temp_3d, **kwargs)

    m_final = catenet_add(input_t, **kwargs)
    return m_final.output, m_final.params

def catenet_spa_temp_3din2d_tfutils(inputs, split_12 = False, **kwargs):

    input_con = deal_with_inputs(inputs)
    #print(input_con.get_shape().as_list())

    if not split_12:
        input_t = catenet_from_3s(input_con, func_each = catenet_spa_temp_3din2d, **kwargs)
    else:
        input_t = catenet_from_12s(input_con, func_each = catenet_spa_temp_3din2d, **kwargs)

    m_final = catenet_add(input_t, **kwargs)
    return m_final.output, m_final.params

def catenet_spa_temp_tfutils(inputs, split_12 = False, **kwargs):

    input_con = deal_with_inputs(inputs)
    #print(input_con.get_shape().as_list())

    if not split_12:
        input_t = catenet_from_3s(input_con, func_each = catenet_spa_temp, **kwargs)
    else:
        input_t = catenet_from_12s(input_con, func_each = catenet_spa_temp, **kwargs)

    m_final = catenet_add(input_t, **kwargs)
    return m_final.output, m_final.params

def catenet_tnn_tfutils(inputs, split_12 = False, **kwargs):

    input_con = deal_with_inputs(inputs)
    #print(input_con.get_shape().as_list())

    if not split_12:
        input_t = catenet_from_3s(input_con, func_each = catenet_tnn, **kwargs)
    else:
        input_t = catenet_from_12s(input_con, func_each = catenet_tnn, **kwargs)

    m_final = catenet_add(input_t, **kwargs)
    return m_final.output, m_final.params

def catenet_tnn_cmu_tfutils(inputs, split_12 = False, **kwargs):

    input_con = deal_with_inputs(inputs)
    #print(input_con.get_shape().as_list())

    assert kwargs['cmu']==1, 'Must set cmu to be 1!'

    if not split_12:
        input_t = catenet_from_3s(input_con, func_each = catenet_tnn, **kwargs)
    else:
        input_t = catenet_from_12s(input_con, func_each = catenet_tnn, **kwargs)

    all_input_t = tf.unstack(input_t, axis=2)

    first_flag = True
    ret_param = None
    all_output = []
    for small_input in all_input_t:
        if first_flag:
            with tf.variable_scope("cmu"):
                m_final = catenet_add(small_input, **kwargs)
                ret_param = m_final.params
                first_flag = False
        else:
            with tf.variable_scope("cmu", reuse = True):
                m_final = catenet_add(small_input, **kwargs)

        all_output.append(m_final.output)
    return all_output, ret_param

# Deprecated
def catenet_old(cfg_initial = None, train=True, seed=0, **kwargs):
    defaults = {'conv': {'batch_norm': False,
                         'kernel_init': 'xavier',
                         'kernel_init_kwargs': {'seed': seed}},
                         'weight_decay': .0005,
                'max_pool': {'padding': 'SAME'},
                'fc': {'batch_norm': False,
                       'kernel_init': 'truncated_normal',
                       'kernel_init_kwargs': {'stddev': .01, 'seed': seed},
                       'weight_decay': .0005,
                       'dropout_seed': 0}}
    m = model_tfutils.ConvNet(defaults=defaults)
    dropout = .5 if train else None

    cfg = cfg_initial

    layernum_sub = cfg['layernum_sub']
    for indx_layer in xrange(layernum_sub):
        do_conv = getWhetherConv(indx_layer, cfg)
        if do_conv:
            layer_name = "conv%i" % (1 + indx_layer)
            if indx_layer==0:
                m.conv(getConvNumFilters(indx_layer, cfg), getConvFilterSize(indx_layer, cfg), getConvStride(indx_layer, cfg),
                       padding='VALID', layer= layer_name)
            else:
                m.conv(getConvNumFilters(indx_layer, cfg), getConvFilterSize(indx_layer, cfg), getConvStride(indx_layer, cfg),
                       layer= layer_name)

            do_pool = getWhetherPool(indx_layer, cfg)
            if do_pool:
                m.max_pool(getPoolFilterSize(indx_layer, cfg), getPoolStride(indx_layer, cfg), layer = layer_name)
        else:
            layer_name = "fc%i" % (1 + indx_layer)
            m.fc(getFcNumFilters(indx_layer, cfg), dropout=dropout, bias=.1, layer=layer_name)

    m_add = model_tfutils.ConvNet(defaults=defaults)
    m_add.fc(117, activation=None, dropout=None, bias=0, layer='fc8')

    return m, m_add

def catenet_tfutils_old(inputs, **kwargs):

    # Deal with inputs
    input_con = deal_with_inputs(inputs)
    print(input_con.get_shape().as_list())
    input1,input2,input3 = tf.split(input_con, 3, 1)

    with tf.variable_scope("cate_root"):
        # Building the network

        with tf.variable_scope("create"):
            m1 = catenet(input1, **kwargs)
            output_1 = m1.output

        #tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("create", reuse=True):

            m2 = catenet(input2, **kwargs)
            output_2 = m2.output

            m3 = catenet(input3, **kwargs)
            output_3 = m3.output

        with tf.variable_scope("create"):
            input_t = tf.concat([output_1, output_2, output_3], 1)
            m_final = catenet_add(input_t, **kwargs)

        return m_final.output, m_final.params

def spatial_slice_concat(data):
    shape_now = data.get_shape().as_list()
    slice0 = tf.slice(data, [0, 0, 0, 0, 0], [-1, -1, 5, -1, -1])
    slice1 = tf.slice(data, [0, 0, 5, 0, 0], [-1, -1, 6, -1, -1])
    slice2 = tf.slice(data, [0, 0, 11, 0, 0], [-1, -1, 14, -1, -1])
    slice3 = tf.slice(data, [0, 0, 25, 0, 0], [-1, -1, 6, -1, -1])

    pad_ten0 = tf.zeros([shape_now[0], shape_now[1], 1, shape_now[3], shape_now[4]])
    pad_ten1 = tf.zeros([shape_now[0], shape_now[1], 1, shape_now[3], shape_now[4]])
    pad_ten2 = tf.zeros([shape_now[0], shape_now[1], 1, shape_now[3], shape_now[4]])
    pad_ten3 = tf.zeros([shape_now[0], shape_now[1], 1, shape_now[3], shape_now[4]])

    data = tf.concat([slice0, pad_ten0, pad_ten1, slice1, pad_ten2, slice2, pad_ten3, slice3], 2)
    #data[new_key] = tf.concat([slice0, slice1, slice2, slice3], 2)

    return data

def parallel_net_builder(inputs, model_func, n_gpus = 2, gpu_offset = 0, inputthre = 0, with_modelprefix = None, expand = 0, **kwargs):
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        #assert n_gpus > 1, ('At least two gpus have to be used')
        outputs = []
        params = []

        if n_gpus > 1:
            #with tf.device('/cpu:0'):
            list_Data_force = tf.split(inputs['Data_force'], axis=0, num_or_size_splits=n_gpus)
            list_Data_torque = tf.split(inputs['Data_torque'], axis=0, num_or_size_splits=n_gpus)
        else:
            #list_Data_force = [tf.tile(inputs['Data_force'], [1,1,1,1,1])]
            #list_Data_torque = [tf.tile(inputs['Data_torque'], [1,1,1,1,1])]
            list_Data_force = [inputs['Data_force']]
            list_Data_torque = [inputs['Data_torque']]

        for i, (force_inp, torque_inp) in enumerate(zip(list_Data_force, list_Data_torque)):
            #print (force_inp, torque_inp)
            if inputthre>0:
                force_inp = tf.minimum(force_inp, tf.constant(  inputthre, dtype = force_inp.dtype))
                force_inp = tf.maximum(force_inp, tf.constant( -inputthre, dtype = force_inp.dtype))

                torque_inp = tf.minimum(torque_inp, tf.constant(  inputthre, dtype = torque_inp.dtype))
                torque_inp = tf.maximum(torque_inp, tf.constant( -inputthre, dtype = torque_inp.dtype))

                #force_inp = tf.Print(force_inp, [tf.reduce_max(force_inp)], message = 'Input max: ')

            if expand==1:
                #with tf.device('/cpu:0'):
                force_inp = spatial_slice_concat(force_inp)
                torque_inp = spatial_slice_concat(torque_inp)

            with tf.device('/gpu:%d' % (i + gpu_offset)):
                print('On gpu %d' % (i + gpu_offset))
                print(force_inp.get_shape().as_list())
                with tf.name_scope('gpu_' + str(i)) as gpu_scope:
                    output, param = model_func({'Data_force': force_inp, 'Data_torque': torque_inp}, **kwargs)
                    outputs.append(output)
                    params.append(param)
                    tf.get_variable_scope().reuse_variables()

        if with_modelprefix is None:
            params = params[0]
        else:
            params = params[0]
            orig_keys = params.keys()
            for key_now in orig_keys:
                params['%s/%s' % (with_modelprefix, key_now)] = params.pop(key_now)

        return [outputs, params]
