import tensorflow as tf
import cPickle
import os

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='The script to write to tfrecords combining images and labels')
    parser.add_argument('--dir', default = '/path/to/store/tfrecords', type = str, action = 'store', help = 'Directory to save the tfrecords')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    key_list = [
        'Data_force',
        'Data_torque',
        'category',
        ]

    for curr_key in key_list:
        curr_dir = os.path.join(args.dir, curr_key)
        pkl_path = os.path.join(curr_dir, 'meta.pkl')

        if curr_key=='category':
            curr_dict = {curr_key: {'dtype': tf.int64, 'shape': ()}}
        else:
            curr_dict = {curr_key: {'dtype': tf.string, 'shape': ()}}

        cPickle.dump(curr_dict, open(pkl_path, 'w'))

if __name__=='__main__':
    main()
