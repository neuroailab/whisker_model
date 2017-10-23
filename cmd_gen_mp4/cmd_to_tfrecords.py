import argparse
import h5py
import os
import tensorflow as tf
import numpy as np
import cPickle
import json

NUM_SWIPES = 12
SWIPE_ORDER = [0, 4, 8,  1, 5, 9,  2, 6, 10,  3, 7, 11]

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_list(fin_name):
    fin = open(fin_name, 'r')
    lines = fin.readlines()

    ret_list = []

    for line in lines:
        split_line = line.split(' ')
        obj_id = split_line[0]
        cat_id = int(split_line[1])

        ret_list.append((obj_id, cat_id))

    return ret_list

def main():
    parser = argparse.ArgumentParser(description='The script to get object category list')
    parser.add_argument('--objsta', default = 0, type = int, action = 'store', help = 'Start index in the object list')
    parser.add_argument('--objlen', default = 1, type = int, action = 'store', help = 'Length for generating')
    parser.add_argument('--seedbas', default = 0, type = int, action = 'store', help = 'Seed basis for randomization')
    parser.add_argument('--bigsamnum', default = 12, type = int, action = 'store', help = 'Sampling number for every object')
    parser.add_argument('--loaddir', default = '/om/user/chengxuz/Data/barrel_dataset/raw_hdf5', type = str, action = 'store', help = 'Directory for the generated hdf5s')
    parser.add_argument('--infodir', default = '/om/user/chengxuz/Data/barrel_dataset/raw_info', type = str, action = 'store', help = 'Directory for the generated hdf5s')
    parser.add_argument('--savedir', default = '/om/user/chengxuz/Data/barrel_dataset/tfrecords', type = str, action = 'store', help = 'Directory for the generated tfrecords')
    parser.add_argument('--objcat', default = 'obj_category.txt', type = str, action = 'store', help = 'The file having the object information')
    parser.add_argument('--checkmode', default = 0, type = int, action = 'store', help = '0 means writting mode, 1 means checking mode (only implemented for Data_force now)')

    args    = parser.parse_args()

    obj_list = get_list(args.objcat)

    # Write the scales
    key_now = 'scale'
    dir_now = os.path.join(args.savedir, key_now)
    if not os.path.isdir(dir_now):
        os.system('mkdir -p %s' % dir_now)

    name_now = "Data%i_%i.tfrecords" % (args.objsta, args.objlen)
    path_now = os.path.join(dir_now, name_now)
    writer = tf.python_io.TFRecordWriter(path_now)

    for obj_indx in xrange(args.objsta, min(args.objsta + args.objlen, len(obj_list))):
        for sam_indx in xrange(args.bigsamnum):
            info_name = "Data%i_%i_%s.txt" % (args.seedbas + obj_indx, sam_indx, obj_list[obj_indx][0])

            info_path = os.path.join(args.infodir, info_name)

            fin_now = open(info_path, 'r')

            lines_now = fin_now.readlines()

            all_scales = []

            for line in lines_now:
                json_acceptable_string = line.replace("'", "\"")
                dict_now = json.loads(json_acceptable_string)
                all_scales.append(dict_now['control_len'][0])

            for group_indx in SWIPE_ORDER:

                example = tf.train.Example(features=tf.train.Features(feature={
                    key_now: _float_feature(all_scales[group_indx])}))
                #print all_scales[group_indx]

                writer.write(example.SerializeToString())
    writer.close()

    # Write the labels

    key_now = 'category'
    dir_now = os.path.join(args.savedir, key_now)
    if not os.path.isdir(dir_now):
        os.system('mkdir -p %s' % dir_now)

    name_now = "Data%i_%i.tfrecords" % (args.objsta, args.objlen)
    path_now = os.path.join(dir_now, name_now)
    writer = tf.python_io.TFRecordWriter(path_now)

    for obj_indx in xrange(args.objsta, min(args.objsta + args.objlen, len(obj_list))):
        label_now = obj_list[obj_indx][1]
        for sam_indx in xrange(args.bigsamnum):
            for group_indx in SWIPE_ORDER:

                example = tf.train.Example(features=tf.train.Features(feature={
                    key_now: _int64_feature(label_now)}))

                writer.write(example.SerializeToString())
    writer.close()

    # Write other organized things

    key_list =[
        u'Data_force',
        u'Data_normal',
        u'Data_torque',
        u'orn',
        u'position',
        u'speed'
        ]

    for key_now in key_list:
        dir_now = os.path.join(args.savedir, key_now)
        if not os.path.isdir(dir_now):
            os.system('mkdir -p %s' % dir_now)

        name_now = "Data%i_%i.tfrecords" % (args.objsta, args.objlen)
        path_now = os.path.join(dir_now, name_now)

        if args.checkmode==1:
            record_iterator = tf.python_io.tf_record_iterator(path=path_now)
            reconstructed_images = []

            for string_record in record_iterator:
                
                example = tf.train.Example()
                example.ParseFromString(string_record)
                
                img_string = (example.features.feature[key_now]
                                              .bytes_list
                                              .value[0])
                
                #img_1d = np.fromstring(img_string, dtype=np.uint16)
                img_1d = np.fromstring(img_string, dtype=np.uint8)
                #print(img_1d.shape)
                reconstructed_img = img_1d.reshape((5, 256, 256, 3))
                #img_1d = np.fromstring(img_string, dtype=np.float32)
                #reconstructed_img = img_1d.reshape((110, 31, 3, 3))
                
                reconstructed_images.append(reconstructed_img)
        else:
            writer = tf.python_io.TFRecordWriter(path_now)

        indx_record = 0
        meta_dict = {}
        
        for obj_indx in xrange(args.objsta, min(args.objsta + args.objlen, len(obj_list))):
            for sam_indx in xrange(args.bigsamnum):
                hdf5_name = "Data%i_%i_%s.hdf5" % (args.seedbas + obj_indx, sam_indx, obj_list[obj_indx][0])

                print(hdf5_name)

                hdf5_path = os.path.join(args.loaddir, hdf5_name)
                fin_now = h5py.File(hdf5_path, 'r')
                for group_indx in SWIPE_ORDER:
                    group_name = "Data%i" % group_indx

                    array_now = np.asarray(fin_now[group_name][key_now])

                    meta_dict[key_now] = {}
                    meta_dict[key_now]["dtype"] = array_now.dtype
                    meta_dict[key_now]["shape"] = array_now.shape

                    if args.checkmode==0:
                        img_raw = array_now.tostring()

                        example = tf.train.Example(features=tf.train.Features(feature={
                            key_now: _bytes_feature(img_raw)}))

                        writer.write(example.SerializeToString())
                    else:
                        print(np.allclose(reconstructed_images[indx_record], array_now))

                    indx_record = indx_record + 1
                fin_now.close()

        meta_path = os.path.join(dir_now, "meta.pkl")
        if not os.path.isfile(meta_path):
            cPickle.dump(meta_dict, open(meta_path, 'w'))

        if args.checkmode==0:
            writer.close()
        #break

if __name__=='__main__':
    main()
