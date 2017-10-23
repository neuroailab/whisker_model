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
    ret_dict = {}

    for line in lines:
        split_line = line.split(' ')
        obj_id = split_line[0]
        cat_id = int(split_line[1])
        if not cat_id in ret_dict:
            ret_dict[cat_id] = []
        ret_dict[cat_id].append((len(ret_list), obj_id))

        ret_list.append((obj_id, cat_id))

    return ret_list, ret_dict

def examin_finish(now_objlist, args):
    finish = True
    for obj_indx, obj_id in now_objlist:
        for sam_indx in xrange(args.bigsamnum):

            hdf5_name = "Data%i_%i_%s.hdf5" % (args.seedbas + obj_indx, sam_indx, obj_id)
            hdf5_path = os.path.join(args.loaddir, hdf5_name)

            if (os.path.exists(hdf5_path) and (os.path.getsize(hdf5_path)==14792976)):
                continue
            else:
                print(hdf5_path)
                finish = False
                break

        if not finish:
            break

    return finish



def write_tfrec(writer, key_now, now_objlist, train_sta, real_obj_len, args):
    for tmp_indx in xrange(train_sta, train_sta + real_obj_len):
        for sam_indx in xrange(args.bigsamnum):
            obj_indx, obj_id = now_objlist[tmp_indx]

            hdf5_name = "Data%i_%i_%s.hdf5" % (args.seedbas + obj_indx, sam_indx, obj_id)

            print(hdf5_name)

            hdf5_path = os.path.join(args.loaddir, hdf5_name)
            fin_now = h5py.File(hdf5_path, 'r')
            for group_indx in SWIPE_ORDER:
                group_name = "Data%i" % group_indx

                array_now = np.asarray(fin_now[group_name][key_now])

                img_raw = array_now.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    key_now: _bytes_feature(img_raw)}))

                writer.write(example.SerializeToString())

            fin_now.close()

def write_cate(writer, label_now, now_objlist, train_sta, real_obj_len, args):
    for tmp_indx in xrange(train_sta, train_sta + real_obj_len):
        for sam_indx in xrange(args.bigsamnum):
            for group_indx in SWIPE_ORDER:

                if isinstance(label_now, int):
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'category': _int64_feature(label_now)}))
                if isinstance(label_now, list):

                    curr_write_value = label_now[tmp_indx][0]

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'objid': _int64_feature(curr_write_value)}))

                    #print(curr_write_value)

                writer.write(example.SerializeToString())

def main():
    parser = argparse.ArgumentParser(description='The script to get object category list')
    parser.add_argument('--catsta', default = 0, type = int, action = 'store', help = 'Start index in the object list')
    parser.add_argument('--catlen', default = 1, type = int, action = 'store', help = 'Length for generating')
    #parser.add_argument('--seedbas', default = 0, type = int, action = 'store', help = 'Seed basis for randomization')
    parser.add_argument('--seedbas', default = 10000, type = int, action = 'store', help = 'Seed basis for randomization')
    parser.add_argument('--bigsamnum', default = 12, type = int, action = 'store', help = 'Sampling number for every object')
    #parser.add_argument('--loaddir', default = '/om/user/chengxuz/Data/barrel_dataset2/raw_hdf5', type = str, action = 'store', help = 'Directory for the generated hdf5s')
    #parser.add_argument('--savedir', default = '/om/user/chengxuz/Data/barrel_dataset2/tfrecords', type = str, action = 'store', help = 'Directory for the generated tfrecords')
    parser.add_argument('--loaddir', default = '/scratch/users/chengxuz/barrel/barrel_relat_files/dataset2/raw_hdf5', type = str, action = 'store', help = 'Directory for the generated hdf5s')
    parser.add_argument('--savedir', default = '/scratch/users/chengxuz/barrel/barrel_relat_files/dataset2/tfrecords', type = str, action = 'store', help = 'Directory for the generated tfrecords')
    parser.add_argument('--objcat', default = 'obj_category.txt', type = str, action = 'store', help = 'The file having the object information')
    parser.add_argument('--suffix', default = 'strain', type = str, action = 'store', help = 'Suffix to indicate whether this is saumple num train/val')
    #parser.add_argument('--checkmode', default = 0, type = int, action = 'store', help = '0 means writting mode, 1 means checking mode (only implemented for Data_force now)')

    args    = parser.parse_args()

    obj_list, cat_dict = get_list(args.objcat)

    sam_rate = 2.0/26.0
    obj_len = 21

    '''
    key_list =[
        u'Data_force',
        #u'Data_normal',
        u'Data_torque',
        u'orn',
        u'position',
        u'speed'
        ]
    '''
    #key_list =[u'scale']
    key_list =[]

    for which_cat in xrange(args.catsta, min(args.catsta + args.catlen, len(cat_dict.keys()))):
        now_objlist = cat_dict[which_cat]
        if examin_finish(now_objlist, args):
            print('Cat %i Finished!' % which_cat)
            pass
        else:
            print('Cat %i NOT finished!' % which_cat)
            break

        if which_cat%2==0:
            val_num_obj = np.ceil(len(now_objlist)*sam_rate)
        else:
            val_num_obj = np.floor(len(now_objlist)*sam_rate)

        val_num_obj = int(val_num_obj)

        for key_now in key_list:
            dir_now = os.path.join(args.savedir, key_now)
            if not os.path.isdir(dir_now):
                os.system('mkdir -p %s' % dir_now)

            train_len = len(now_objlist) - val_num_obj
            for train_sta in xrange(0, train_len, obj_len):
                real_obj_len = min(obj_len, train_len - train_sta)
                name_now = "ctrain_%i_%i_%i_%s.tfrecords" % (which_cat, train_sta, real_obj_len, args.suffix)
                path_now = os.path.join(dir_now, name_now)

                writer = tf.python_io.TFRecordWriter(path_now)

                write_tfrec(writer, key_now, now_objlist, train_sta, real_obj_len, args)

                writer.close()

            name_now = "cval_%i_%i_%i_%s.tfrecords" % (which_cat, train_len, val_num_obj, args.suffix)
            path_now = os.path.join(dir_now, name_now)

            writer = tf.python_io.TFRecordWriter(path_now)

            write_tfrec(writer, key_now, now_objlist, train_len, val_num_obj, args)

            writer.close()

        #for key_now in ['category']:
        #for key_now in []:
        for key_now in ['objid']:
            dir_now = os.path.join(args.savedir, key_now)
            if not os.path.isdir(dir_now):
                os.system('mkdir -p %s' % dir_now)

            train_len = len(now_objlist) - val_num_obj

            if key_now=='objid':
                label_now = now_objlist
            else:
                label_now = which_cat

            for train_sta in xrange(0, train_len, obj_len):
                real_obj_len = min(obj_len, train_len - train_sta)
                name_now = "ctrain_%i_%i_%i_%s.tfrecords" % (which_cat, train_sta, real_obj_len, args.suffix)
                path_now = os.path.join(dir_now, name_now)

                writer = tf.python_io.TFRecordWriter(path_now)

                #write_cate(writer, which_cat, now_objlist, train_sta, real_obj_len, args)
                write_cate(writer, label_now, now_objlist, train_sta, real_obj_len, args)

                writer.close()

            name_now = "cval_%i_%i_%i_%s.tfrecords" % (which_cat, train_len, val_num_obj, args.suffix)
            path_now = os.path.join(dir_now, name_now)

            writer = tf.python_io.TFRecordWriter(path_now)

            #write_cate(writer, which_cat, now_objlist, train_len, val_num_obj, args)
            write_cate(writer, label_now, now_objlist, train_len, val_num_obj, args)

            writer.close()

if __name__=='__main__':
    main()
