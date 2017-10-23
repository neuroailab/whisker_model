import os
import numpy as np
import tensorflow as tf
import copy
import argparse

def get_next_split(now_split, file_name, to_dir):
    split_indx = file_name.rfind('_')
    new_file_name = file_name[:split_indx] + "_" + str(now_split) + file_name[split_indx:]
    new_file_path = os.path.join(to_dir, new_file_name)
    writer = tf.python_io.TFRecordWriter(new_file_path)
    now_split = now_split + 1
    return writer, now_split

def main():
    parser = argparse.ArgumentParser(description='The script to split the tfrecords')

    #parser.add_argument('--key', default = 'category', type = str, action = 'store', help = 'Which key to split')
    parser.add_argument('--key', default = 'Data_force', type = str, action = 'store', help = 'Which key to split')
    parser.add_argument('--staindx', default = 0, type = int, action = 'store', help = 'Start index of the file')
    parser.add_argument('--lenindx', default = 1, type = int, action = 'store', help = 'Length of index of the file')

    # Usually fixed parameters
    parser.add_argument('--fromdir', default = '/mnt/fs2/chengxuz/Data/whisker2/tfrecords', type = str, action = 'store', help = 'From which the tfrecs to split')
    parser.add_argument('--todir', default = '/mnt/fs2/chengxuz/Data/whisker2/tfrecords2', type = str, action = 'store', help = 'To which the tfrecs to split')

    args = parser.parse_args()

    from_dir = os.path.join(args.fromdir, args.key)
    to_dir = os.path.join(args.todir, args.key)

    os.system('mkdir -p %s' % to_dir)

    max_num = 11*144

    file_list = os.listdir(from_dir)
    file_list.sort()
    #print(len(file_list))

    for file_indx in xrange(args.staindx, min(args.staindx+args.lenindx, len(file_list))):
        file_name = file_list[file_indx]
        if not file_name.endswith('tfrecords'):
            print(file_name)
            continue

        file_path = os.path.join(from_dir, file_name)

        record_iterator = tf.python_io.tf_record_iterator(path=file_path)

        writer = None
        now_split = 0
        now_num = 0

        for string_record in record_iterator:
            if writer is None:
                writer, now_split = get_next_split(now_split, file_name, to_dir)

            writer.write(string_record)

            now_num = now_num + 1
            if now_num==max_num:
                now_num = 0
                writer.close()
                writer = None

        writer.close()

if __name__=='__main__':
    main()
