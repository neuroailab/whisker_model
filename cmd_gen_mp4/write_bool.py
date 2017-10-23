import os
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#refer_dir = '/mnt/fs0/chengxuz/Data/whisker/tfrecs_all/tfrecords/Data_force'
#write_dir = '/mnt/fs0/chengxuz/Data/whisker/tfrecs_all/tfrecords/trainflag'
#write_value = 1

refer_dir = '/mnt/fs0/chengxuz/Data/whisker/val_tfrecs/tfrecords_val/Data_force'
write_dir = '/mnt/fs0/chengxuz/Data/whisker/val_tfrecs/tfrecords_val/trainflag'
write_value = 0

file_list = os.listdir(refer_dir)

for file_name in file_list:
    if not file_name.endswith('tfrecords'):
        continue

    break

    tfrec_path = os.path.join(refer_dir, file_name)
    record_iterator = tf.python_io.tf_record_iterator(path=tfrec_path)

    write_path = os.path.join(write_dir, file_name)
    writer = tf.python_io.TFRecordWriter(write_path)

    for string_record in record_iterator:
        example = tf.train.Example(features=tf.train.Features(feature={
            'trainflag': _int64_feature(write_value)}))

        writer.write(example.SerializeToString())
    writer.close()

    #break
