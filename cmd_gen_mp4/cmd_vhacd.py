import argparse
import os
import sys
import pymongo

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='The script to generate the vhacd .objs for each file')
    parser.add_argument('--startIndx', default = 0, type = int, action = 'store', help = 'Start index of .objs')
    parser.add_argument('--lenIndx', default = 10, type = int, action = 'store', help = 'End index of .objs')
    parser.add_argument('--pathvhacd', default = "/om/user/chengxuz/threedworld_related/v-hacd/build/linux2/test/testVHACD", type = str, action = 'store', help = 'Path to testVHACD')
    parser.add_argument('--dirsave', default = "/om/user/chengxuz/threedworld_related/shapenet_onlyobj/after_vhacd", type = str, action = 'store', help = 'Path to save .objs after v-hacd')
    parser.add_argument('--dirload', default = "/om/user/chengxuz/threedworld_related/shapenet_onlyobj/ShapeNetCore.v2", type = str, action = 'store', help = 'Path to save .objs after v-hacd')
    parser.add_argument('--nport', default = 22334, type = int, action = 'store', help = 'Port number for mongodb')
    parser.add_argument('--soft', default = 1, type = int, action = 'store', help = '1 (default) for check if obj file exists, 0 for generating anyway')
    parser.add_argument('--run', default = 1, type = int, action = 'store', help = '1 (default) generating, 0 for not')

    args    = parser.parse_args()

    conn = pymongo.MongoClient(port=args.nport)
    coll = conn['synthetic_generative']['3d_models']
    test_coll  = coll.find({'type': 'shapenet', 'version': 2})

    cached_coll     = list(test_coll)

    cmd_str_set     = '%s --input %s --output %s --resolution 500000 --maxNumVerticesPerCH 64'
    print(len(cached_coll))

    exist_num       = 0

    for indx in xrange(args.startIndx, min(args.lenIndx + args.startIndx, len(cached_coll))):
        now_coll    = cached_coll[indx]
        dir_name    = now_coll['shapenet_synset'][1:]
        subdir_name = now_coll['id']

        orig_obj_path = os.path.join(args.dirload, dir_name, subdir_name, subdir_name + '.obj')

        new_obj_dir = os.path.join(args.dirsave, dir_name, subdir_name)
        new_obj_path = os.path.join(new_obj_dir, subdir_name + '.obj')

        if args.soft==1 and os.path.isfile(new_obj_path):
            exist_num = exist_num + 1
            continue

        if args.run==0:
            if indx % 100==0:
                print('Now indx %i' % indx)
            continue

        if not os.path.isdir(new_obj_dir):
            os.system('mkdir -p %s' % new_obj_dir)

        cmd_str_now = cmd_str_set % (args.pathvhacd, orig_obj_path, new_obj_path)
        #print(cmd_str_now)

        os.system(cmd_str_now)

    print(exist_num)

