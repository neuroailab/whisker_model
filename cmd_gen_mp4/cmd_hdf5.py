import argparse
import os
from itertools import *
import copy
from cmd_gen import *
import numpy as np

def get_speed_list(mode=0):
    #return [[0,-12,0], [0, -10, 0], [0, -8, 0]]
    if mode==0:
        return [[0,-12,0], [0, -10, 0], [0, -8, 0]]
    elif mode==1:
        return [[0,-12.5,0], [0, -10.5, 0], [0, -8.5, 0]]
    elif mode==2:
        start_speed = -14
        end_speed = -7

        ret_list = []

        for now_speed in np.arange(start_speed, end_speed):
            now_speed_list = [0, now_speed, 0]
            ret_list.append(now_speed_list)

        start_speed = -14.5
        end_speed = -7.5

        for now_speed in np.arange(start_speed, end_speed):
            now_speed_list = [0, now_speed, 0]
            ret_list.append(now_speed_list)

        return ret_list
    elif mode==3:
        return [0, -14+7*np.random.rand(), 0]
    else:
        return [[0,-12,0], [0, -10, 0], [0, -8, 0]]

def qua_from_euler(euler):
    cos_x = np.cos(euler[0]/2)
    sin_x = np.sin(euler[0]/2)
    cos_y = np.cos(euler[1]/2)
    sin_y = np.sin(euler[1]/2)
    cos_z = np.cos(euler[2]/2)
    sin_z = np.sin(euler[2]/2)

    qua_0 = cos_x*cos_y*cos_z + sin_x*sin_y*sin_z
    qua_1 = sin_x*cos_y*cos_z - cos_x*sin_y*sin_z
    qua_2 = cos_x*sin_y*cos_z + sin_x*cos_y*sin_z
    qua_3 = cos_x*cos_y*sin_z - sin_x*sin_y*cos_z

    return [qua_0, qua_1, qua_2, qua_3]

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return [w, x, y, z]

def get_orn_list(mode=0):
    if mode==0:
        return [[0,0,0,1], [1,0,0,0], [0,1,1,1], [1,1,1,1]]
    elif mode==1:
        return [[0,0,0,1], [1,0,0,0], [0,1,1,1], [1,1,1,1]]
    elif mode==2:
        ret_val = []
        offset_unit = (5.0/180.0)*np.pi

        for offset_mul in xrange(4):
            if offset_mul==2:
                continue
            start_orn = []
            for indx_tmp in xrange(3):
                start_orn.append(offset_unit*offset_mul)

            ret_val.append(qua_from_euler(start_orn))

            delta_deg = np.pi/2

            #for which_ax in xrange(3):
            for which_ax in xrange(2):
                #for mul_change_deg in xrange(3):
                for mul_change_deg in xrange(2):
                    change_deg  = mul_change_deg*delta_deg

                    new_orn     = copy.deepcopy(start_orn)
                    new_orn[which_ax] = new_orn[which_ax] + change_deg

                    ret_val.append(qua_from_euler(new_orn))

        return ret_val
    elif mode==3:
        return qua_from_euler([np.random.rand()*2*np.pi, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi])
    else:
        return [[0,0,0,1], q_mult([0,0,0,1], qua_from_euler([0, 0, 2*np.pi/2])), [1,0,0,0], [0,1,1,1], [1,1,1,1]]


def get_scale_list(mode=0):
    if mode==0:
        return [[40], [30], [50]]
    elif mode==1:
        return [[44], [34], [54]]
    elif mode==2:
        #return [[60], [70], [80]]
        ret_val = []

        start_sc = 30
        #start_sc = 25
        
        end_sc = 90
        #end_sc = 91
        #end_sc = 136
        step_sc = 10

        for inter_sc in xrange(start_sc, end_sc, step_sc):
            ret_val.append([inter_sc])
        #return [[60], [70], [80]]
        return ret_val
    elif mode==3:
        #return [30 + 60*np.random.rand()]
        return [25 + 110*np.random.rand()]
    else:
        return [[40], [30], [50]]

def get_pos_list(mode=0):
    center_pos = [-10.1199,-13.1702,-22.9956-4]
    #start_pos = [-10.1199,10,-22.9956,0]
    #start_pos = [-11.1199,12,-20.9956,0]

    if mode==3:
        return [center_pos[0] + (np.random.rand() - 0.5)*20, 13 + (np.random.rand() - 0.5)*8, center_pos[2] + (np.random.rand() - 0.5)*20, 0]

    if mode==0:
        start_poses = [[-10.1199,10,-22.9956,0]]
    elif mode==2:
        start_poses = [[-10.1199,10,-22.9956,0]]
        change_pos = [-2, 2, 2]
        for indx_tmp in xrange(3):
            new_pos = copy.deepcopy(start_poses[0])
            for indx_tmp2 in xrange(3):
                if indx_tmp>0:
                    new_pos[indx_tmp2] = new_pos[indx_tmp2] + change_pos[indx_tmp2]*indx_tmp
                else:
                    new_pos[indx_tmp2] = new_pos[indx_tmp2] + change_pos[indx_tmp2]*3
            start_poses.append(new_pos)

    else:
        start_poses = [[-11.1199,12,-24.9956,0]]

    if not mode==2:
        deg_aways = [(10.0/180.0)*np.pi, -(10.0/180.0)*np.pi]
    else:
        change_deg = 10.0
        deg_aways = []
        #for indx_tmp in xrange(1, 4):
        for indx_tmp in xrange(1, 3):
            deg_aways.append( change_deg*indx_tmp/180.0*np.pi)
            deg_aways.append(-change_deg*indx_tmp/180.0*np.pi)

    which_axs = [0,2]

    ret_list = copy.deepcopy(start_poses)

    for start_pos in start_poses:
        r_now   = start_pos[1] - center_pos[1]
        for deg_away in deg_aways:
            for which_ax in which_axs:
                new_pos = copy.deepcopy(start_pos)
                new_pos[which_ax]   = new_pos[which_ax] + r_now*np.sin(deg_away)
                new_pos[1]          = center_pos[1] + r_now*np.cos(deg_away)

                #if not new_pos[2] > start_pos[2] + 1:
                #print(new_pos)
                ret_list.append(new_pos)
                #print(len(ret_list))

    return ret_list

def set_basic_value(args):

    config_dict = get_config_dict()
    config_dict = re_get_unitparams(config_dict, args.indxend - args.indxsta, args.indxsta)

    if args.fromcfg is not None:

        config_dict.pop("parameter_each")
        whisker_config_name     = []
        for curr_indx in xrange(args.indxsta, args.indxend):
            whisker_config_name.append("%s%i.cfg" % (args.fromcfg, curr_indx))
        config_dict["whisker_config_name"] = {"value":whisker_config_name, "type":"list", "type_in": "string"}

    config_dict["add_objs"]["value"] = 1
    config_dict["time_limit"]["value"] = 11.0
    config_dict["flag_time"]["value"] = 1
    config_dict["camera_yaw"]["value"] = 183
    config_dict["camera_pitch"]["value"] = 83

    orig_config_dict = copy.deepcopy(config_dict)

    return config_dict, orig_config_dict

def generate_iter_list(args):

    ret_val = []

    obj_path = ''
    hdf5_prefix = ''

    if args.objindx.isdigit():
        args.objindx = int( args.objindx )
        if args.objindx==0:
            obj_path = [os.path.join(obj_path_prefix, "duck_vhacd.obj")]
            hdf5_prefix = "duck"
        elif args.objindx==1:
            obj_path = [os.path.join(obj_path_prefix, "teddy.obj")]
            hdf5_prefix = "teddy"
        elif args.objindx==2:
            obj_path = [os.path.join(obj_path_prefix, "11d2b7d8c377632bd4d8765e3910f617.obj")]
            hdf5_prefix = "test"
        elif args.objindx==3:
            obj_path = [os.path.join(obj_path_prefix, "3957332e2d23ff61ce2704286124606b.obj")]
            hdf5_prefix = "test"
        elif args.objindx==4:
            obj_path = [os.path.join(obj_path_prefix, "hat_aftervhacd.obj")]
            hdf5_prefix = "aftervhacd"
        elif args.objindx==5:
            obj_path = [os.path.join(obj_path_prefix, "hat_aftervhacd_2.obj")]
            hdf5_prefix = "aftervhacd"
        elif args.objindx==6:
            obj_path = [os.path.join(obj_path_prefix, "teddy2_VHACD_CHs.obj")]
            hdf5_prefix = "teddy"
        elif args.objindx==7:
            obj_path = [os.path.join(obj_path_prefix, "chair_aftervhacd.obj")]
            hdf5_prefix = "chair"
        elif args.objindx==8:
            obj_path = [os.path.join(obj_path_prefix, "cube.obj")]
            hdf5_prefix = "cube"
        elif args.objindx==9:
            obj_path = [os.path.join(obj_path_prefix, "duck.obj")]
            hdf5_prefix = "duck"
    else:
        obj_path = [args.objindx]
        hdf5_prefix = "Data"

    if args.generatemode<=2:
        if args.generatemode>0:
            pos_list = get_pos_list(args.generatemode)
            speed_list = get_speed_list(args.generatemode)
            orn_list = get_orn_list(args.generatemode)
            scale_list = get_scale_list(args.generatemode)
            print(len(pos_list), len(speed_list), len(orn_list), len(scale_list))
        else:
            pos_list = [get_pos_list(3)]
            #pos_list = [[-10.1199,15,-22.9956,0]]
            speed_list = [get_speed_list(3)]
            #speed_list = [[0,0.1,0]]
            orn_list = [get_orn_list(3)]
            #orn_list = [[0,0,0,-1]]
            scale_list = [get_scale_list(3)]

        for indx_pos_now in xrange(args.pindxsta, min(args.pindxsta + args.pindxlen, len(pos_list))):
            for indx_scale_now in xrange(args.scindxsta, min(args.scindxsta + args.scindxlen, len(scale_list))):
                for indx_speed_now in xrange(args.spindxsta, min(args.spindxsta + args.spindxlen, len(speed_list))):
                    for indx_orn_now in xrange(args.oindxsta, min(args.oindxsta + args.oindxlen, len(orn_list))):
                        now_add_dict = {}

                        now_add_dict['obj_pos_list'] = pos_list[indx_pos_now]
                        now_add_dict['control_len'] = scale_list[indx_scale_now]
                        now_add_dict['obj_speed_list'] = speed_list[indx_speed_now]
                        now_add_dict['obj_orn_list'] = orn_list[indx_orn_now]
                        now_add_dict['obj_filename'] = obj_path

                        hash_value = make_hash(now_add_dict)
                        #hash_value = "%i_%i_%i_%i_%i" % (hash_value, indx_pos_now, indx_scale_now, indx_speed_now, indx_orn_now)
                        hash_value = "%i_%i_%i_%i" % (indx_pos_now, indx_scale_now, indx_speed_now, indx_orn_now)

                        now_add_dict["FILE_NAME"] = os.path.join(args.pathhdf5, "%s_%s.hdf5" % (hdf5_prefix, hash_value))
                        now_add_dict["_hash_value"] = hdf5_prefix + hash_value

                        ret_val.append(now_add_dict)

    else:
        np.random.seed(args.randseed)

        offset_center_pos = [0,0,-4]

        offset_unit = 20

        for which_big in xrange(args.bigsamnum):
            pos_now = get_pos_list(args.generatemode)
            speed_now = get_speed_list(args.generatemode)
            orn_now = get_orn_list(args.generatemode)
            scale_now = get_scale_list(args.generatemode)

            if args.whichbig>-1:
                if not args.whichbig==which_big:
                    continue

            if args.whichcontrol>0:
                if args.whichcontrol==1:
                    pos_now = [-10.1199,10,-22.9956,0]
                if args.whichcontrol==2:
                    speed_now = [0, -10.5, 0]
                if args.whichcontrol==3:
                    orn_now = [0,0,0,1]
                if args.whichcontrol==4:
                    scale_now = [75]
                if args.whichcontrol==5:
                    scale_now = [75]
                    orn_now = [0,0,0,1]
                if args.whichcontrol==6:
                    scale_now = [75]
                    speed_now = [0, -10.5, 0]

            for which_smallp in xrange(args.smallpsta, min(args.smallpsta + args.smallplen, 3)):
                for which_smallo in xrange(args.smallosta, min(args.smallosta + args.smallolen, 4)):
                    new_pos = copy.deepcopy(pos_now)
                    new_offset = copy.deepcopy(offset_center_pos)
                    if which_smallp==1:
                        new_pos[0] = new_pos[0] - offset_unit
                        new_offset[0] = new_offset[0] - offset_unit
                    elif which_smallp==2:
                        new_pos[0] = new_pos[0] + offset_unit
                        new_offset[0] = new_offset[0] + offset_unit

                    new_orn = q_mult(orn_now, qua_from_euler([0, 0, which_smallo*np.pi/2]))

                    now_small_indx = which_smallp*4 + which_smallo

                    group_name = '/Data%i' % now_small_indx
                    hdf5_prefix = 'Data%i_%i' % (args.randseed, which_big)
                    hash_value = '%s_%i_%i_%i' % (args.hdf5suff, args.randseed, which_big, now_small_indx)

                    now_add_dict = {}

                    now_add_dict['obj_pos_list'] = new_pos
                    now_add_dict['control_len'] = scale_now
                    now_add_dict['obj_speed_list'] = speed_now
                    now_add_dict['obj_orn_list'] = new_orn
                    now_add_dict['obj_filename'] = obj_path
                    now_add_dict["FILE_NAME"] = os.path.join(args.pathhdf5, "%s_%s.hdf5" % (hdf5_prefix, args.hdf5suff))
                    now_add_dict["offset_center_pos"] = new_offset
                    now_add_dict["group_name"] = group_name

                    now_add_dict['_hash_value'] = hash_value

                    ret_val.append(now_add_dict)

    return ret_val

def update_config_dict(config_dict, now_add_dict):
    for key_add in now_add_dict:
        if not key_add.startswith('_'):
            config_dict[key_add]['value'] = now_add_dict[key_add]

    return config_dict

if __name__=="__main__":

    # General settings
    parser = argparse.ArgumentParser(description='The script to generate the hdf5 data through command line')
    parser.add_argument('--pathhdf5', default = "/home/chengxuz/barrel/related_files/hdf5_files", type = str, action = 'store', help = 'Path to hdf5 folder')
    parser.add_argument('--pathexe', default = "/home/chengxuz/barrel/build_examples/Constraints/App_TestHinge", type = str, action = 'store', help = 'Path to App_ExampleBrowser')
    #parser.add_argument('--pathexe', default = "/home/chengxuz/barrel/build_examples/ExampleBrowser/App_ExampleBrowser", type = str, action = 'store', help = 'Path to App_ExampleBrowser')
    parser.add_argument('--fromcfg', default = "/home/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_", type = str, action = 'store', help = 'None means no, if the path of file sent, then get config from the file')
    parser.add_argument('--pathconfig', default = "/home/chengxuz/barrel/related_files/configs", type = str, action = 'store', help = 'Path to config folder')
    parser.add_argument('--indxsta', default = 0, type = int, action = 'store', help = 'Start index of whisker needed')
    parser.add_argument('--indxend', default = 31, type = int, action = 'store', help = 'End index of whisker needed')
    parser.add_argument('--testmode', default = 1, type = int, action = 'store', help = 'Whether run the test command or not')
    parser.add_argument('--generatemode', default = 0, type = int, action = 'store', help = 'Control the parameter sampling method, >2 means dataset generation method, 3*4 sweeps')
    parser.add_argument('--checkmode', default = 0, type = int, action = 'store', help = 'Whether run the check first before running. 0 (default) for no, 1 for yes, 2 for write the information to some file')
    parser.add_argument('--mp4flag', default = None, type = str, action = 'store', help = 'Whether generate mp4 files, if not None, will be used as mp4 name')

    # Original parameters, not used for actual dataset generation
    parser.add_argument('--spindxsta', default = 0, type = int, action = 'store', help = 'Start index of speed')
    parser.add_argument('--spindxlen', default = 1, type = int, action = 'store', help = 'Length index of speed')
    parser.add_argument('--scindxsta', default = 0, type = int, action = 'store', help = 'Start index of scale')
    parser.add_argument('--scindxlen', default = 1, type = int, action = 'store', help = 'Length index of scale')
    parser.add_argument('--oindxsta', default = 0, type = int, action = 'store', help = 'Start index of orn')
    parser.add_argument('--oindxlen', default = 1, type = int, action = 'store', help = 'Length index of orn')
    parser.add_argument('--pindxsta', default = 0, type = int, action = 'store', help = 'Start index of pos')
    parser.add_argument('--pindxlen', default = 1, type = int, action = 'store', help = 'Length index of pos')

    parser.add_argument('--objindx', default = '0', type = str, action = 'store', help = 'Object index, 0 for duck, 1 for teddy, if it is a string, then it will be used as parameter directly')
    parser.add_argument('--whichcontrol', default = 0, type = int, action = 'store', help = 'Which control among speed, scale, orn, position to use, 0 means no')

    # Parameters used for dataset generating
    parser.add_argument('--bigsamnum', default = 1, type = int, action = 'store', help = 'The big sampling number, controlling how many different sampling settings will be drawn')
    parser.add_argument('--smallpsta', default = 0, type = int, action = 'store', help = 'Small sampling (3*4 sampling), start index for position')
    parser.add_argument('--smallplen', default = 1, type = int, action = 'store', help = 'Length for position')
    parser.add_argument('--smallosta', default = 0, type = int, action = 'store', help = 'Start index for orn')
    parser.add_argument('--smallolen', default = 1, type = int, action = 'store', help = 'Length for orn')
    parser.add_argument('--hdf5suff', default = '', type = str, action = 'store', help = 'Suffix for hdf5 file saved')
    parser.add_argument('--randseed', default = 0, type = int, action = 'store', help = 'Seed for randomization')
    parser.add_argument('--whichbig', default = -1, type = int, action = 'store', help = 'Controlling which big sample number to use')

    args    = parser.parse_args()

    config_dict, orig_config_dict = set_basic_value(args)

    now_add_dicts = generate_iter_list(args)

    os.system('mkdir -p %s' % args.pathhdf5)

    exist_num = 0
    not_exist = 0

    for now_add_dict in now_add_dicts:
        config_dict = update_config_dict(config_dict, now_add_dict)

        #if not 'Data9160_6_7c078d8ceed96b2a115a312301bdd286.hdf5' in config_dict["FILE_NAME"]["value"]:
        #    continue
        #if not 'Data15554_10_da92c8d35fabe4093a67185d75524e9c.hdf5' in config_dict["FILE_NAME"]["value"]:
        #    continue

        if not 'Data4039_9_6e5bf008a9259e95fa80fb391ee7ccee.hdf5' in config_dict["FILE_NAME"]["value"]:
            #continue
            pass

        if not 'Data3' in config_dict["group_name"]["value"]:
            #continue
            pass

        size_wanted = 14792976
        if args.smallplen*args.smallolen==1:
            size_wanted = 1233100

        if args.checkmode==1:
            if (os.path.exists(config_dict["FILE_NAME"]["value"]) and (os.path.getsize(config_dict["FILE_NAME"]["value"])==size_wanted)):
                exist_num = exist_num + 1
                print(config_dict["FILE_NAME"]["value"])
                sys.stdout.flush()
                continue
            else:
                not_exist = not_exist + 1
        elif args.checkmode==2:
            info_filename = config_dict["FILE_NAME"]["value"]
            info_filename = info_filename.replace('raw_hdf5', 'raw_info')
            info_filename = info_filename[:-4] + 'txt'

            fout = open(info_filename, 'a')
            fout.write("%s\n" % str(now_add_dict))
            fout.close()

            continue

        now_config_fn   = os.path.join(args.pathconfig, "test_%s.cfg" % now_add_dict["_hash_value"])

        make_config(config_dict, now_config_fn)

        if args.mp4flag is None:
            if args.testmode==1:
                cmd_tmp         = "%s --config_filename=%s --start_demo_name=TestHingeTorque"
            else:
                cmd_tmp         = "%s %s"
            cmd_str         = cmd_tmp % (args.pathexe, now_config_fn)
        else:
            cmd_tmp         = "%s --config_filename=%s --mp4=%s --start_demo_name=TestHingeTorque"
            cmd_str         = cmd_tmp % (args.pathexe, now_config_fn, args.mp4flag)

        os.system(cmd_str)

    if args.checkmode>=1:
        print(exist_num)
