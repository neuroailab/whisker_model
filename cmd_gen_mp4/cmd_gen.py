import argparse
import os
from itertools import *
import copy
import multiprocessing
import numpy as np
from get_ratMap import get_wholeS
import copy
import subprocess
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials
import copy
import json
import sys

'''
The script to run the whisker thing and generate the mp4s through command line
'''

host_sys = os.uname()[0]

obj_path_prefix = "/Users/chengxuz/barrel/bullet/bullet3/data/"

if host_sys=='Darwin':
    default_pathconfig = "/Users/chengxuz/barrel/bullet/barrle_related_files/configs"
    default_pathexe = "/Users/chengxuz/barrel/bullet/example_build/Constraints/App_TestHinge"
else:
    default_pathconfig = "/scratch/users/chengxuz/barrel/barrel_relat_files/configs"
    default_pathexe = "/scratch/users/chengxuz/barrel/examples_build/Constraints/App_TestHinge"
    #obj_path_prefix = "/home/chengxuz/barrel/bullet3/data/"
    obj_path_prefix = "/scratch/users/chengxuz/barrel/bullet3/data/"

host_name = os.uname()[1]

if host_name.startswith('node') or host_name == 'openmind7':
    default_pathconfig = "/om/user/chengxuz/barrel/configs"
    default_pathexe = "/om/user/chengxuz/barrel/example_build/Constraints/App_TestHinge"
    obj_path_prefix = "/om/user/chengxuz/barrel/bullet3/data/"
elif host_name=="kanefsky":
    default_pathconfig = "/home/chengxuz/barrel/barrel_github/configs"
    default_pathexe = "/home/chengxuz/barrel/barrel_github/example_build/Constraints/App_TestHinge"
    obj_path_prefix = "/home/chengxuz/barrel/bullet3/data/"

args        = []
all_items   = []
config_dict = {}
para_search = {}
nu_rela_key = ['camera_dist', 'time_limit']
y_len_link = 1.04

def make_config(config_dict, config_filename):
    fout        = open(config_filename, 'w')
    line_tmp    = {"float":"%s=%f\n", "int":"%s=%i\n", "string":"%s=%s\n"}
    line_com    = "#%s\n"

    for key_value in config_dict:
        tmp_key     = config_dict[key_value]
        if "help" in tmp_key:
            fout.write(line_com % tmp_key["help"])
        if not tmp_key["type"]=="list" and not tmp_key["type"]=="list_dict":
            fout.write(line_tmp[tmp_key["type"]] % (key_value, tmp_key["value"]))
        elif tmp_key["type"]=="list":
            #fout.write(line_tmp[tmp_key["type"]] % (key_value, str(tmp_key["value"])[1:-1].replace(',', '') ))
            #fout.write(line_tmp[tmp_key["type"]] % (key_value, str(tmp_key["value"])[1:-1] ))
            for value_list in tmp_key["value"]:
                fout.write(line_tmp[tmp_key["type_in"]] % (key_value, value_list))
        elif tmp_key["type"]=="list_dict":
            for key_d, value_d in enumerate(tmp_key["value"]):
                new_fname = config_filename[:-4] + "_" + str(key_d) + config_filename[-4:]
                make_config(value_d, new_fname)
                fout.write("whisker_config_name=%s\n" % new_fname)
        fout.write("\n")

def my_product(dicts):
    return (dict(izip(dicts, x)) for x in product(*[x['range'] for x in dicts.itervalues()]))

def run_it(ind):
    global all_items 
    global args
    global config_dict
    global para_search

    cmd_tmp1        = "%s --config_filename=%s --mp4=%s --start_demo_name=TestHingeTorque"
    cmd_tmp2        = "%s --config_filename=%s --start_demo_name=TestHingeTorque"

    start_indx  = min(ind * args.mapn, len(all_items))
    end_indx    = min((ind+1)*args.mapn, len(all_items))
    curr_items  = all_items[start_indx: end_indx]

    our_config  = copy.deepcopy(config_dict)
    
    for item in curr_items:

        now_file_name   = ""

        for key_value in item:
            if not key_value=='damp':
                our_config[key_value]["value"]  = item[key_value]
            else:
                for sub_key in para_search[key_value]["key_value"]:
                    our_config[sub_key]["value"]  = item[key_value]
            now_file_name   = now_file_name + para_search[key_value]["short"] + str(para_search[key_value]["range"].index(item[key_value]))

        now_nu  = item["const_numLinks"]
        our_config['initial_poi']["value"]  = now_nu - 1
        #our_config['camera_dist']["value"]  = dist_dict[now_nu]
        for tmp_key in nu_rela_key:
            our_config[tmp_key]["value"]  = our_config[tmp_key]["dict_nu"][now_nu]

        # Make config files
        now_config_fn   = os.path.join(args.pathconfig, now_file_name + ".cfg")
        now_mp4_fn      = os.path.join(args.pathmp4, now_file_name + ".mp4")

        make_config(our_config, now_config_fn)
        if args.mp4flag==1:
            cmd_tmp         = "%s --config_filename=%s --mp4=%s --start_demo_name=TestHingeTorque"
            cmd_str         = cmd_tmp % (args.pathexe, now_config_fn, now_mp4_fn)
        else:
            cmd_tmp         = "%s --config_filename=%s --start_demo_name=TestHingeTorque"
            cmd_str         = cmd_tmp % (args.pathexe, now_config_fn)

        os.system(cmd_str)

def build_array(num_whskr, indx_sta = 0):
    x_pos_base      = []
    y_pos_base      = []
    z_pos_base      = []
    const_numLinks  = []
    qua_list        = []
    yaw_y_base      = []
    pitch_x_base    = []
    roll_z_base     = []

    x_pos_st        = -0.4
    x_pos_step      = 10
    y_pos_va        = 7
    z_pos_st        = 0
    z_pos_step      = 10
    const_num_l     = 25

    qua_st          = -0.1
    yaw_y_base_st   = 0.5
    pitch_x_base_st = 0.3
    roll_z_base_st  = 0.6

    S   = get_wholeS()

    for indx_w in xrange(indx_sta, indx_sta + num_whskr):
        x_pos_base.append(S.C_baseZ[indx_w])
        y_pos_base.append(S.C_baseY[indx_w])
        z_pos_base.append(S.C_baseX[indx_w])

        const_numLinks.append(np.ceil(S.C_s[indx_w]/(y_len_link*2)))
        qua_list.append(-S.C_a[indx_w])

        yaw_y_base.append(S.C_phi[indx_w])
        pitch_x_base.append(S.C_zeta[indx_w])
        roll_z_base.append(S.C_theta[indx_w])

    return {'x':x_pos_base, 'y':y_pos_base, 'z':z_pos_base, 'c':const_numLinks, 
            'yaw':yaw_y_base, 'pitch':pitch_x_base, 'roll':roll_z_base, 'qua':qua_list}

def make_hash(o):

    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).
    """

    if isinstance(o, (set, tuple, list)):

        return tuple([make_hash(e) for e in o])        

    elif not isinstance(o, dict):

        return hash(o)

    new_o = copy.deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)

    return hash(tuple(frozenset(sorted(new_o.items()))))

array_dict      = build_array(1)

every_spring_value = []
inter_spring_value = []
base_spring_stiffness = []
spring_stfperunit_list = []
for indx_spring in xrange(2, 30):
    every_spring_value.append(indx_spring)
    inter_spring_value.append(1)
    base_spring_stiffness.append(3964)
    spring_stfperunit_list.append(2517)

num_whis = len(array_dict['x'])
parameter_each = []
for indx_unit in xrange(num_whis):
    curr_dict = {
        "basic_str":{"value":8374, "type":"float"}, 
        "base_ball_base_spring_stf":{"value":8374, "type":"float"}, 
        "spring_stfperunit":{"value":2517, "type":"float"}, 
        "base_spring_stiffness":{"value":base_spring_stiffness, "type":"list", "type_in": "float"}, 
        "spring_stfperunit_list":{"value":spring_stfperunit_list, "type":"list", "type_in": "float"}, 
        "linear_damp":{"value":0.66, "help":"Control the linear damp ratio", "type":"float"},
        "ang_damp":{"value":0.015, "help":"Control the angle damp ratio", "type":"float"},
    }
    parameter_each.append(curr_dict)

def get_config_dict():

    config_dict     = {"x_len_link":{"value":0.53, "help":"Size x of cubes", "type":"float"}, 
            "y_len_link":{"value":y_len_link, "help":"Size y of cubes", "type":"float"},
            "z_len_link":{"value":0.3, "help":"Size z of cubes", "type":"float"}, 
            "x_pos_base":{"value":array_dict['x'], "help":"Position x of base", "type":"list", "type_in":"float"},
            "y_pos_base":{"value":array_dict['y'], "help":"Position y of base", "type":"list", "type_in":"float"},
            "z_pos_base":{"value":array_dict['z'], "help":"Position z of base", "type":"list", "type_in":"float"},
            "const_numLinks":{"value":array_dict['c'], "help":"Number of units", "type":"list", "type_in":"int"},
            "yaw_y_base":{"value":array_dict['yaw'], "help":"Yaw of base", "type":"list", "type_in":"float"},
            "pitch_x_base":{"value":array_dict['pitch'], "help":"Pitch of base", "type":"list", "type_in":"float"},
            "roll_z_base":{"value":array_dict['roll'], "help":"Roll of base", "type":"list", "type_in":"float"},
            "qua_a_list":{"value":array_dict['qua'], "help":"Quadratic Coefficient", "type":"list", "type_in":"float"},
            "inter_spring":{"value":inter_spring_value, "help":"Number of units between two strings", "type":"list", "type_in": "int"}, 
            "every_spring":{"value":every_spring_value, "help":"Number of units between one strings", "type":"list", "type_in": "int"},

            "parameter_each": {"value":parameter_each, "type": "list_dict"},

            "time_leap":{"value":1.0/240.0, "help":"Time unit for simulation", "type":"float"},
            #"time_leap":{"value":1.0/120.0, "help":"Time unit for simulation", "type":"float"},
            #"camera_dist":{"value":60, "help":"Distance of camera", "type":"float", "dict_nu":{5: 20, 15:45, 25:70}}, 
            "camera_dist":{"value":40, "help":"Distance of camera", "type":"float", "dict_nu":{5: 20, 15:45, 25:70}}, 
            #"camera_yaw":{"value":183, "type":"float"}, 
            "camera_yaw":{"value":21, "type":"float"}, 
            #"camera_pitch":{"value":83, "type":"float"}, 
            "camera_pitch":{"value":270, "type":"float"}, 

            "add_objs":{"value":0, "type":"int"},
            #"obj_filename":{"value":["/Users/chengxuz/barrel/bullet/bullet3/data/teddy.obj"], "type":"list", "type_in":"string"},
            #"obj_filename":{"value":[os.path.join(obj_path_prefix, "teddy.obj")], "type":"list", "type_in":"string"},
            #"obj_filename":{"value":["/Users/chengxuz/barrel/bullet/bullet3/data/cube.obj"], "type":"list", "type_in":"string"},
            "obj_filename":{"value":[os.path.join(obj_path_prefix, "duck.obj")], "type":"list", "type_in":"string"},
            #"obj_scaling_list":{"value":[1,1,1,1], "type":"list", "type_in":"float"},
            #"obj_mass_list":{"value":[100], "type":"list", "type_in":"float"},
            #"obj_filename":{"value":["/Users/chengxuz/barrel/bullet/bullet3/data/sphere8.obj"], "type":"list", "type_in":"string"},
            #"obj_filename":{"value":["/Users/chengxuz/barrel/bullet/bullet3/data/cube.obj"], "type":"list", "type_in":"string"},
            #"obj_scaling_list":{"value":[30,30,30,1], "type":"list", "type_in":"float"},
            "obj_mass_list":{"value":[100000], "type":"list", "type_in":"float"},
            "obj_pos_list":{"value":[-10.1199,10,-18.9956,0], "type":"list", "type_in":"float"},
            #"obj_orn_list":{"value":[0,0,0,1], "type":"list", "type_in":"float"},
            "obj_orn_list":{"value":[0,0,0,1], "type":"list", "type_in":"float"},
            #"obj_speed_list":{"value":[0,-5,0], "type":"list", "type_in":"float"},
            #"obj_speed_list":{"value":[0,-10,0], "type":"list", "type_in":"float"},
            "obj_speed_list":{"value":[0,-12,0], "type":"list", "type_in":"float"},
            "offset_center_pos":{"value":[0,0,-4], "type":"list", "type_in":"float"},
            "control_len":{"value":[40], "type":"list", "type_in":"float"},
            "reset_pos":{"value":1, "type":"int"},
            "reset_speed":{"value":1, "type":"int"},
            "avoid_coll":{"value":1, "type":"int"},
            "avoid_coll_z_off":{"value":5, "type":"float"},
            "avoid_coll_x_off":{"value":10, "type":"float"},
            #"avoid_coll":{"value":0, "type":"int"},
            #"reset_pos":{"value":2, "type":"int"},

            "do_save":{"value":1, "type":"int"},
            "FILE_NAME":{"value":"Select.h5", "type":"string"},
            "num_unit_to_save":{"value":3, "type":"int"},
            "sample_rate":{"value":24, "type":"int"},
            "group_name":{"value":"/Data", "type":"string"},

            "time_limit":{"value":60.0, "help":"Time limit for recording", "type":"float", "dict_nu": {5: 20.0/4, 15: 35.0/4, 25:50.0/4}}, 
            #"time_limit":{"value":12.0, "help":"Time limit for recording", "type":"float", "dict_nu": {5: 20.0/4, 15: 35.0/4, 25:50.0/4}}, 
            #"time_limit":{"value":11.0, "help":"Time limit for recording", "type":"float", "dict_nu": {5: 20.0/4, 15: 35.0/4, 25:50.0/4}}, 
            "initial_str":{"value":10000, "help":"Initial strength of force applied", "type":"float"}, 
            "max_str":{"value":10000, "help":"Max strength of force applied", "type":"float"}, 
            "initial_stime":{"value":3.1/8, "help":"Initial time to apply force", "type":"float"}, 
            "angl_ban_limit":{"value":0.5, "help":"While flag_time is 2, used for angular velocities of rigid bodys to judge whether stop", "type":"float"}, 
            "velo_ban_limit":{"value":0.5, "help":"While flag_time is 2, used for linear velocities of rigid bodys to judge whether stop", "type":"float"}, 
            "force_limit":{"value":40, "help":"While flag_time is 2, used for force states of hinges to judge whether stop", "type":"float"}, 
            "torque_limit":{"value":120, "help":"While flag_time is 2, used for torque states of hinges to judge whether stop", "type":"float"}, 
            "dispos_limit":{"value":50, "help":"While flag_time is 2, used for distance to balance states of rigid bodys to judge whether stop", "type":"float"}, 
            "test_mode":{"value":0, "help":"Whether enter test mode for some temp test codes, default is 0", "type":"int"},
            #"test_mode":{"value":0, "help":"Whether enter test mode for some temp test codes, default is 0", "type":"int"},
            #"force_mode":{"value":2, "help":"Force mode to apply at the beginning, default is 0", "type":"int"},
            #"force_mode":{"value":1, "help":"Force mode to apply at the beginning, default is 0", "type":"int"},
            "force_mode":{"value":-1, "help":"Force mode to apply at the beginning, default is 0", "type":"int"},
            "flag_time":{"value":2, "help":"Whether open time limit", "type":"int"}}
            #"flag_time":{"value":0, "help":"Whether open time limit", "type":"int"}}
            #"flag_time":{"value":1, "help":"Whether open time limit", "type":"int"}}
    return config_dict

config_dict = get_config_dict()
orig_config_dict = copy.deepcopy(config_dict)

#inner_loop = {0: {'force_mode': 0, "initial_str": 30000}, 1: {'force_mode': 1, "initial_str": 10000}, 2: {'force_mode': 2, "initial_str": 10000}, 3: {'force_mode': 2, "initial_str": 8000}}
inner_loop = {0: {'force_mode': 0, "initial_str": 30000}, 1: {'force_mode': 1, "initial_str": 10000}, 2: {'force_mode': 2, "initial_str": 10000}, 3: {'force_mode': 2, "initial_str": 8000}, 4: {'force_mode': 3, "initial_str": -20000}}

def get_value(kwargs, pathconfig =default_pathconfig, pathexe =default_pathexe,  
        coe_curr_dis = 1.0/40.0, coe_min_dis = 1.0, coe_all_time = 20.0, coe_ave_speed = -2):

    for key, value in kwargs.iteritems():
        if key in config_dict:
            config_dict[key]['value'] = value

    all_ret_val = 0
    for key, value in inner_loop.iteritems():
        for key_i, value_i in value.iteritems():
            if key_i in config_dict:
                config_dict[key_i]['value'] = value_i

        hash_value = make_hash(config_dict)
        #print(hash_value)
        #print(pathconfig)
        now_config_fn   = os.path.join(pathconfig, "test_%s.cfg" % str(hash_value))

        make_config(config_dict, now_config_fn)

        tmp_outputs = subprocess.check_output([pathexe, now_config_fn])
        tmp_splits = tmp_outputs.split('\n')
        curr_dis = float(tmp_splits[1].split(':')[1])
        min_dis = float(tmp_splits[2].split(':')[1])
        all_time = float(tmp_splits[3].split(':')[1])
        ave_speed = curr_dis/all_time
        retval = coe_curr_dis*curr_dis + coe_min_dis*min_dis + coe_all_time*all_time + coe_ave_speed*ave_speed

        all_ret_val = all_ret_val + retval

        for key_i, value_i in value.iteritems():
            if key_i in config_dict:
                config_dict[key_i]['value'] = orig_config_dict[key_i]['value']

    return all_ret_val


_default_prior_weight = 1.0

# -- suggest best of this many draws on every iteration
_default_n_EI_candidates = 24

# -- gamma * sqrt(n_trials) is fraction of to use as good
#_default_gamma = 0.25
_default_gamma = 0.15

_default_n_startup_jobs = 100

def my_suggest(new_ids, domain, trials, seed,
            prior_weight=_default_prior_weight,
            n_startup_jobs=_default_n_startup_jobs,
            n_EI_candidates=_default_n_EI_candidates,
            gamma=_default_gamma):
    return tpe.suggest(new_ids, domain, trials, seed,
            prior_weight, n_startup_jobs, n_EI_candidates, gamma)

def re_get_unitparams(config_dict, num_whis = 1, indx_sta = 0):
    array_dict      = build_array(num_whis, indx_sta)

    num_whis = len(array_dict['x'])
    parameter_each = []
    for indx_unit in xrange(num_whis):
        curr_dict = {
            "basic_str":{"value":8374, "type":"float"}, 
            "base_ball_base_spring_stf":{"value":8374, "type":"float"}, 
            "spring_stfperunit":{"value":2517, "type":"float"}, 
            "base_spring_stiffness":{"value":base_spring_stiffness, "type":"list", "type_in": "float"}, 
            "spring_stfperunit_list":{"value":spring_stfperunit_list, "type":"list", "type_in": "float"}, 
            "linear_damp":{"value":0.66, "help":"Control the linear damp ratio", "type":"float"},
            "ang_damp":{"value":0.015, "help":"Control the angle damp ratio", "type":"float"},
        }
        parameter_each.append(curr_dict)

    config_dict["x_pos_base"]["value"] = array_dict['x']
    config_dict["y_pos_base"]["value"] = array_dict['y']
    config_dict["z_pos_base"]["value"] = array_dict['z']
    config_dict["const_numLinks"]["value"] = array_dict['c']
    config_dict["yaw_y_base"]["value"] = array_dict['yaw']
    config_dict["pitch_x_base"]["value"] = array_dict['pitch']
    config_dict["roll_z_base"]["value"] = array_dict['roll']
    config_dict["qua_a_list"]["value"] = array_dict['qua']
    config_dict["parameter_each"]["value"] = parameter_each

    return config_dict

def do_hyperopt(eval_num, use_mongo = False, portn = 23333, db_name = "test_db", exp_name = "exp1", num_whis = 1, indx_sta = 0):

    global config_dict

    print("indx_sta:%i" % indx_sta)
    sys.stdout.flush()

    config_dict = re_get_unitparams(config_dict, num_whis, indx_sta)

    if (use_mongo):
        trials = MongoTrials('mongo://localhost:%i/%s/jobs' % (portn, db_name), exp_key=exp_name)
    else:
        trials = Trials()

    space_base_spring_stiffness = []
    space_spring_stfperunit_list = []
    for indx_spring in xrange(2, 30):
        space_base_spring_stiffness.append(hp.uniform('base_spring_stiffness$%i' % indx_spring, 0, 20000))
        space_spring_stfperunit_list.append(hp.uniform('spring_stfperunit_list$%i' % indx_spring, 0, 20000))

    space_parameter_each = []

    for indx_unit in xrange(num_whis):
        curr_space_dict = {
                "basic_str":{"value":hp.uniform('basic_str', 0, 9000), "type":"float"}, 
                "base_ball_base_spring_stf":{"value":hp.uniform('base_ball_base_spring_stf', 0, 20000), "type":"float"}, 
                "spring_stfperunit":{"value":hp.uniform('spring_stfperunit', 0, 9000), "type":"float"}, 
                "base_spring_stiffness":{"value":space_base_spring_stiffness, "type":"list", "type_in": "float"}, 
                "spring_stfperunit_list":{"value":space_spring_stfperunit_list, "type":"list", "type_in": "float"}, 
                "linear_damp":{"value":hp.uniform('linear_damp', 0, 0.9), "help":"Control the linear damp ratio", "type":"float"},
                "ang_damp":{"value":hp.uniform('ang_damp', 0, 0.9), "help":"Control the angle damp ratio", "type":"float"},
             }
        space_parameter_each.append(curr_space_dict)

    best = fmin(fn=get_value, 
        space=hp.choice('a', [
            {'parameter_each': space_parameter_each},
            ]),
        algo=my_suggest,
        trials=trials,
        max_evals=eval_num)
    print best

def recover_from_cfg(config_dict, pathcfg):
    fin = open(pathcfg, 'r')
    curr_line = fin.readline()
    unit_indx = -1
    while (len(curr_line)>0) and (not curr_line[0]=='{'):
        if curr_line.startswith('indx_sta'):
            unit_indx = int(curr_line.split(':')[1])

        curr_line = fin.readline()
    if len(curr_line)==0:
        return unit_indx, None
    curr_line = curr_line.replace("'", '"')
    curr_line = curr_line.replace('u"', '"')
    #print(curr_line)
    config_add = json.loads(curr_line)

    for key, value in config_add.items():
        if key in config_dict:
            config_dict[key]['value'] = value
        elif "$" in key:
            new_names = key.split("$")
            new_name = new_names[0]
            curr_indx = int(new_names[1])
            config_dict[new_name]['value'][curr_indx - 2] = value
        #print(key, value)

    return unit_indx, config_dict

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='The script to generate the mp4s through command line')
    parser.add_argument('--nproc', default = 4, type = int, action = 'store', help = 'Number of processes')
    parser.add_argument('--pathconfig', default = "/home/chengxuz/barrel/barrel/bullet_demos_extracted/configs", type = str, action = 'store', help = 'Path to config folder')
    parser.add_argument('--pathexe', default = "/home/chengxuz/barrel/build_examples/ExampleBrowser/App_ExampleBrowser", type = str, action = 'store', help = 'Path to App_ExampleBrowser')
    parser.add_argument('--mapn', default = 30, type = int, action = 'store', help = 'Number of items in each processes')

    parser.add_argument('--mp4flag', default = 1, type = int, action = 'store', help = 'Whether generate mp4 files')
    parser.add_argument('--pathmp4', default = "/home/chengxuz/barrel/barrel/cmd_gen_mp4/generated_mp4s", type = str, action = 'store', help = 'Path to mp4 folder')

    parser.add_argument('--testmode', default = 0, type = int, action = 'store', help = 'Whether run the test command or not')
    parser.add_argument('--innernum', default = -1, type = int, action = 'store', help = 'Number of inner loop')
    parser.add_argument('--fromcfg', default = None, type = str, action = 'store', help = 'None means no, if the path of file sent, then get config from the file')
    #parser.add_argument('--fromcfglist', default = 0, type = int, action = 'store', help = 'If 1, then load each whisker parameter')
    parser.add_argument('--indxsta', default = 0, type = int, action = 'store', help = 'Start index of whisker needed')
    parser.add_argument('--indxend', default = 1, type = int, action = 'store', help = 'End index of whisker needed')

    args    = parser.parse_args()

    global config_dict

    config_dict = re_get_unitparams(config_dict, args.indxend - args.indxsta, args.indxsta)

    if args.fromcfg is not None:

        config_dict.pop("parameter_each")
        whisker_config_name     = []
        for curr_indx in xrange(args.indxsta, args.indxend):
            whisker_config_name.append("%s%i.cfg" % (args.fromcfg, curr_indx))
        config_dict["whisker_config_name"] = {"value":whisker_config_name, "type":"list", "type_in": "string"}

    if args.innernum>=0 and args.innernum in inner_loop:
        inner_loop_orig = copy.deepcopy(inner_loop)
        for key in inner_loop_orig:
            if not key==args.innernum:
                inner_loop.pop(key)

    if args.innernum==-1:
        inner_loop = {0: {'force_mode': -1}}
        config_dict['flag_time']['value'] = 0

    if args.testmode==0:
        para_search     = {"basic_str":{"range":[1000, 3000, 5000, 7000], "short":"ba"}, 
                "const_numLinks":{"range":[5, 15, 25], "short":"nu"},
                #"const_numLinks":{"range":[25], "short":"nu"},
                "damp":{"range":[0.1, 0.5, 0.9], "key_value": ["linear_damp", "ang_damp"], "short":"dp"},
                "inter_spring":{"range":[1, 3, 5, 7], "short":"is"},
                "every_spring":{"range":[2, 3, 5, 7, 9, 11, 13, 17, 21], "short":"es"},
                "spring_stiffness":{"range":[300, 500, 700, 900], "short":"ss"}
                }

        for check_item in my_product(para_search):

            right_flag      = 1

            for key_value in check_item:
                if check_item[key_value] not in para_search[key_value]['range']:
                    right_flag  = 0

            if check_item['every_spring'] > check_item["const_numLinks"]:
                right_flag      = 0

            if check_item['inter_spring'] > check_item["const_numLinks"]:
                right_flag      = 0

            if right_flag==1:
                all_items.append(check_item)


        nu_args     = range(int(np.ceil(len(all_items)*1.0/args.mapn)))
        #nu_args     = range(2)
        pool = multiprocessing.Pool(processes=args.nproc)
        r = pool.map_async(run_it, nu_args)
        r.get()
        #print('Done!')
        #run_it(0)
    elif args.testmode==1:
        # Make config files
        now_config_fn   = "test.cfg"
        now_mp4_fn      = "test.mp4"

        for key, value in inner_loop.iteritems():
            for key_i, value_i in value.iteritems():
                if key_i in config_dict:
                    config_dict[key_i]['value'] = value_i

            make_config(config_dict, now_config_fn)
            if args.mp4flag==1:
                cmd_tmp         = "%s --config_filename=%s --mp4=%s --start_demo_name=TestHingeTorque"
                cmd_str         = cmd_tmp % (args.pathexe, now_config_fn, now_mp4_fn)
            else:
                cmd_tmp         = "%s --config_filename=%s --start_demo_name=TestHingeTorque"
                cmd_str         = cmd_tmp % (args.pathexe, now_config_fn)

            os.system(cmd_str)

            for key_i, value_i in value.iteritems():
                if key_i in config_dict:
                    config_dict[key_i]['value'] = orig_config_dict[key_i]['value']
    else:
        now_config_fn   = "test.cfg"

        #print(make_hash(config_dict))
        for key, value in inner_loop.iteritems():
            for key_i, value_i in value.iteritems():
                if key_i in config_dict:
                    config_dict[key_i]['value'] = value_i
            make_config(config_dict, now_config_fn)
            cmd_tmp         = "%s %s"
            cmd_str         = cmd_tmp % (args.pathexe, now_config_fn)

            os.system(cmd_str)

            for key_i, value_i in value.iteritems():
                if key_i in config_dict:
                    config_dict[key_i]['value'] = orig_config_dict[key_i]['value']
