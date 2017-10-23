import os
import argparse

# Check whether the objects in the list exist
def check_exist(dirload = "/om/user/chengxuz/threedworld_related/shapenet_onlyobj/after_vhacd", fin_name = 'obj_choice_2.txt'):

    #fin = open('obj_choice.txt', 'r')
    fin = open(fin_name, 'r')
    lines = fin.readlines()

    for line in lines:
        split_line = line.split(' ')
        dir_name = split_line[0][1:]
        subdir_name = split_line[1][:-1]

        new_obj_path = os.path.join(dirload, dir_name, subdir_name, subdir_name + '.obj')

        if not os.path.isfile(new_obj_path):
            print(line)

def get_list(fin_name, dirload):
    fin = open(fin_name, 'r')
    lines = fin.readlines()

    ret_list = []

    for line in lines:
        split_line = line.split(' ')
        dir_name = split_line[0][1:]
        subdir_name = split_line[1][:-1]

        new_obj_path = os.path.join(dirload, dir_name, subdir_name, subdir_name + '.obj')
        ret_list.append((subdir_name, new_obj_path))

    return ret_list

def main():
    parser = argparse.ArgumentParser(description='The script to actually generate the dataset')
    parser.add_argument('--objsta', default = 0, type = int, action = 'store', help = 'Start index in the object list')
    parser.add_argument('--objlen', default = 1, type = int, action = 'store', help = 'Length for generating')
    parser.add_argument('--seedbas', default = 0, type = int, action = 'store', help = 'Seed basis for randomization')
    parser.add_argument('--bigsamnum', default = 1, type = int, action = 'store', help = 'Sampling number for every object')
    parser.add_argument('--savedir', default = '/om/user/chengxuz/Data/barrel_dataset/raw_hdf5', type = str, action = 'store', help = 'Directory to save the generated hdf5s')
    parser.add_argument('--loaddir', default = "/om/user/chengxuz/threedworld_related/shapenet_onlyobj/after_vhacd", type = str, action = 'store', help = 'Where to get the objects after vhacd')
    parser.add_argument('--objlist', default = 'obj_choice_2.txt', type = str, action = 'store', help = 'The file having the object information')
    parser.add_argument('--checkmode', default = 1, type = int, action = 'store', help = '1 means that if the file already exists in the right size, then avoid running, 0 means generate it anyway')

    # Platform related parameters, default is for openmind7
    parser.add_argument('--pathexe', default = "/om/user/chengxuz/barrel/example_build/Constraints/App_TestHinge", type = str, action = 'store', help = 'Path to App_ExampleBrowser')
    parser.add_argument('--fromcfg', default = "/om/user/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_", type = str, action = 'store', help = 'None means no, if the path of file sent, then get config from the file')
    parser.add_argument('--pathconfig', default = "/om/user/chengxuz/barrel/configs_2", type = str, action = 'store', help = 'Path to config folder')

    #cmd_str = "python cmd_hdf5.py --pathhdf5 %s --pathexe %s --fromcfg %s --pathconfig %s --objindx %s --generatemode 3 --testmode 2 --hdf5suff %s --smallolen 4 --smallplen 3 --randseed %i --bigsamnum %i --checkmode %i"
    cmd_str = "python cmd_hdf5.py --pathhdf5 %s --pathexe %s --fromcfg %s --pathconfig %s --objindx %s --generatemode 3 --testmode 1 --hdf5suff %s --smallolen 4 --smallplen 3 --randseed %i --bigsamnum %i --checkmode %i"

    args    = parser.parse_args()

    obj_list = get_list(args.objlist, args.loaddir)

    for obj_indx in xrange(args.objsta, min(args.objsta + args.objlen, len(obj_list))):
        now_hdf5suff, now_objindx = obj_list[obj_indx]
        now_randseed = args.seedbas + obj_indx

        now_cmd = cmd_str % (args.savedir, args.pathexe, args.fromcfg, args.pathconfig, now_objindx, now_hdf5suff, now_randseed, args.bigsamnum, args.checkmode)

        os.system(now_cmd)

if __name__=='__main__':
    main()
