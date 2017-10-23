import os
import argparse
import cmd_gen

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='The script to gather and reformat the files')
    parser.add_argument('--pathout', default = "/om/user/chengxuz/slurm_out_all", type = str, action = 'store', help = 'Path to slurm output folder')
    parser.add_argument('--outprefix', default = "hyper_server_", type = str, action = 'store', help = 'Prefix of output files')
    parser.add_argument('--outsuffix', default = ".out", type = str, action = 'store', help = 'Suffix of output files')
    parser.add_argument('--indxsta', default = 6220401, type = int, action = 'store', help = 'Index of starting job id')
    parser.add_argument('--indxend', default = 6220496, type = int, action = 'store', help = 'Index of ending job id')
    parser.add_argument('--dirconfigs', default = "/om/user/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results", type = str, action = 'store', help = 'Path to slurm output folder')

    args    = parser.parse_args()

    base_spring_stiffness = []
    spring_stfperunit_list = []
    for indx_spring in xrange(2, 30):
        base_spring_stiffness.append(3964)
        spring_stfperunit_list.append(2517)

    tmp_dict = {
            "basic_str":{"value":8374, "type":"float"}, 
            "base_ball_base_spring_stf":{"value":8374, "type":"float"}, 
            "spring_stfperunit":{"value":2517, "type":"float"}, 
            "base_spring_stiffness":{"value":base_spring_stiffness, "type":"list", "type_in": "float"}, 
            "spring_stfperunit_list":{"value":spring_stfperunit_list, "type":"list", "type_in": "float"}, 
            "linear_damp":{"value":0.66, "help":"Control the linear damp ratio", "type":"float"},
            "ang_damp":{"value":0.015, "help":"Control the angle damp ratio", "type":"float"},
        }
    
    num_cfg = 0
    for curr_indx in xrange(args.indxsta, args.indxend+1):
        tmp_path = os.path.join(args.pathout, "%s%i%s" %(args.outprefix, curr_indx, args.outsuffix) )
        if os.path.isfile(tmp_path):
            print tmp_path,
            unit_indx, _tmp_dict = cmd_gen.recover_from_cfg(tmp_dict, tmp_path)
            none_flag = _tmp_dict is None
            print unit_indx, none_flag
            if none_flag or unit_indx==-1:
                continue

            cfg_filename = os.path.join(args.dirconfigs, "para_%i.cfg" % unit_indx)
            cmd_gen.make_config(_tmp_dict, cfg_filename)
            num_cfg = num_cfg + 1
   
    print(num_cfg)
