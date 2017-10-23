import cmd_gen
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='The script to run the hyperopt')
    parser.add_argument('--neval', default = 100, type = int, action = 'store', help = 'Number of eval_num')
    parser.add_argument('--usemongo', default = 0, type = int, action = 'store', help = 'Whether to use mongodb, 0 is False, 1 is true')
    parser.add_argument('--portn', default = 23333, type = int, action = 'store', help = 'Port number')
    parser.add_argument('--dbname', default = "test_db", type = str, action = 'store', help = 'Database name')
    parser.add_argument('--expname', default = "exp1", type = str, action = 'store', help = 'Experiment name')
    parser.add_argument('--indxsta', default = 0, type = int, action = 'store', help = 'Port number')


    args    = parser.parse_args()

    cmd_gen.do_hyperopt(args.neval, use_mongo=(args.usemongo==1), portn=args.portn, db_name=args.dbname, exp_name=args.expname, indx_sta = args.indxsta)
