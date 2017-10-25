Codes for reproducing results in paper: https://arxiv.org/abs/1706.07555

# Whisker model and dataset generating

Codes for building whisker model and generating datasets include codes in folders `bullet_demos_extracted/` and `cmd_gen_mp4/`.

## Whisker model

### Before compiling

For compiling the whisker model, you need to have:

- `cmake`
- `boost`, for parsing the config files. Local installation is supported.
- `Bullet`, we use it as physic engine. We also require that the build of bullet is done in `bullet_build` under bullet source code directory. Local installation is supported.
- `hdf5`, hdf5 is needed to generate the dataset. Only global installation is supported.

Currently, we only support Linux and Mac OS systems. Windows are not supported now.

Our whisker model is based on demos in [Bullet](https://github.com/bulletphysics/bullet3). Therefore, all requirements of Bullet demos also apply here.
As the basic framework and some useful tools are borrowed from these demos, building the whisker model requires building all the demos in folder `bullet_demos_extracted/`, which includes demos extracted by us.
Specifically, we implement our whisker model in demo `TestHingeTorque` and the actual code is in `bullet_demos_extracted/examples/Constraints/TestHingeTorque.cpp`.

### How to compile

Make a new directory in any place you want, then run following command for compiling: `cmake -D BULLET_PHYSICS_SOURCE_DIR:SPRING=/path/to/your/bullet/code/repo/ /path/to/this/repo/bullet_demos_extracted/examples/`. 
The reason we need path of bullet source code as input is that we need one library file compiled during building Bullet, which is also why we require that the build of bullet is done in `bullet_build` under bullet source code directory.

If you install bullet and boost locally, then you also need to specify the `BOOST_ROOT` and `BULLET_ROOT` in cmake command by inserting `-D BOOST_ROOT:SPRING=/path/to/your/boost/installation/` and `-D BULLET_ROOT:SPRING=/path/to/your/bullet/installation/` before `/path/to/this/repo/bullet_demos_extracted/examples/`.

After successfully running cmake command, you could run `make` under that directory to actually make the model.
Once `make` is done, change directory to `/path/to/your/build/ExampleBrowser/` and then run `./App_ExampleBrowser --start_demo_name=TestHingeTorque`. 
If everything is correct, you will see one single whisker behaving unnaturalistically, as we need to provide correct config file to the program using following commands. 

### Provide correct config

To take a quick view of whisker model used in our experiment, run following command under folder `cmd_gen_mp4/`: `python cmd_gen.py --mp4flag 0 --testmode 1 --pathexe /path/to/your/build/ExampleBrowser/App_ExampleBrowser --fromcfg /path/to/this/repo/cmd_gen_mp4/opt_results/para_ --indxend 31`. 
Like interacting in Bullet demos, you could use "Ctrl + Pressing left mouse + rotate" within the window to rotate the view and "Pressing left mouse + move" to try to apply forces to the whiskers.

Within the command we use, `mp4flag` determines whether the program will generate a video in mp4 format (`ffmpeg` required) and `fromcfg` specifies the location of specific parameters for each whiskers, which we got from behavior optimization mentioned in the paper.
You can also read the source code in `cmd_gen.py` to understand more of those parameters. 
Especially, if you want to use the whisker model in your way, you could read each parameter in `config_dict` in `cmd_gen.py`, as these parameters are sent to the whisker model to modify its behavior.

If you are also interested in reproducing the behavior optimization results by yourself, please contact [Chengxu](mailto:chengxuz@stanford.edu). The code is also provided here, but reproduction is complex and may not be of general interest.

## Dataset generating

### Preprocessing objects from ShapeNet

We use objects in [ShapeNet](https://www.shapenet.org/) to generate the dataset.
After downloading the 3D models, you need to use [v-hacd](https://github.com/kmammou/v-hacd) to process all the models to transfer each object into a set of "near" convex parts.
The parameter we used in v-hacd is `--resolution 500000 --maxNumVerticesPerCH 64`.
After processing the models, the path of one model should be organized as following example: `/path/to/your/models/02691156/a6693555a4c0bd47434e905131c8d6c6/a6693555a4c0bd47434e905131c8d6c6.obj`.

We sampled 9981 objects from ShapeNet to get a balanced distribution in categories (see our paper for details, the actual code is in `get_obj_list.py`). 
The object information we used is stored in `obj_choice_2.txt` under `cmd_gen_mp4/`.

### Generate the dataset in hdf5s

To generate the dataset using those 9981 objects, we use non-interactive whisker model, which should be at `/path/to/your/build/Constraints/App_TestHinge`. Starting `App_TestHinge` will not start a winodw and the simulation will be done in the same way as starting `App_ExampleBrowser`. Besides, another folder (`config_folder`) needs to be created to hold all the configs generated during dataset generation. The command to generate the whole dataset under `cmd_gen_mp4/` is as following:

```
python cmd_dataset.py --objsta 0 --objlen 9981 --bigsamnum 24 --pathexe /path/to/your/build/Constraints/App_TestHinge --fromcfg /path/to/this/repo/cmd_gen_mp4/opt_results/para_ --pathconfig /path/to/your/config_folder --savedir /path/to/store/hdf5s --loaddir /path/to/your/models --seedbas 0
```

This command will generate `9981*24` hdf5s in `/path/to/store/hdf5s`. 
You can parallel it by running multiple commands with different `--objsta` (starting index of objects for generation, between 0 and 9980) and `--objlen` (number of objects for generation). 
And here `--bigsamnum 24` means 24 independent settings will be generated for each object.
We will use `1/13` of those hdf5s as validation dataset.

With `--seedbas 10000 --bigsamnum 2` and different folder for hdf5s, we can generate validation dataset as well. (Here we set seedbas to be 10000 as in the program, `seedbas + objIndex` will be used as random seed for each object) 

### Generate tfrecords from hdf5s

After all the hdf5s have been generated, we use `cmd_to_tfr_bycat.py` under `cmd_gen_mp4/` to generate tfrecords needed to train the models using tensorflow.

The command is as following:

```
python cmd_to_tfr_bycat.py --catsta 0 --catlen 117 --seedbas 0 --loaddir /path/to/store/hdf5s --savedir /path/to/store/tfrecords --suffix strain --bigsamnum 24
```

Here, we are generating tfrecords by each category. There are overall 117 categories. 
Parameter `catsta` indicates the starting index of this generation and `catlen` is the number of categories this generation will cover. 
Parameter `suffix` is just the suffix of tfrecord names, which we will use later to distinguish train/val split.

If you want to generate tfrecords for validation, the command can be modified as following:

```
python cmd_to_tfr_bycat.py --catsta 0 --catlen 117 --seedbas 10000 --loaddir /path/to/store/validation/hdf5s --savedir /path/to/store/tfrecords --suffix sval --bigsamnum 2
```

# Network training

Codes for training deep neural networks reported in paper are in folder `train_barrel_net/`.
