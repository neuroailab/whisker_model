psta=0
osta=0
for psta in 0 1 2
do
python cmd_hdf5.py --pathhdf5 /Users/chengxuz/barrel/bullet/barrle_related_files/hdf5s --pathexe /Users/chengxuz/barrel/bullet/example_build/ExampleBrowser/App_ExampleBrowser --fromcfg /Users/chengxuz/barrel/bullet/barrel_github/cmd_gen_mp4/opt_results/para_ --pathconfig /Users/chengxuz/barrel/bullet/barrle_related_files/configs --generatemode 3 --objindx 7 --smallpsta ${psta} --smallosta ${osta} --mp4flag chair_${psta}_${osta}.mp4
done

psta=0
for osta in 1 2 3
do
python cmd_hdf5.py --pathhdf5 /Users/chengxuz/barrel/bullet/barrle_related_files/hdf5s --pathexe /Users/chengxuz/barrel/bullet/example_build/ExampleBrowser/App_ExampleBrowser --fromcfg /Users/chengxuz/barrel/bullet/barrel_github/cmd_gen_mp4/opt_results/para_ --pathconfig /Users/chengxuz/barrel/bullet/barrle_related_files/configs --generatemode 3 --objindx 7 --smallpsta ${psta} --smallosta ${osta} --mp4flag chair_${psta}_${osta}.mp4
done
