# generate the config files and run the experiments

date_str=`date`
date_str=${date_str// /-}
date_str=${date_str//:/-}
env_name=$1

for trpo_iteration in 20 10 30 40; do
    # generate the config files
    exp_name=${date_str}_${env_name}_trpo_iteration_${trpo_iteration}

    # modify the config files
    cp ./configs/params_${env_name}_template.json ./configs/params_${env_name}.json
    sed -i "s/TRPO_ITERATION/${trpo_iteration}/g" configs/params_${env_name}.json

    # run the experiments
    python main.py --env $1 --exp_name $exp_name --sub_exp_name $1

done
