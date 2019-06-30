# generate the config files and run the experiments

date_str=`date`
date_str=${date_str// /-}
date_str=${date_str//:/-}
env_name=$1
env_length=$2

for plan_length in 100 200 500 800 1000; do
    if [ "$plan_length" -gt "$env_length" ]
    then
        continue
    fi

    # generate the config files
    exp_name=${date_str}_${env_name}_depth_search_${env_length}_${plan_length}

    # modify the config files
    cfg_name=./configs/params_${env_name}_depth_search_${env_length}_${plan_length}.json
    cp ./configs/params_${env_name}_depth_search_template.json $cfg_name


    sed -i "s/ENV_LENGTH/${env_length}/g" $cfg_name
    sed -i "s/PLAN_LENGTH/${plan_length}/g" $cfg_name
    path_num=`expr 6000 / ${env_length}`
    sed -i "s/PATH_NUM/${path_num}/g" $cfg_name  # 6000 / env_length

    # run the experiments
    python main.py --env $1 --exp_name $exp_name --sub_exp_name $1 --param_path $cfg_name

done
