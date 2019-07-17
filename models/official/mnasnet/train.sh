~/anaconda3/envs/tensorflow_p36/bin/mpirun -np 8 -hostfile hosts.txt -mca plm_rsh_no_tree_spawn 1 \
        -bind-to socket -map-by slot \
        -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
        -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
        -x NCCL_SOCKET_IFNAME=ens5 -mca btl_tcp_if_exclude lo,docker0 \
        -x TF_CPP_MIN_LOG_LEVEL=0 \
        python mnasnet_main_hvd.py --use_tpu=False --data_dir=/home/ubuntu/data --model_dir=./results_hvd --train_batch_size=256 --eval_batch_size=256 --train_steps=218949 --skip_host_call=False --data_format='channels_first' --transpose_input=False --use_horovod=True