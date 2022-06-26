srun --qos=16gpu3d  --partition=common --gres=gpu:0 --time=3-0 --nodelist=asusgpu2 --constraint=scidatalg  --constraint=homedir --pty /bin/bash -i
