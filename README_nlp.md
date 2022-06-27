1. Clone the repo
1. `cd poleval-2021`
1. `python3 -m venv venv`
1. `source venv/bin/activate`
1. `cd ./overwrite`
1. `sh overwrite.sh`
1. `cd ..`
1. put archive.tar.gz in `./data/`
1. `cd ./data`
1. `tar -xzvf archive.tar.gz`
1. `cd ..`
1.  `srun --partition=common --qos=gsn --gres=gpu:n --time=4:00:00 python3 main.py model=(base|large|xxl|plt5-small) task=question_answering(|_cz|_plt5)`

All models: ./config/model
All tasks: ./config/task