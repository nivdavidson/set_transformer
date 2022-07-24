#! /bin/sh

#SBATCH --partition=studentbatch
#SBATCH --job-name=set_trans
#SBATCH --output=set_trans.output
#SBATCH --error=set_trans.error
#SBATCH --gpus=1

/home/yandex/MLW2022/nivdavidson/anaconda3/envs/MLenv/bin/python3 /home/yandex/MLW2022/nivdavidson/set_transformer/main_particle_collision.py --signal_data_path /home/yandex/MLW2022/nivdavidson/cleaned_data_2022_07_04/signal/signal-0.pkl --background_data_path /home/yandex/MLW2022/nivdavidson/cleaned_data_2022_07_04/background/background-0.pkl
