# CS5478-project

## STEP1 install AERIAL GYM

Follow the guide at [Aerial Gym: Getting Started](https://ntnu-arl.github.io/aerial_gym_simulator/2_getting_started/)

## STEP2 download TINY-PULP-DRONET-v3 dataset

This is for training the VAE model: https://zenodo.org/records/13348430 

To run the VAE train-test loop, place the dataset in /aerial_gym_simulator/aerial_gym/utils/vae/data
Run:
```
python ./aerial_gym_simulator/aerial_gym/utils/vae/vae_image_training.py
```

## STEP3 Run simulation

This is for running the simulator for the D+P VAE version of policy
Run:
```
python3 runner.py --file=./ppo_aerial_quad_navigation.yaml --num_envs=16 --headless=False --task=navigation_task --checkpoint=./checkpoints/gen_ppo_03-01-06-11_dronetV3_navObs/nn/gen_ppo.pth --play
```