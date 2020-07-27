# generalization
## Running Experiments on OPTSPACE
Optspace server has 8 RTX 2080 Ti NVIDIA GPUs.
Recommended manner of running experiments is with tmux

### Run Experiment
```
$ CUDA_VISIBLE_DEVICES=<device_num> python main.py --gpu_idx 0 --model_num <model_num> ...
```
Make sure to specify optional hyperparameters as desired.