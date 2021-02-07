import numpy as np
import os,sys,time,datetime
from os.path import expanduser
import argparse


import subprocess
git_rev = subprocess.Popen(
    "git rev-parse --short HEAD",
    shell=True,
    stdout=subprocess.PIPE,
    universal_newlines=True
).communicate()[0]
git_branch = subprocess.Popen(
    "git symbolic-ref --short -q HEAD",
    shell=True,
    stdout=subprocess.PIPE,
    universal_newlines=True
).communicate()[0]

timestamp = time.time()
timestamp = datetime.datetime.fromtimestamp(
    int(timestamp)).strftime('%Y-%m-%d %H-%M-%S'
    )

# Set random seed
#seed = 123
#np.random.seed(seed)
#tf.set_random_seed(seed)

parser = argparse.ArgumentParser(
    description="argument for GraphSAINT training"
)
parser.add_argument(
    "--num_cpu_core",
    default=20,
    type=int,
    help="Number of CPU cores for parallel sampling"
)
parser.add_argument(
    "--log_device_placement",
    default=False,
    action="store_true",
    help="Whether to log device placement"
)
parser.add_argument(
    "--data_prefix",
    required=True,
    type=str,
    help="prefix identifying training data"
)
parser.add_argument(
    "--dir_log",
    default=".",
    type=str,
    help="base directory for logging and saving embeddings"
)
parser.add_argument(
    "--gpu",
    default="-1234",
    type=str,
    help="which GPU to use"
)
parser.add_argument(
    "--eval_train_every",
    default=15,
    type=int,
    help="How often to evaluate training subgraph accuracy"
)
parser.add_argument(
    "--eval_val_every",
    default=1,
    type=int,
    help="How often to evaluate training subgraph accuracy on val data"
)
parser.add_argument(
    "--train_config",
    required=True,
    type=str,
    help="path to the configuration of training (*.yml)"
)
parser.add_argument(
    "--dtype",
    default="s",
    type=str,
    help="d for double, s for single precision floating point"
)
parser.add_argument(
    "--timeline",
    default=False,
    action="store_true",
    help="to save timeline.json or not"
)
parser.add_argument(
    "--tensorboard",
    default=False,
    action="store_true",
    help="to save data to tensorboard or not"
)
parser.add_argument(
    "--dualGPU",
    default=False,
    action="store_true",
    help="whether to distribute the model to two GPUs"
)
parser.add_argument(
    "--cpu_eval",
    default=False,
    action="store_true",
    help="whether to use CPU to do evaluation"
)
#parser.add_argument(
#    "--saved_model_path",
#    default="",
#    type=str,
#    help="path to pretrained model file"
#)
parser.add_argument(
    "--loss_dim_expand",
    default=False,
    action="store_true",
    help="whether to expand loss dim to accomodate ogbn data"
)
parser.add_argument(
    "--num_global_train_steps",
    default=1,
    type=int,
    help="How many times to perform the training operation i.e. num calls to sess.run()"
)
parser.add_argument(
    "--train_log_freq",
    default=20,
    type=int,
    help="How often perform the log ops spec'd in the estimator RunConfig, like train loss"
)

# Cerebras args
parser.add_argument('--cs_ip', help='CS-1 IP address, defaults to None')
parser.add_argument(
    '-m',
    '--mode',
    choices=['validate_only', 'compile_only', 'train', 'eval', 'eval_all'],
    default='train',
    help=(
        'Can choose from validate_only, compile_only, train, eval or eval_all.'
        + '  Validate only will only go up to kernel matching.'
        + '  Compile only continues through and generate compiled'
        + '  executables.'
        + '  Train will compile and train if on CS-1,'
        + '  and just train locally (CPU/GPU) if not on CS-1.'
        + '  Eval will run eval locally for the last checkpoint.'
        + '  Eval_all will run eval locally for all available checkpoints.'
    ),
)
#parser.add_argument(
#    '-p',
#    '--params',
#    default=DEFAULT_PARAMS_FILE,
#    help='Path to .yaml file with model parameters',
#)
parser.add_argument(
    '-o',
    '--model_dir',
    type=str,
    help='Save compilation and non-simfab outputs',
)
#parser.add_argument(
#    '--steps',
#    type=int,
#    help='Num of steps to run in total for mode=train or eval',
#)
#parser.add_argument(
#    '--device',
#    default=None,
#    help='Force model to run on a specific device (e.g., --device /gpu:0)',
#)

args_global = parser.parse_args()

NUM_PAR_SAMPLER = args_global.num_cpu_core
SAMPLES_PER_PROC = -(-200 // NUM_PAR_SAMPLER) # round up division

EVAL_VAL_EVERY_EP = args_global.eval_val_every #25 #1       # get accuracy on the validation set every this # epochs
NUM_GLOBAL_TRAIN_STEPS = args_global.num_global_train_steps
TRAIN_LOG_FREQ = args_global.train_log_freq

# auto choosing available NVIDIA GPU
gpu_selected = args_global.gpu
if gpu_selected == '-1234':
    # auto detect gpu by filtering on the nvidia-smi command output
    gpu_stat = subprocess.Popen("nvidia-smi",shell=True,stdout=subprocess.PIPE,universal_newlines=True).communicate()[0]
    gpu_avail = set([str(i) for i in range(8)])
    for line in gpu_stat.split('\n'):
        if 'python' in line:
            if line.split()[1] in gpu_avail:
                gpu_avail.remove(line.split()[1])
            if len(gpu_avail) == 0:
                gpu_selected = -2
            else:
                gpu_selected = sorted(list(gpu_avail))[0]
    if gpu_selected == -1:
        gpu_selected = '0'
    args_global.gpu = int(gpu_selected)
if str(gpu_selected).startswith('nvlink'):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selected).split('nvlink')[1]
elif int(gpu_selected) >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selected)
    GPU_MEM_FRACTION = 0.8
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
args_global.gpu = int(args_global.gpu)

# global vars

f_mean = lambda l: sum(l)/len(l)

DTYPE = "float32" if args_global.dtype=='s' else "float64"      # NOTE: currently not supporting float64 yet
