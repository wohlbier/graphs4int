import argparse
import json
import numpy as np
import os
import scipy as sp
import tensorflow as tf
import yaml
from sklearn.preprocessing import StandardScaler

def parse_args():
    '''
    Parse command line args.
    '''
    parser = argparse.ArgumentParser(description='GraphSAINT')
    parser.add_argument('--cs_ip', help='CS-1 IP address, defaults to None')
    parser.add_argument(
        '-m',
        '--mode',
        choices=['validate_only', 'compile_only', 'train', 'eval', 'eval_all'],
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
    parser.add_argument(
        '-p',
        '--params',
        default='./graphsaint/jgw_tf/params.yaml',
        help='Path to .yaml file with model parameters',
    )
    parser.add_argument(
        '-o',
        '--model_dir',
        type=str,
        help='Save compilation and non-simfab outputs',
    )
    parser.add_argument(
        '--steps',
        type=int,
        help='Num of steps to run in total for mode=train or eval',
    )
    parser.add_argument(
        '--device',
        default=None,
        help='Force model to run on a specific device (e.g., --device /gpu:0)',
    )
    parser.add_argument("--data-prefix", default='./data/ogbn-products', type=str,
                        help="prefix identifying training data")
    return parser.parse_args()

def get_params(params_file, args):
    # Load yaml into params
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)
    set_defaults(params)
    set_cmdline_args(params, args)
    # store params_file
    params["params_file"] = params_file

    return params

def set_defaults(params):
    train_input_params = params["train_input"]
    training_params = params["training"]
    runconfig_params = params["runconfig"]
    model_params = params["model"]
    optim_params = params["optimizer"]
    # params for OG GraphSAINT
    archgcn_params = params["arch_gcn"]
    network_params = params["network"]
    train_params = params["train_params"]
    phase_params = params["phase"]

    # Input dataset parameters
    train_input_params["mixed_precision"] = training_params.get(
        "mixed_precision", True
    )

    # Model parameters
    model_params["use_bias"] = model_params.get("use_bias", False)
    model_params["pre_activation"] = model_params.get("pre_activation", False)
    model_params["dropout_rate"] = model_params.get("dropout_rate", 0.0)
    model_params["use_norm"] = model_params.get("use_norm", False)
    model_params["layer_norm_epsilon"] = float(
        model_params.get("layer_norm_epsilon", 1e-8)
    )

    # Estimator parameters
    runconfig_params["save_summary_steps"] = runconfig_params.get(
        "save_summary_steps", 1000
    )
    runconfig_params["save_checkpoints_steps"] = runconfig_params.get(
        "save_checkpoints_steps", 1000
    )
    runconfig_params["keep_checkpoint_max"] = runconfig_params.get(
        "keep_checkpoint_max", 5
    )
    runconfig_params["tf_random_seed"] = runconfig_params.get(
        "tf_random_seed", None
    )
    runconfig_params["log_step_count_steps"] = runconfig_params.get(
        "log_step_count_steps", 1000
    )
    training_params["enable_distributed"] = training_params.get(
        "enable_distributed", False
    )
    training_params["multiple_workers"] = training_params.get(
        "multiple_workers", False
    )
    training_params["max_steps"] = training_params.get("max_steps", 100)

    # optimizer parameters
    optim_params["loss_scaling_factor"] = optim_params.get(
        "loss_scaling_factor", 1.0
    )
    optim_params["max_gradient_norm"] = optim_params.get(
        "max_gradient_norm", None
    )
    optim_params["grad_accum_steps"] = optim_params.get("grad_accum_steps", 1)
    optim_params["log_summaries"] = optim_params.get("log_summaries", False)

    # from OG GraphSAINT, converted to params
    archgcn_params["dim"] = network_params["dim"]
    assert phase_params["end"] is not None
    assert phase_params["sampler"] is not None

def set_cmdline_args(params, args):
    params["model_dir"] = (
        args.model_dir if args.model_dir is not None else params["model_dir"]
    )
    params["mode"] = args.mode if args.mode is not None else params["mode"]
    params["training"]["steps"] = (
        args.steps if args.steps is not None else params["training"]["steps"]
    )
    params["cs_ip"] = args.cs_ip
    params["data_prefix"] = args.data_prefix

#_curdir = os.path.dirname(os.path.abspath(__file__))
#DEFAULT_PARAMS_FILE = os.path.join(_curdir, 'configs/params_gcn.yaml')

def load_data(prefix, normalize=True):
    """
    Load the various data files residing in the `prefix` directory.
    Files to be loaded:
    adj_full.npz
       sparse matrix in CSR format, stored as scipy.sparse.csr_matrix
       The shape is N by N. Non-zeros in the matrix correspond to all
       the edges in the full graph. It doesn't matter if the two nodes
       connected by an edge are training, validation or test nodes.
       For unweighted graph, the non-zeros are all 1.
    adj_train.npz
       sparse matrix in CSR format, stored as a scipy.sparse.csr_matrix
       The shape is also N by N. However, non-zeros in the matrix only
       correspond to edges connecting two training nodes. The graph
       sampler only picks nodes/edges from this adj_train, not adj_full.
       Therefore, neither the attribute information nor the structural
       information are revealed during training. Also, note that only
       a x N rows and cols of adj_train contains non-zeros. For
       unweighted graph, the non-zeros are all 1.
    role.json
       a dict of three keys. Key 'tr' corresponds to the list of all
       'tr':     list of all training node indices
       'va':     list of all validation node indices
       'te':     list of all test node indices
       Note that in the raw data, nodes may have string-type ID. You
       need to re-assign numerical ID (0 to N-1) to the nodes, so that
       you can index into the matrices of adj, features and class labels.
    class_map.json
       a dict of length N. Each key is a node index, and each value is
       either a length C binary list (for multi-class classification)
       or an integer scalar (0 to C-1, for single-class classification).
    feats.npz
       a numpy array of shape N by F. Row i corresponds to the attribute
       vector of node i.

    Inputs:
    prefix
       string, directory containing the above graph related files
    normalize
       bool, whether or not to normalize the node features

    Outputs:
    adj_full
       scipy sparse CSR (shape N x N, |E| non-zeros), the adj matrix of
       the full graph, with N being total num of train + val + test nodes.
    adj_train
       scipy sparse CSR (shape N x N, |E'| non-zeros), the adj matrix of
       the training graph. While the shape is the same as adj_full, the
       rows/cols corresponding to val/test nodes in adj_train are all-zero.
    feats
       np array (shape N x f), the node feature matrix, with f being the
       length of each node feature vector.
    class_map
       dict, where key is the node ID and value is the classes this node
       belongs to.
    role
       dict, where keys are: 'tr' for train, 'va' for validation and 'te'
       for test nodes. The value is the list of IDs of nodes belonging to
       the train/val/test sets.
    """
    adj_full = sp.sparse.load_npz('{}/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = sp.sparse.load_npz('{}/adj_train.npz'.format(prefix)).astype(np.bool)
    role = json.load(open('{}/role.json'.format(prefix)))
    feats = np.load('{}/feats.npy'.format(prefix))
    class_map = json.load(open('{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    return adj_full, adj_train, feats, class_map, role

def process_graph_data(adj_full, adj_train, feats, class_map, role):
    """
    setup vertex property map for output classes, train/val/test masks, and feats
    """
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1
    return adj_full, adj_train, feats, class_arr, role

#def parse_layer_yml(arch_gcn,dim_input):
#    """
#    Parse the *.yml config file to retrieve the GNN structure.
#    """
#    num_layers = len(arch_gcn['arch'].split('-'))
#    # set default values, then update by arch_gcn
#    bias_layer = [arch_gcn['bias']]*num_layers
#    act_layer = [arch_gcn['act']]*num_layers
#    aggr_layer = [arch_gcn['aggr']]*num_layers
#    dims_layer = [arch_gcn['dim']]*num_layers
#    order_layer = [int(o) for o in arch_gcn['arch'].split('-')]
#    return [dim_input]+dims_layer,order_layer,act_layer,bias_layer,aggr_layer

#def log_dir(f_train_config,prefix,git_branch,git_rev,timestamp):
#    import getpass
#    log_dir = args_global.dir_log+"/log_train/" + prefix.split("/")[-1]
#    log_dir += "/{ts}-{model}-{gitrev:s}/".format(
#            model='graphsaint',
#            gitrev=git_rev.strip(),
#            ts=timestamp)
#    if not os.path.exists(log_dir):
#        os.makedirs(log_dir)
#    if f_train_config != '':
#        from shutil import copyfile
#        copyfile(f_train_config,'{}/{}'.format(log_dir,f_train_config.split('/')[-1]))
#    return log_dir
#
#def sess_dir(dims,train_config,prefix,git_branch,git_rev,timestamp):
#    import getpass
#    log_dir = "saved_models/" + prefix.split("/")[-1]
#    log_dir += "/{ts}-{model}-{gitrev:s}-{layer}/".format(
#            model='graphsaint',
#            gitrev=git_rev.strip(),
#            layer='-'.join(dims),
#            ts=timestamp)
#    if not os.path.exists(log_dir):
#        os.makedirs(log_dir)
#    return sess_dir

#def adj_norm(adj, deg=None, sort_indices=True):
#    """
#    Normalize adj according to the method of rw normalization.
#    Note that sym norm is used in the original GCN paper (kipf),
#    while rw norm is used in GraphSAGE and some other variants.
#    Here we don't perform sym norm since it doesn't seem to
#    help with accuracy improvement.
#
#    # Procedure:
#    #       1. adj add self-connection --> adj'
#    #       2. D' deg matrix from adj'
#    #       3. norm by D^{-1} x adj'
#    if sort_indices is True, we re-sort the indices of the returned adj
#    Note that after 'dot' the indices of a node would be in descending order
#    rather than ascending order
#    """
#    diag_shape = (adj.shape[0],adj.shape[1])
#    D = adj.sum(1).flatten() if deg is None else deg
#    norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)
#    adj_norm = norm_diag.dot(adj)
#    if sort_indices:
#        adj_norm.sort_indices()
#    return adj_norm
#
#
#
#
###################
## PRINTING UTILS #
##----------------#
#
#_bcolors = {'header': '\033[95m',
#            'blue': '\033[94m',
#            'green': '\033[92m',
#            'yellow': '\033[93m',
#            'red': '\033[91m',
#            'bold': '\033[1m',
#            'underline': '\033[4m'}
#
#
#def printf(msg,style=''):
#    if not style or style == 'black':
#        print(msg)
#    else:
#        print("{color1}{msg}{color2}".format(color1=_bcolors[style],msg=msg,color2='\033[0m'))

def input_fn(params, mode=tf.estimator.ModeKeys.TRAIN, input_context=None):
    #return getattr(
    #    sys.modules[__name__], params["train_input"]["data_processor"]
    #)(params["train_input"]).create_tf_dataset(
    #    mode=mode, input_context=input_context,
    #)

    # relocated from parse_n_prepare
    print("Loading training data..")
    temp_data = load_data(params["data_prefix"])
    train_data = process_graph_data(*temp_data)
    print("Done loading training data..")

    features = { "adj_full": train_data[0], "adj_train": train_data[1],
                 "feats": train_data[2], "role": train_data[4] }
    labels = { "class_arr": train_data[3] }
    return features, labels
