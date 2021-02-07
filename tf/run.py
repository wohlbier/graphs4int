# Collation of the blocks from estimator_working_v1.ipynb into a main script

# # cmds from "GraphSAINT" dir
# NOTE the train_log_freq, and the num_global_steps are calc'd by hand
# relative to the number of epochs specified in the config yml file
# and currently aren't very precise, just something to make sure it works
# e.g. global train steps is sort of just a max here, yml epochs is shorter effectively

# python3 -m train_estimator --data_prefix ./data/ogbn-products \
#                            --train_config train_config/open_graph_benchmark/ogbn-products_3_e_gat.yml \
#                            --gpu 0 \
#                            --cpu_eval \
#                            --eval_val_every 20 \
#                            --loss_dim_expand \
#                            --train_log_freq 100 \
#                            --num_global_train_steps 1200

# python3 -m train_estimator --data_prefix data/ppi \
#                            --train_config train_config/table2/ppi2_e.yml \
#                            --gpu 0 \
#                            --cpu_eval \
#                            --eval_val_every 20 \
#                            --train_log_freq 40 \
#                            --num_global_train_steps 1200

# current checkpoint clearing command from GraphSAINT `rm -rf models/* && rm -rf test-saving-* && rm -rf tmp.chkpt*`

import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from graphsaint.globals import *
from graphsaint.tf.inits import *
from graphsaint.tf.model import GraphSAINT
from graphsaint.tf.minibatch import Minibatch
from graphsaint.utils import *
from graphsaint.metric import *
from tensorflow.python.client import timeline

#from common_zoo.estimator.tf.cs_estimator import CerebrasEstimator
#from common_zoo.estimator.tf.run_config import CSRunConfig
#from common_zoo.run_utils.utils import (
#    save_dict,
#    prep_env,
#)

# really major todo around val/test evaluation - less efficient than reference
# right now
def evaluate_full_batch(sess, model, minibatch_iter, many_runs_timeline, mode):
    """
    Full batch evaluation
    NOTE: HERE GCN RUNS THROUGH THE FULL GRAPH. HOWEVER, WE CALCULATE F1 SCORE
        FOR VALIDATION / TEST NODES ONLY.
    """
    options = tf.compat.v1.RunOptions(
        trace_level=tf.compat.v1.RunOptions.FULL_TRACE
    )
    run_metadata = tf.compat.v1.RunMetadata()
    t1 = time.time()
    num_cls = minibatch_iter.class_arr.shape[-1]
    feed_dict, labels = minibatch_iter.feed_dict(mode)
    if args_global.timeline:
        preds,loss = sess.run(
            [model.preds, model.loss], feed_dict=feed_dict,
            options=options, run_metadata=run_metadata
        )
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        many_runs_timeline.append(chrome_trace)
    else:
        preds,loss = sess.run(
            [model.preds, model.loss], feed_dict=feed_dict
        )
    node_val_test = minibatch_iter.node_val \
        if mode=='val' else minibatch_iter.node_test
    t2 = time.time()
    f1_scores = calc_f1(
        labels[node_val_test],preds[node_val_test],model.sigmoid_loss
    )
    return loss, f1_scores[0], f1_scores[1], (t2-t1)

# kind of the root of all evil
def construct_placeholders(num_classes):
    placeholders = {
        'node_subgraph':
        tf.compat.v1.placeholder(
            tf.int32, shape=(None), name='node_subgraph'
        ),
        'labels':
        tf.compat.v1.placeholder(
            DTYPE, shape=(None, num_classes), name='labels'
        ),
        'dropout':
        tf.compat.v1.placeholder(
            DTYPE, shape=(None), name='dropout'
        ),
        'norm_loss':
        tf.compat.v1.placeholder(
            DTYPE, shape=(None), name='norm_loss'
        ),
        'adj_subgraph':
        tf.compat.v1.sparse_placeholder(
            DTYPE, name='adj_subgraph', shape=(None,None)
        ),
        'adj_subgraph_0':
        tf.compat.v1.sparse_placeholder(
            DTYPE, name='adj_subgraph_0'
        ),
        'adj_subgraph_1':
        tf.compat.v1.sparse_placeholder(
            DTYPE, name='adj_subgraph_1'
        ),
        'adj_subgraph_2':
        tf.compat.v1.sparse_placeholder(
            DTYPE,name='adj_subgraph_2'
        ),
        'adj_subgraph_3':
        tf.compat.v1.sparse_placeholder(
            DTYPE, name='adj_subgraph_3'
        ),
        'adj_subgraph_4':
        tf.compat.v1.sparse_placeholder(
            DTYPE, name='adj_subgraph_4'
        ),
        'adj_subgraph_5':
        tf.compat.v1.sparse_placeholder(
            DTYPE, name='adj_subgraph_5'
        ),
        'adj_subgraph_6':
        tf.compat.v1.sparse_placeholder(
            DTYPE, name='adj_subgraph_6'
        ),
        'adj_subgraph_7':
        tf.compat.v1.sparse_placeholder(
            DTYPE, name='adj_subgraph_7'
        ),
        'dim0_adj_sub':
        tf.compat.v1.placeholder(
            tf.int64, shape=(None), name='dim0_adj_sub'
        ),
        'is_train':
        tf.compat.v1.placeholder(
            tf.bool, shape=(None), name='is_train'
        )
    }
    return placeholders

# These are very much a bandaid
placeholders = -1
minibatch = -1
model = -1
epoch = -1

# this class name is clearly from reference code but unrelated to
# functionality now
class FeederHook(tf.estimator.SessionRunHook):
    def __init__(self):
        super(FeederHook, self).__init__()
        self.iterator_initiliser_func = None
        self.feed_update_func = None

    def before_run(self, sess_run_context):
        return self.feed_update_func(sess_run_context.original_args)

def train_input_fn(*args, **kwargs):

    phase=kwargs["phase"]

    feeder_hook = FeederHook()

    def epoch_generator(minibatch):
        for e in range(int(phase['end'])):
            global epoch
            epoch = e
            printf('Epoch {:4d}'.format(e),style='bold')
            minibatch.shuffle()
            while not minibatch.end():
                feed_dict, labels = minibatch.feed_dict(mode='train')
                yield feed_dict, labels

    def input_fn():

        train_data,train_params,arch_gcn = args[0], args[1], args[2]
        adj_full,adj_train,feats,class_arr,role = train_data
        adj_full = adj_full.astype(np.int32)
        adj_train = adj_train.astype(np.int32)
        adj_full_norm = adj_norm(adj_full)
        num_classes = class_arr.shape[1]

        global minibatch
        global placeholders

        placeholders = construct_placeholders(num_classes)
        minibatch = Minibatch(
            adj_full_norm, adj_train, role, class_arr, placeholders,
            train_params
        )

        print("set sampler/phase, get epoch")
        minibatch.set_sampler(phase)
        num_batches = minibatch.num_training_batches()
        print("num batches after sampling: ", num_batches)

        gen = epoch_generator(minibatch)

        feeder_hook.feed_update_func = \
            lambda original_args: tf.estimator.SessionRunArgs(
                original_args.fetches, feed_dict= next(gen)[0],
                options=original_args.options
            )

        features, labels = None, None

        return features, labels

    return input_fn, placeholders, feeder_hook

def custom_model_fn(*model_args, **kwargs):

    train_data, train_params, arch_gcn = \
        model_args[0], model_args[1], model_args[2]
    adj_full, adj_train, feats, class_arr, role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]

    def model_fn(features,labels, mode):

        global placeholders
        global model

        model = GraphSAINT(
            num_classes, placeholders, feats, arch_gcn, train_params,
            adj_full_norm, logging=True
        )

        loss = model.loss

        train_op = model.train_op #includes model.opt_op and global step update

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op
        )
    return model_fn

# this is a first attempt at a callback to do validation, but this feels like
# a big bottleneck as it stands
class SessionSaverHook(tf.estimator.SessionRunHook):
    def __init__(self):
        super(SessionSaverHook, self).__init__()
        self.session_save_func = None
        self.saver = None
        self.session = None
        self.sess_cpu = None
        self.eval_saver = None
        self.last_eval = -1

    def begin(self):
        self.saver=tf.compat.v1.train.Saver()
        with tf.device('/cpu:0'):
            self.sess_cpu = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(device_count={'GPU': 0})
            )
            self.sess_cpu.run(tf.compat.v1.global_variables_initializer())
            self.eval_saver = tf.compat.v1.train.Saver()

    def after_run(self, sess_run_context, sess_run_values):
        global epoch
        if (epoch+1)%EVAL_VAL_EVERY_EP == 0 and self.last_eval < epoch:
            self.last_eval = epoch
            #global_step = tf.train.get_global_step()
            self.session = sess_run_context.session
            #self.saver.save(
            #    self.session, 'test-saving', global_step=global_step
            #)

            self.saver.save(self.session,'./tmp.chkpt')
            self.eval_saver.restore(self.sess_cpu, './tmp.chkpt')

            global minibatch
            global model
            many_runs_timeline = None

            loss_val, f1mic_val, f1mac_val, time_eval = \
                evaluate_full_batch(
                    self.sess_cpu, model, minibatch, many_runs_timeline,
                    mode='val'
                )
            printf(
                ' VALIDATION:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}'.format(loss_val,f1mic_val,f1mac_val),style='blue'
            )

def main(argv=None):
    #args = parse_args()
    #params = get_params(args.params, args)

    train_params, train_phases, train_data, arch_gcn = parse_n_prepare(
        args_global
    )
    print("train_params: " + str(train_params))
    print("train_phases: " + str(train_phases))
    print("arch_gcn: " + str(arch_gcn))
    model_args = [train_data,train_params,arch_gcn]

    params = vars(args_global)
    print("params: " + str(params))
    #cb#save_dict(params, model_dir=params['model_dir'])

    use_cs = (
        params['mode'] in ('train', 'compile_only')
        and params['cs_ip'] is not None
    )
    cs_ip = params['cs_ip'] + ':9000' if use_cs else None
    #cb#prep_env(params)

    # Cerebras stack params
    stack_params = dict()
    #if use_cs:
    #    from cerebras.pb.stack.full_pb2 import FullConfig
    #    from cerebras.pb.common.tri_state_pb2 import TS_DISABLED
    #
    #    # Custom Cerebras config for improved performance
    #    config = FullConfig()
    #    config.placement.match_port_format.prefer_dense_packets = TS_DISABLED
    #    stack_params['config'] = config

    #if not (use_cs or params['optimizer']['grad_accum_steps'] < 2):
    #    params['runconfig']['save_summary_steps'] = 1

    #config = CSRunConfig(
    #    cs_ip=cs_ip, stack_params=stack_params, **params['runconfig'],
    #)
    config = tf.estimator.RunConfig(
        model_dir=params["model_dir"],
        save_summary_steps=TRAIN_LOG_FREQ,
        save_checkpoints_steps=TRAIN_LOG_FREQ,
        log_step_count_steps=TRAIN_LOG_FREQ
    )

    #est = CerebrasEstimator(
    #    model_fn,
    #    model_dir=params['model_dir'],
    #    params=params,
    #    use_cs=use_cs,
    #    config=config,
    #)
    input_fn, placeholders, iterator_hook = train_input_fn(
        *model_args, phase=train_phases[0]
    )
    model_fn = custom_model_fn(
        *model_args, placeholders=placeholders
    )
    est = tf.compat.v1.estimator.Estimator(
        model_fn=model_fn, config=config
    )
    session_saver_hook = SessionSaverHook()

    if params['mode'] == 'train':
        est.train(
            input_fn=input_fn,
            steps=NUM_GLOBAL_TRAIN_STEPS,
            hooks=[iterator_hook, session_saver_hook]
        )
    #elif params['mode'] == 'eval':
    #    est.evaluate(
    #        input_fn=lambda: input_fn(params, tf.estimator.ModeKeys.EVAL),
    #    )
    #elif params['mode'] == 'eval_all':
    #    ckpt_list = tf.train.get_checkpoint_state(
    #        params['model_dir']
    #    ).all_model_checkpoint_paths
    #    for ckpt in ckpt_list:
    #        est.evaluate(
    #            input_fn=lambda: input_fn(params, tf.estimator.ModeKeys.EVAL),
    #            checkpoint_path=ckpt
    #        )
    #else:
    #    est.compile(
    #        input_fn,
    #        validate_only=(params['mode'] == 'validate_only'),
    #    )

if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()
