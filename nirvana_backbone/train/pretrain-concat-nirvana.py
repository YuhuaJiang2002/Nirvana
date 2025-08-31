import gc
import argparse
import os
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

from accelerate import Accelerator
from mmengine import mkdir_or_exist
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env

import torch
import torch.distributed as dist
from torch.nn import functional as F
import torch.distributed.checkpoint as dcp
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     get_state_dict, set_state_dict)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.distributed_c10d import ReduceOp

import itertools
from concurrent.futures import wait

from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.import_utils import is_flash_attn_2_available

from xtuner._lite import (AutoTokenizer, get_device, get_logger,
                          get_torch_device_module)
from xtuner._lite.accelerate import (LORA_TARGET_MAP, dispatch_hf_code, LoadWoInit,
                                     packed_sequence, varlen_attn_is_available, profile_time_and_memory)
from xtuner._lite.datasets import (DATASET_CLS_MAP, OPENAI_CONVERT_MAP,
                                   SoftPackDataset, HardPackDataset, load_datasets)
from xtuner._lite.parallel import (ParallelSampler, get_dp_mesh, get_fsdp_mesh,
                                   get_sp_mesh, get_tp_mesh, get_world_mesh, get_same_data_mesh,
                                   pad_for_sequence_parallel, setup_parallel,
                                   reduce_sequence_parallel_loss,
                                   split_for_sequence_parallel)
from xtuner._lite.parallel.megatron import megatron_parallelize
from xtuner._lite.parallel.fsdp import clip_grad_norm_

from internlm.utils.common import assert_current_device_empty
from internlm.utils.execution_time import execution_time_collecter as etc
from torch.utils.tensorboard import SummaryWriter
import wandb
import threading
import queue

from wsd_scheduler import get_wsd_schedule

assert_current_device_empty()
with etc.collect_execute_time("import_time"):
    from internlm.core.context import ParallelMode
    from internlm.core.context import global_context as gpc
    from internlm.data.build_dataloader import (
        build_train_loader_with_data_type,
    )
    from internlm.data.utils import get_lang_subset_types
    from internlm.train.pipeline import load_new_batch_with_train_state
    from internlm.data.train_state import get_train_state
    from internlm.initialize import initialize_distributed_env
    from internlm.utils.common import (
        BatchSkipper,
        catch_error_node,
        enable_pytorch_expandable_segments,
        get_current_device,
        get_gpu_id,
        get_megatron_flops,
        launch_time,
        switch_topology_aware_rank_scheduling,
    )

logger = get_logger()

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()

SUPPORT_DATA_FORMATS = OPENAI_CONVERT_MAP.keys()

try:
    from c4d_perftracker_collector.PerfTracker import PerfTracker
    my_tracer = PerfTracker()
except:
    my_tracer = None

def record_wandb(wandb_kargs, queue: queue.Queue):
    # Initialize the WandB run
    wandb.init(**wandb_kargs)
    i = 0
    while True:
        if not queue.empty():
            tag, value, step = queue.get()
            wandb.log({tag: value}, step=step)
            i += 1
        else:
            time.sleep(0.01)


class WandbWrapper:
    def __init__(
            self,
            project_name="internlm2-1_8b",
            entity=None,
            name="",
            queue_size=1000,
            dataset_types=[],
    ):
        wandb_kargs = dict(
            project=project_name,
            entity=entity,
            name=name,  # You can use name as run name
        )
        if dist.get_rank() == 0:
            self.queue = queue.Queue(maxsize=queue_size)
            self.thread = threading.Thread(
                target=record_wandb, args=(wandb_kargs, self.queue)
            )
            self.thread.start()
        else:
            self.queue = None
            self.thread = None
        self.dataset_types = dataset_types + \
            ["undefined"]  # TODO: Check this mapping

    def add_scalar(
            self,
            tag,
            scalar_value,
            global_step=None,
            new_style=False,
            double_precision=False,
            reduce_op=None,
    ):
        if reduce_op is not None:
            scalar_value = torch.tensor(scalar_value).cuda()
            dist.all_reduce(scalar_value, op=reduce_op)
            scalar_value = scalar_value.item()

        # Put the data in the queue to be logged by the thread
        if dist.get_rank() == 0:
            self.queue.put((tag, scalar_value, global_step))

    def add_train_dynamics(self, loss, unreduced_losses, type_ids, steps):
        self.add_scalar("train_loss/total_loss", loss,
                        global_step=steps, reduce_op=ReduceOp.AVG)
        # loss per class type
        return 
        unreduced_losses = torch.cat(
            unreduced_losses, dim=0).flatten()  # (B T-1)
        type_ids = type_ids.to(unreduced_losses.device)  # B T
        type_ids = type_ids[:, :-1].flatten()  # (B T-1)
        type_ids[type_ids == -1] = len(self.dataset_types) - 1
        loss_scatter = torch.zeros(
            [len(self.dataset_types)],
            device=unreduced_losses.device,
            dtype=unreduced_losses.dtype,
        )
        count = torch.bincount(type_ids, minlength=len(self.dataset_types))
        loss_scatter.scatter_add_(0, type_ids, unreduced_losses)
        loss_scatter = loss_scatter / (count + 1e-6)
        loss_scatter = loss_scatter.tolist()
        for i, loss in enumerate(loss_scatter):
            self.add_scalar(
                f"train_loss/{self.dataset_types[i]}", loss, global_step=steps, reduce_op=ReduceOp.AVG
            )

    def add_optimize_info(self, grad_norm, train_state, cur_lr, steps):
        self.add_scalar("optimize/grad_norm", grad_norm,
                        global_step=steps, reduce_op=ReduceOp.AVG)
        self.add_scalar(
            "optimize/inf_nan_skip_batches",
            train_state.inf_nan_skip_batches,
            global_step=steps,
            reduce_op=ReduceOp.AVG
        )

        self.add_scalar("optimize/learning_rate", cur_lr,
                        global_step=steps, reduce_op=ReduceOp.AVG)

    def add_data_infos(self, type_ids, train_state, step):
        # tokens for classes
        type_ids[type_ids == -1] = len(self.dataset_types) - 1
        count = torch.bincount(
            type_ids.flatten(), minlength=len(self.dataset_types))
        count = dict(
            (f"{self.dataset_types[i]}", v) for i, v in enumerate(count.tolist())
        )
        for k, v in count.items():
            self.add_scalar("data_tokens/" + k, v, step,
                            reduce_op=ReduceOp.AVG)

        # epochs for subsets
        used_epochs = train_state.data_state_dict["used_epochs"]
        for file_name, e in used_epochs.items():
            self.add_scalar(
                f"data_subset_epochs/{file_name}", e, step, ReduceOp.SUM
            )  # only in rank 0

    def add_speed_info(self, tgs, e2e_tgs, step):
        self.add_scalar("speed/tgs", tgs, step, reduce_op=ReduceOp.AVG)
        self.add_scalar("speed/e2e_tgs", e2e_tgs, step, reduce_op=ReduceOp.AVG)

    def finish(self):
        wandb.join()


def record_tensorboard(tensorboard_kargs, queue: queue.Queue):
    writer = SummaryWriter(**tensorboard_kargs)
    i = 0
    while True:
        if not queue.empty():
            tag, value, step = queue.get()
            writer.add_scalar(tag, value, step)
            i += 1
            # if i % 1000 == 0:
            #     print(f"qsize {queue.qsize()}")
        else:
            time.sleep(0.01)


class SummaryWriterWrapper(SummaryWriter):
    def __init__(
            self,
            log_dir=None,
            comment="",
            purge_step=None,
            max_queue=10,
            flush_secs=120,
            filename_suffix="",
            dataset_types=[],
            queue_size=1000,
    ):
        tensorboard_kargs = dict(
            log_dir=log_dir,
            comment=comment,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(
            target=record_tensorboard, args=(tensorboard_kargs, self.queue)
        )
        self.thread.start()
        self.dataset_types = dataset_types + \
            ["undefined"]  # TODO: Check this mapping

    def add_scalar(
            self,
            tag,
            scalar_value,
            global_step=None,
            walltime=None,
            new_style=False,
            double_precision=False,
            reduce_op=None,
    ):
        if reduce_op is not None:
            scalar_value = torch.tensor(scalar_value).cuda()
            dist.all_reduce(scalar_value, op=reduce_op)
            scalar_value = scalar_value.item()
        self.queue.put((tag, scalar_value, global_step))

    def add_train_dynamics(self, loss, unreduced_losses, type_ids, steps):
        self.add_scalar("train_loss/total_loss", loss,
                        global_step=steps, reduce_op=ReduceOp.AVG)
        # loss per class type

        return
        unreduced_losses = torch.cat(
            unreduced_losses, dim=0).flatten()  # (B T-1)
        type_ids = type_ids.to(unreduced_losses.device)  # B T
        type_ids = type_ids[:, :-1].flatten()  # (B T-1)
        type_ids[type_ids == -1] = len(self.dataset_types) - 1
        loss_scatter = torch.zeros(
            [len(self.dataset_types)],
            device=unreduced_losses.device,
            dtype=unreduced_losses.dtype,
        )
        count = torch.bincount(type_ids, minlength=len(self.dataset_types))
        loss_scatter.scatter_add_(0, type_ids, unreduced_losses)
        loss_scatter = loss_scatter / (count + 1e-6)
        loss_scatter = loss_scatter.tolist()
        for i, loss in enumerate(loss_scatter):
            self.add_scalar(
                f"train_loss/{self.dataset_types[i]}", loss, global_step=steps, reduce_op=ReduceOp.AVG
            )

        '''cancel the calculation of acc
        # acc per class type
        correct_preds = (
            torch.cat(correct_preds, dim=-1).flatten().to(unreduced_loss.dtype)
        )
        right_number = torch.zeros(
            [len(self.dataset_types)],
            device=unreduced_loss.device,
            dtype=unreduced_loss.dtype,
        )
        right_number.scatter_add_(0, type_ids, correct_preds)
        acc = right_number / (count + 1e-6)
        acc = acc.tolist()
        for i, acc_per_type in enumerate(acc):
            self.add_scalar(
                f"train_acc/{self.dataset_types[i]}", acc_per_type, global_step=steps, reduce_op=ReduceOp.AVG
            )
        total_acc = right_number.sum() / (count.sum() + 1e-6)
        self.add_scalar(f"train_acc/total", total_acc, global_step=steps, reduce_op=ReduceOp.AVG)
        '''

    def add_optimize_info(self, grad_norm, train_state, cur_lr, steps):
        self.add_scalar("optimize/grad_norm", grad_norm,
                        global_step=steps, reduce_op=ReduceOp.AVG)
        self.add_scalar(
            "optimize/inf_nan_skip_batches",
            train_state.inf_nan_skip_batches,
            global_step=steps,
            reduce_op=ReduceOp.AVG
        )

        self.add_scalar("optimize/learning_rate", cur_lr,
                        global_step=steps, reduce_op=ReduceOp.AVG)

    def add_data_infos(self, type_ids, train_state, step):
        # tokens for classes
        type_ids[type_ids == -1] = len(self.dataset_types) - 1
        count = torch.bincount(
            type_ids.flatten(), minlength=len(self.dataset_types))
        count = dict(
            (f"{self.dataset_types[i]}", v) for i, v in enumerate(count.tolist())
        )
        for k, v in count.items():
            self.add_scalar("data_tokens/" + k, v, step,
                            reduce_op=ReduceOp.AVG)

        # epochs for subsets
        used_epochs = train_state.data_state_dict["used_epochs"]
        for file_name, e in used_epochs.items():
            self.add_scalar(
                f"data_subset_epochs/{file_name}", e, step, ReduceOp.SUM
            )  # only in rank 0

    def add_speed_info(self, tgs, e2e_tgs, step):
        self.add_scalar("speed/tgs", tgs, step, reduce_op=ReduceOp.AVG)
        self.add_scalar("speed/e2e_tgs", e2e_tgs, step, reduce_op=ReduceOp.AVG)


def log_format(rank, debug=False):
    sp_rank = get_sp_mesh().get_local_rank()
    dp_rank = get_dp_mesh().get_local_rank()
    tp_rank = get_tp_mesh().get_local_rank()
    fsdp_rank = get_fsdp_mesh().get_local_rank()

    formatter = f'[XTuner][RANK {rank}][DP {dp_rank}][SP {sp_rank}][TP {tp_rank}]'
    formatter += '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]'

    if debug:
        formatter += '[<cyan>{name}</cyan>:'
        formatter += '<cyan>{function}</cyan>:'
        formatter += '<cyan>{line}</cyan>]'

    formatter += ' <level>{message}</level>'
    return formatter


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')

    model_args = parser.add_argument_group('model', 'Model Related Settings')
    model_args.add_argument('--llm', default='/cpfs01/shared/public/liudawei/models/internlm2-1_8b-mycfg',
                            help='repo id or local path of the model')
    model_args.add_argument(
        '--train-cfg', default='../configs/internlm2-1_8b.py', help='interntrain config file')
    parser.add_argument(  # load hf weight
        '--load-pretrain', action='store_true', help='Set to load pretrain HF CKPT!')
    parser.add_argument(
        '--reshard-after-forward', action='store_true', help='')
    parser.add_argument(
        '--use-wsd', action='store_true', help='')
    parser.add_argument(
        '--use-hsdp', action='store_true', help='to use hsdp or not')
    model_args.add_argument(
        '--dtype',
        default='auto',
        choices=['fp16', 'bf16', 'auto'],
        help=("the dtype of the model forward. When set to 'auto', it will "
              'automatically determine whether bf16 is available, '
              'prioritizing the use of bf16.'))
    model_args.add_argument(
        '--selective-recompute',
        default=1.0,
        type=float,
        help=('the ratio of re-computation for transforemer layers. '
              'The maximum is 1; the larger the value, the less memory '
              'required for training. The default is 1, meaning all layers '
              'need to be re-computated.'))
    model_args.add_argument(
        '--shard-strategy',
        default='full',
        choices=['full', 'hybrid'],
        help=('The sharding strategy to be used for distributed training.'))
    model_args.add_argument('--cpu-offload', action='store_true', help=(''))
    model_args.add_argument('--sp-size', type=int, default=1, help='')
    model_args.add_argument(
        '--max-grad-norm', default=1.0, type=float, help='gradient clipping')
    model_args.add_argument(
        '--attn-implementation', default='flash_attention_2',
        choices=['eager', 'flash_attention_2', 'my_flash_attention_2', "sdpa"],
    )
    parser.add_argument(
        '--work-dir',
        default='work_dirs/internlm2-1_8b',
        help='the dir to save logs and checkpoints')
    parser.add_argument(
        '--checkpoint-interval',
        default=10000,
        type=float,
        help=('how many steps to save a checkpoint; it can be a floating '
              'point number less than 1, or an integer greater than or equal '
              "to 1. When it's a floating point, it will be multiplied by the "
              'total number of training steps.'))
    parser.add_argument(
        '--hf-interval',
        default=10000,
        type=float,
        help=('how many steps to save a hf model; it can be a floating '
              'point number less than 1, or an integer greater than or equal '
              "to 1. When it's a floating point, it will be multiplied by the "
              'total number of training steps.'))
    parser.add_argument(
        '--max-keep-ckpts',
        type=int,
        default=20,
        help='the maximum number of checkpoints to keep.')
    parser.add_argument(
        '--checkpoint-drop-optimizer',
        action='store_true',
        help=('only model parameters are saved when saving a checkpoint. '
              'This can significantly reduce the size of checkpoint files, '
              'but the saved checkpoints cannot be resumed.'))
    parser.add_argument(
        '--tb-interval', default=1, type=int, help='tensorboard interval')
    parser.add_argument(
        '--log-interval', default=1, type=int, help='log interval')
    parser.add_argument(
        '--gc-interval', default=200, type=int, help='gc interval monitoring detection')
    parser.add_argument(
        '--resume', action='store_true', help='resume from the last checkpoint')
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='specify checkpoint path to be resumed from.')
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed for the training')
    parser.add_argument(
        '--debug', action='store_true', help='Set logger level to `DEBUG`')
    parser.add_argument(
        '--port', type=int, default=8888, help='port')
    parser.add_argument(
        '--wandb-project-name', type=str)
    parser.add_argument(
        '--wandb-name', type=str)
    parser.add_argument(
        '--log-wandb', action='store_true')
    parser.add_argument(
        '--abnormal-detect', action='store_true')
    parser.add_argument(
        '--abnormal-steps', default=[], nargs='+', type=int, help='start from 1 instead of 0')
    parser.add_argument(
        '--min_sample_length', default=3, type=int, help='Drop samples less than min_sample_length')
    args = parser.parse_args()
    return args


def is_interval(step, total_steps, interval):
    return (step+1) % interval == 0 or (step+1) == total_steps


def map_meta_modules(model, meta_model):
    modules = {name: mod for name, mod in model.named_modules()}
    meta_module_map = {
        mod: modules[name]
        for name, mod in meta_model.named_modules()
    }
    return meta_module_map


def build_llm_model(args, config, world_size, dtype=torch.bfloat16):
    if args.load_pretrain:
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm,
            config=config,
            torch_dtype=dtype
        )
    else:
        new_llm_cfg = {
            "tie_word_embeddings": False,
            "vocab_size": args.vocab_size,
        }
        llm_cfg = AutoConfig.from_pretrained(
            args.llm, **new_llm_cfg)
        llm_cfg.use_cache = False
        llm_cfg.torch_dtype = dtype

        llm = AutoModelForCausalLM.from_config(
                config=llm_cfg)  # must open flash-attn

    # Ensure all numerical values in the optimizer are fp32.
    # FSDP will use low precision during forward.
    llm.to(dtype)
    return llm

# @logger.catch
def pretrain(args):
    ###########################################################################
    #                           1. Environment                                #
    ###########################################################################
    with etc.collect_execute_time("init_comm_time"):
        catch_error_node(initialize_distributed_env)(
            config=args.train_cfg,
            launcher='torch',
            master_port=args.port,
            seed=args.seed,
            old_config=True
        )
    assert hasattr(gpc, "config") and gpc.config is not None

    # train_folder = gpc.config.data.train_folder
    # dataset_types, dataset_subset_types = get_lang_subset_types(gpc.config.data.train_folder)

    data_rank = gpc.get_local_rank(ParallelMode.DATA)
    data_world_size = gpc.get_world_size(ParallelMode.DATA)

    setup_parallel(sp_size=args.sp_size, tp_size=1)
    set_random_seed(args.seed)

    dp_mesh = get_dp_mesh()
    tp_mesh = get_tp_mesh()
    sp_mesh = get_sp_mesh()
    fsdp_mesh = get_fsdp_mesh()  # dp_size * sp_size
    world_mesh = get_world_mesh()  # dp_size * sp_size * tp_size

    dp_size = dp_mesh.size()
    tp_size = tp_mesh.size()
    sp_size = sp_mesh.size()
    world_size = world_mesh.size()

    if args.use_hsdp:
        hsdp_device_mesh = init_device_mesh(
            DEVICE, (world_size // 8, 8), mesh_dim_names=('internode', 'intranode'))
    else:
        hsdp_device_mesh = None

    # if args.global_batch_size < dp_size or args.global_batch_size % dp_size:
    #     raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
    #                      'should be divisible by the '
    #                      f'world_size({world_size}).')
    #
    # if (args.global_batch_size / dp_size) % args.mirco_batch_size:
    #     raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
    #                      f'should be divisible by the world_size({world_size})'
    #                      f' * `mirco_batch_size`({args.mirco_batch_size})')

    rank = dist.get_rank()
    mkdir_or_exist(args.work_dir)
    abnormal_log_dir = os.path.join(args.work_dir, 'abnormal_logs')
    mkdir_or_exist(abnormal_log_dir)

    log_file = os.path.join(args.work_dir, f'rank{rank}.log')
    dataset_types, dataset_subset_types = get_lang_subset_types(
        gpc.config.data.train_folder
    )

    log_tb = False
    log_wandb = args.log_wandb
    if log_tb:
        tbwriter = SummaryWriterWrapper(
            log_dir=args.work_dir+f"/rank_{dist.get_rank()}",
            dataset_types=dataset_types,
            queue_size=gpc.config.data.total_steps*400)

    if log_wandb:
        wandb.login(key="your wandb key") 
        log_dir = args.work_dir
        wandbwriter = WandbWrapper(
            project_name=args.wandb_project_name,
            name=args.wandb_name,
            dataset_types=dataset_types,
            queue_size=gpc.config.data.total_steps*400)
        print(f"log_dir: {log_dir}")

    # Change the log format printed in the terminal
    lvl = 'DEBUG' if args.debug else 'INFO'
    logger.add(sys.stderr, level=lvl, format=log_format(rank, args.debug))
    # Change the format saved in the log file
    logger.add(log_file, format=log_format(rank), backtrace=True, catch=True)

    if rank == 0:
        env = collect_env()
        import transformers

        import xtuner
        env['Transformers'] = transformers.__version__
        env['XTuner'] = f'{xtuner.__version__}+{get_git_hash(digits=6)}'
        runtime_env = OrderedDict()
        runtime_env.update(env)
        runtime_env['Seed'] = args.seed
        runtime_env['World Size'] = world_size
        runtime_env['DP Size'] = dp_size
        runtime_env['SP Size'] = sp_size
        runtime_env['TP Size'] = tp_size
        # runtime_env['Distributed launcher'] = dist_launcher

        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        logger.info('\n' + dash_line + '\nRuntime environment:' +
                    runtime_env_info + '\n' + dash_line + '\n')
    # -------------------    Environment  End  ------------------------------ #
    # -------------------    Environment  End  ------------------------------ #
    if args.resume_from and args.resume is False:
        args.resume = True
    if args.resume is True and args.resume_from is None:
        # find last checkpoint
        ckpt_dirs = [d for d in os.listdir(args.work_dir) if
                     os.path.isdir(os.path.join(args.work_dir, d)) and d.startswith('ckpt-')]
        if len(ckpt_dirs) > 0:
            ckpt_dirs.sort(reverse=True)
            is_success = False
            for ckpt_dir in ckpt_dirs:
                if os.path.exists(os.path.join(args.work_dir, ckpt_dir, '.metadata')):
                    args.resume_from = os.path.join(args.work_dir, ckpt_dir)
                    is_success = True
                    break
                else:
                    os.system(
                        f'rm -rf {os.path.join(args.work_dir, ckpt_dir)}')
            if is_success is False:
                logger.warning(
                    'Did not find last_checkpoint to be resumed. training from scratch.')
                args.resume = False
        else:
            logger.warning(
                'Did not find last_checkpoint to be resumed. training from scratch.')
            args.resume = False
    if args.resume:
        assert not args.checkpoint_drop_optimizer, '`resume` and `checkpoint_drop_optimizer` cannot be set at the same time.'

    ###########################################################################
    #                     1.2 replace config                                  #
    ###########################################################################
    
    args.wd = gpc.config.adam.weight_decay
    args.lr = gpc.config.adam.lr
    args.adam_beta1 = gpc.config.adam.adam_beta1
    args.adam_beta2 = gpc.config.adam.adam_beta2
    args.adam_epsilon = gpc.config.adam.adam_eps
    args.total_steps = gpc.config.data.total_steps
    args.decay_steps = gpc.config.data.decay_steps
    args.warmup_steps = gpc.config.data.warmup_steps
    args.decay_type = gpc.config.data.decay_type
    args.iters_per_step = gpc.config.data.gradient_accumulation
    args.seq_len = gpc.config.data.seq_len
    args.lr_min = gpc.config.MIN_LEARNING_RATE
    args.vocab_size = gpc.config.model.vocab_size
    args.different_bsz = gpc.config.data.different_bsz
    args.micro_bsz = gpc.config.data.micro_bsz

    assert args.micro_bsz == 1, f"micro_bsz must be 1, got {args.micro_bsz}"

    logger.info(args)
    logger.info(f"data_rank: {data_rank}, data_world_size: {data_world_size}")
    ###########################################################################
    #                     2. Dataset & Dataloader                             #
    ###########################################################################

    start_load_data_t = time.time()

    assert varlen_attn_is_available()

    with etc.collect_execute_time("load_data_time"):
        train_dl = build_train_loader_with_data_type(
            data_cfg=gpc.config.data,
            data_rank=data_rank,
            data_world_size=data_world_size,
        )
    train_state = get_train_state(train_dl)

    load_data_cost_time = time.time() - start_load_data_t
    logger.info(f'[Dataset & Dataloader] Cost {load_data_cost_time:.2f}s')
    # -------------------    Dataset & Dataloader  End  --------------------- #

    ###########################################################################
    #                          3. FSDP                                        #
    ###########################################################################
    if args.dtype == 'auto':
        args.dtype = 'bf16' if DEVICE_MODULE.is_bf16_supported() else 'fp16'

    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        if DEVICE_MODULE.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            raise RuntimeError('The device does not support `bf16`, '
                               'please set `dtype` to `fp16`.')
    else:
        raise RuntimeError('`dtype` only supports `fp16`, `bf16` or `auto`, '
                           f'but found {args.dtype}.')

    if args.dtype == 'bf16':
        print("这里开启了 bf16 精度")
        
    # AutoConfig.register("transformer_rnn", TransformerConfig_rnn)
    # AutoModelForCausalLM.register(TransformerConfig_rnn, TransformerForCausalLM_rnn)

    # current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # sys.path.append(current_dir)
    llm_dir = os.path.dirname(args.llm) # args.llm should be .json
    sys.path.append(llm_dir)

    from modeling_transformer_rnn import TransformerForCausalLM_rnn
    from configuration_transformer_rnn import TransformerConfig_rnn
    AutoConfig.register("transformer_rnn", TransformerConfig_rnn)
    AutoModelForCausalLM.register(TransformerConfig_rnn, TransformerForCausalLM_rnn)

    llm_cfg = AutoConfig.from_pretrained(args.llm)
    # if is_flash_attn_2_available():
    #     llm_cfg.attn_implementation = 'flash_attention_2'

    llm_cfg.use_cache = False
    llm_cfg.torch_dtype = dtype

    # Only load parameters on rank 0 to avoid each rank repeatedly loading the
    # same model into the CPU, wasting memory
    xtuner_load_timeout = timedelta(minutes=60)
    group_gloo = dist.new_group(backend='gloo', timeout=xtuner_load_timeout)

    # Only load parameters on rank 0 to avoid each rank repeatedly loading the
    # same model into the CPU, wasting memory
    if rank == 0:
        with torch.device('cpu'):
            rank0_llm = build_llm_model(args, llm_cfg, world_size, dtype)
    else:
        rank0_llm = None

    dist.monitored_barrier(group=group_gloo, timeout=xtuner_load_timeout)
    logger.info('after barrier')

    with torch.device('meta'):
        # Ensure all numerical values in the optimizer are fp32.
        # FSDP will use low precision during forward.
        llm = build_llm_model(args, llm_cfg, world_size, dtype)
        print("注意这里 dispatch model")
        dispatch_hf_code(llm)
        for module in llm.modules():
            for p_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    param_fp32 = torch.nn.Parameter(
                        param.to(dtype=torch.float32))
                    setattr(module, p_name, param_fp32)

    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

    with profile_time_and_memory('[Parallelize LLM]'):
        megatron_parallelize(
            llm,
            rank0_llm,
            dp_mesh=hsdp_device_mesh if args.use_hsdp else fsdp_mesh,
            tp_mesh=tp_mesh,
            mp_policy=mp_policy,
            recompute_ratio=args.selective_recompute,
            reshard_after_forward=True if args.reshard_after_forward else False)

        llm.train()

    required_grad_params = [
        param for param in llm.parameters() if param.requires_grad
    ]
    if rank == 0:
        logger.info(llm)
        logger.info(llm.config)
        total_params = non_emb_params = 0
        for param in required_grad_params:
            if args.vocab_size not in param.shape:
                non_emb_params += param.numel()
            total_params += param.numel()
        logger.info(f"模型总参数: {total_params}, 非 embedding/output 层(tie=false)参数: {non_emb_params}")

    dist.barrier()
    
    # --------------------------    FSDP  End  ------------------------------ #

    ###########################################################################
    #                      4. Optimizer & Scheduler                           #
    ###########################################################################
    optimizer = AdamW(
        required_grad_params,
        lr=args.lr,
        weight_decay=args.wd,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        fused=True
    )

    # `iter` means once forward+backward
    # `step` means once optimizer step
    # `iters_per_step` means gradient accumulative counts
    iters_per_step = args.iters_per_step
    total_steps = args.total_steps  # batch

    if args.checkpoint_interval == -1:
        checkpoint_interval = total_steps
    elif args.checkpoint_interval < 1:
        checkpoint_interval = int(total_steps * args.checkpoint_interval)
    else:
        checkpoint_interval = int(args.checkpoint_interval)

    if args.hf_interval == -1:
        hf_interval = total_steps
    elif args.hf_interval < 1:
        hf_interval = int(total_steps * args.hf_interval)
    else:
        hf_interval = int(args.hf_interval)

    max_keep_ckpts = args.max_keep_ckpts
    if max_keep_ckpts <= 0:
        # save all checkpoints
        max_keep_ckpts = total_steps + 100000

    save_hf_ckpt_names = []
    save_pt_ckpt_names = []
    ckpt_dirs = [os.path.join(args.work_dir, d) for d in os.listdir(args.work_dir) if
                 os.path.isdir(os.path.join(args.work_dir, d)) and d.startswith('ckpt-')]
    if len(ckpt_dirs) > 0:
        ckpt_dirs.sort()
        save_pt_ckpt_names = ckpt_dirs

    hf_dirs = [os.path.join(args.work_dir, d) for d in os.listdir(args.work_dir) if
               os.path.isdir(os.path.join(args.work_dir, d)) and d.startswith('hf-')]
    if len(hf_dirs) > 0:
        hf_dirs.sort()
        save_pt_ckpt_names = hf_dirs

    if args.warmup_steps < 1:
        warmup_steps = int(args.warmup_steps * total_steps)
    else:
        warmup_steps = int(args.warmup_steps)

    if args.decay_steps < 1:
        decay_steps = int(args.decay_steps * total_steps)
    else:
        decay_steps = int(args.decay_steps)

    '''
    def warmup_fn(x):
        return x / warmup_steps if x < warmup_steps else 1
    warmup_scheduler = LambdaLR(optimizer, warmup_fn)

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr_min)
    '''

    lr_scheduler = get_wsd_schedule(
        # warmup + stable + decay
        # decay阶段 sqrt衰减
        optimizer,
        num_warmup_steps=warmup_steps,
        num_stable_steps=total_steps-warmup_steps-decay_steps,
        num_decay_steps=decay_steps,
        min_lr_ratio=args.lr_min/args.lr,  # how much the learning rate will be reduced at the end of the training
        warmup_type="linear",
        decay_type=args.decay_type,
    )

    dp_rank = get_dp_mesh().get_local_rank()

    # ----------------    Optimizer & Scheduler End   ----------------------- #
    if args.resume:
        logger.info(f'[Resume] Resume from {args.resume_from}')
        _options = StateDictOptions(
            cpu_offload=True, ignore_frozen_params=True)
        (shard_model_state_dict,
         shard_optimizer_state_dict) = get_state_dict(
            llm, optimizer, options=_options)
        state_dict = {
            'model': shard_model_state_dict,
            'optimizer': shard_optimizer_state_dict,
            'train_state': train_state,
            'lr_scheduler': lr_scheduler
        }
        # inplace state_dict
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=args.resume_from,
        )

        _options = StateDictOptions(
            cpu_offload=True, strict=False)
        set_state_dict(
            llm,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
            options=_options
        )
        if hasattr(train_state, "batch_sampler") and not isinstance(
                train_state.batch_sampler, torch.utils.data.sampler.BatchSampler
        ):
            sampler_states = torch.load(
                os.path.join(args.resume_from, "sampler.pt"))
            train_dl.batch_sampler.load_state_dict(sampler_states)
            # track the actual updates of sampler when using weighted sampling
            train_state.init_batch_sampler(train_dl.batch_sampler)

        assert hasattr(train_state, "data_state_dict") and hasattr(
            train_state, "batch_sampler")
        dataset_dirs = [os.path.join(args.resume_from, d) for d in os.listdir(args.resume_from) if
                        d.startswith('dataset_')]
        dataset_dirs.sort()
        state_dict_list = []
        for dataset_dir in dataset_dirs:
            state_dict_pre_rank = torch.load(dataset_dir)
            state_dict_list.append(state_dict_pre_rank)

        if dp_rank == 0:
            cur_dataset_consumed_tokens = state_dict_list[0].pop(
                "dataset_consumed_tokens", {})
            train_state.data_state_dict["dataset_consumed_tokens"].update(
                cur_dataset_consumed_tokens)

        if len(state_dict_list) == dp_size:
            train_dl.dataset.load_state_dict(state_dict_list[dp_rank])
        else:
            if state_dict_list[0]["epochs_to_use"]:
                raise NotImplementedError(
                    "Cannot resume training if dp_size changed with `epochs_to_use` set."
                    " Try set `epochs_to_use` to None."
                )
            # if dp_rank == 0:
            #     logger.info(state_dict_list)
            # logger.info('=============================================================')
            multiple_packed_states_group: Dict[str,
                                               List[Dict]] = defaultdict(list)
            consumed_samples = defaultdict(int)
            for state_dict in state_dict_list:
                for key, value in state_dict["consumed_samples"].items():
                    consumed_samples[key] += value
                for key, value in state_dict["multiple_packed_states"].items():
                    multiple_packed_states_group[key].append(value)
            used_epochs = [state_dict["used_epochs"]
                           for state_dict in state_dict_list]
            max_used_epochs = {k: max(d[k] for d in used_epochs)
                               for k in used_epochs[0]}

            for key in list(multiple_packed_states_group.keys()):
                sort_metrics = [
                    (
                        state_dict["tokenization_states"]["aggregation_states"]["file_shift"],
                        state_dict["tokenization_states"]["aggregation_states"]["jsonl_states"]["line_shift"],
                        state_dict["seq_offset"],
                    )
                    for state_dict in multiple_packed_states_group[key]
                ]
                multiple_packed_states_group[key] = sorted(
                    zip(sort_metrics, multiple_packed_states_group[key]), key=lambda x: x[0]
                )[-1][-1]

            if dp_rank == 0:
                state_dict = {
                    "rng_state": np.random.RandomState(seed=args.seed).get_state(),
                    "multiple_packed_states": multiple_packed_states_group,
                    "consumed_samples": consumed_samples,
                    "used_epochs": max_used_epochs,
                }
            else:
                state_dict = {
                    "rng_state": np.random.RandomState(
                        seed=args.seed + dp_rank
                    ).get_state(),
                    "multiple_packed_states": multiple_packed_states_group,
                    "consumed_samples": {},
                    "used_epochs": max_used_epochs,
                }
            # logger.info(f" --------------- {state_dict['consumed_samples'], multiple_packed_states_group}")
            train_dl.dataset.load_state_dict(state_dict)

    # print('===============',train_state.batch_count)
    if train_state.batch_count + 1 >= total_steps:
        logger.info("Training has finished, exiting...")
        return

    gpc.train_state = train_state
    
    ###########################################################################
    #                          5. Training                                    #
    ###########################################################################
    
    # 训练开始前调用 gc
    gc.collect()

    '''e2e_tgs 开始计时'''
    start_train_t = time.time()
    total_consumed_tokens = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_memory = torch.cuda.max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024 ** 3):.1f}GB')

    train_iter = iter(train_dl)

    # print("create tokenizer")
    # tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)
    last_step_loss, last_grad_norm = 1e5, 1e5

    for batch_count in itertools.count(train_state.batch_count):
        if batch_count > total_steps:
            break

        '''aimaster detection, in the loop, same level as optimizer.step()'''
        if my_tracer is not None:
            my_tracer.step()
        '''gc detection'''
        if is_interval(train_state.batch_count, total_steps, args.gc_interval):
            gc.collect()

        lr_scheduler.step()
        cur_lr = lr_scheduler.get_last_lr()[0]

        torch.cuda.reset_peak_memory_stats()

        step_loss = 0
        '''tgs start timing'''
        step_start_t = time.time()
        step_consumed_tokens = 0

        _data_start_t = time.time()

        step_data_list = []
        rank_grad_tokens = 0

        # the first dim is grad accumulated step
        # train_iter seems not used later
        # TODO: here we need to concatenate data from different subfolders
        # micro_bsz=1, use different_bsz
        # input shape: (micro_num, packed_len), packed_len = different_bsz * seq_len
        # 更改 inputs, labels, cu_seqlens, type_ids
        # 更改 batch (tuple)中的 batch[0]['type_ids'], 后续统计三个 lang info 需要用到
        # here three quantities must be int type
        input_ids, labels, type_ids = (
            torch.tensor([], dtype=torch.int32), ) * 3
        # cu_seqlens is tensor (grad accumulation=1) or list (>1)
        # here we use list
        cu_seqlens = [torch.tensor([0], dtype=torch.int32)] * iters_per_step

        for i in range(args.different_bsz):
            batch, train_iter = load_new_batch_with_train_state(
                train_dl=train_dl, train_iter=train_iter, train_state=train_state)

            cur_inputs, cur_labels = batch
            # (micro_num==1, packed_len), from the same subfolder
            cur_input_ids = cur_inputs['input_ids']
            # cu_seqlens like: [[   0,   80,  272,  ..., 2036, 2048]]
            # varlen: cu_seqlens may have different lengths in each row, but the last value must be seqlen
            cur_cu_seqlens = cur_inputs['cu_seqlens']
            
            # Fix: `cu_seqlens` perhaps start with 2 zeros
            # Note: Considering the problem of gradient accumulation,
            # `cu_seqlens` may be a 2d-tensor or a List of tensors with different lengths, 
            # `cu_seqlens.shape` == (1, packed_len) when grad accumulation=1
            cur_cu_seqlens = [torch.unique(cu_seqlens_iter) for cu_seqlens_iter in cur_cu_seqlens]
            
            cur_type_ids = cur_inputs['type_ids']
            # considering the varlen problem, cu_seqlens is a tensor when grad accumulation=1;
            # `cu_seqlens` could be a tensor or list

            assert cur_input_ids.shape[0] == cur_labels.shape[0] == cur_type_ids.shape[0] == \
                iters_per_step, "shape[0] of input_ids, labels, type_ids must be the number of grad accumulation steps"

            input_ids = torch.cat([input_ids, cur_input_ids], dim=1)
            labels = torch.cat([labels, cur_labels], dim=1)
            type_ids = torch.cat([type_ids, cur_type_ids], dim=1)
            # unify cu_seqlens as list
            # concat([[0]], [[0, 10, 20, 48]], [[0, 30, 48]])
            # -> [[0, 10, 20, 48, 30+48, 48+48]]
            cu_seqlens = [torch.cat([a, b[1:] + i*args.seq_len])
                          for a, b in zip(cu_seqlens, cur_cu_seqlens)]

            # when cur_labels is a 2d tensor, here we only += the number of grad accumulation steps
            train_state.num_consumed_samples_in_epoch += len(cur_labels)

        gpc.config.batch_count = batch_count
        train_state.batch_count = batch_count
        # here it was len(batch[1]), which is labels
        # train_state.num_consumed_samples_in_epoch += len(labels)
        '''
        cuseq = cu_seqlens[0]
        diffseq = torch.diff(cuseq)
        if diffseq.min() == 0:
            logger.warning(f"""
        [Data] Step {batch_count} number of sequences: {len(diffseq)} min/max sequence length: {diffseq.min()},{diffseq.max()}, cuseq: {cuseq}, diffseq: {diffseq}
        [Abnormal Input IDS] {input_ids}
        """)
            
        logger.info(f"""
        [Data info] cuseq: {cuseq}, diffseq: {diffseq}
        min/max sequence length: {diffseq.min()},{diffseq.max()}
        """)
        '''
        

        for _iter in range(iters_per_step):

            input_ids_iter = input_ids[_iter: _iter + 1]
            labels_iter = labels[_iter: _iter + 1]
            cu_seqlens_iter = cu_seqlens[_iter]
            num_token = cu_seqlens_iter[1:] - cu_seqlens_iter[:-1]

            if num_token[-1] == 0:
                num_token = num_token[:-1]

            rank_grad_tokens += (labels_iter >= 0).sum()
            step_data_list.append({"input_ids": input_ids_iter,
                                   "labels": labels_iter,
                                   "num_tokens": num_token})

        rank_grad_tokens = rank_grad_tokens.to(DEVICE)
        dist.all_reduce(rank_grad_tokens)
        global_grad_tokens = rank_grad_tokens / tp_size / sp_size

        step_data_time = time.time() - _data_start_t
        unreduced_losses = []
        for _iter in range(iters_per_step):
            data = step_data_list[_iter]
            input_ids = data['input_ids'].to(DEVICE)
            labels = data['labels'].to(DEVICE)
            num_tokens = data['num_tokens'].to(DEVICE)
            
            # TODO: Avoid `num_tokens` got only 1 or 2 sample length
            if num_tokens.min() < args.min_sample_length:
                # 1. only keep samples with length >= `min_sample_length`
                keep_mask = num_tokens >= args.min_sample_length
                filtered_num_tokens = num_tokens[keep_mask]
                # 2. construct the start and end indices of the samples based on num_tokens
                cum_tokens = torch.cumsum(num_tokens, dim=0)
                start_indices = torch.cat([
                    torch.tensor([0], device=cum_tokens.device), cum_tokens[:-1]])
                # 3. filter input_ids and labels
                # get the start and end indices of the retained samples
                start = start_indices[keep_mask]
                end = cum_tokens[keep_mask]
                # concatenate the token intervals of each retained sample
                indices = torch.cat(
                    [torch.arange(s, e, device=input_ids.device) for s, e in zip(start, end)])
                
                logger.warning(
                    f"[Drop Data] sequence length {num_tokens.min()}<{args.min_sample_length} is too small "
                    f"delete some data. original {num_tokens=}, deleted {filtered_num_tokens=}")
                # Update input_ids, labels, num_tokens
                input_ids, labels = input_ids[:, indices], labels[:, indices]
                num_tokens = filtered_num_tokens

            if sp_size > 1:
                # `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
                input_ids = pad_for_sequence_parallel(
                    input_ids, 0, sp_mesh, dim=1)
                _num_pad = input_ids.numel() - num_tokens.sum()
                if _num_pad > 0:
                    _num_pad = torch.IntTensor([_num_pad]).to(DEVICE)
                    num_tokens = torch.cat([num_tokens, _num_pad], dim=-1)

                input_ids = split_for_sequence_parallel(
                    input_ids, dim=1, sp_mesh=sp_mesh)

                labels = pad_for_sequence_parallel(
                    labels, -100, sp_mesh, dim=1)

                labels = split_for_sequence_parallel(
                    labels, dim=1, sp_mesh=sp_mesh)

            # packed_ctx = packed_sequence(num_tokens, sp_mesh=sp_mesh)

            # with packed_ctx:

            # original code
            '''
            logits = llm(input_ids=input_ids, use_cache=False).logits

            loss = F.cross_entropy(logits.squeeze(), labels.squeeze(), reduction='none')  # 1, seqlen
            '''
            # copy from xtuner/_lite/accelerate/packed.py @contextmanager
            _zero_length = torch.zeros(1, device=DEVICE)
            _pad_length = torch.cat([_zero_length, num_tokens]).int()
            cumulative_lengths = torch.cumsum(_pad_length, 0).int()
            position_ids = [torch.arange(num.item()) for num in num_tokens]
            position_ids = torch.cat(position_ids, dim=0).to(DEVICE)
            # (1, packed_len)
            # `causal_conv1d_cuda.causal_conv1d_fwd` requires `seq_ids` to be int32
            position_ids = position_ids.unsqueeze(0).to(torch.int32)
            if sp_mesh:
                # `dim` is 1 as the shape of tensor is (bs, seq_len)
                position_ids = split_for_sequence_parallel(
                    position_ids, dim=1, sp_mesh=sp_mesh)
            
            loss = llm(input_ids, 
                       labels=labels,
                       position_ids=position_ids,
                       cu_seqlens=cumulative_lengths,
                       max_seqlen=num_tokens.max()
                    #    attention_mask = ...
                    ).loss # mean loss
            loss /= iters_per_step

            # collect for logging
            # unreduced_losses: list, 长度为 micro_num
            # unreduced_losses.append(unreduced_loss.detach().clone())

            '''here we don't use logits, remove the calculation of pred acc
            # original code
            pred = logits.argmax(dim=-1)  # B L
            correct_pred = pred == labels
            correct_preds.append(correct_pred)
            '''
            # unreduced_losses.append(loss.detach().clone())

            if sp_size > 1:
                sp_group = sp_mesh.get_group()
                sp_pt_loss = dist.nn.functional.all_gather(loss, sp_group)
                sp_pt_labels = dist.nn.functional.all_gather(
                    labels, sp_group)

                loss = torch.cat(sp_pt_loss, dim=-1)
                labels = torch.cat(sp_pt_labels, dim=-1)

            # loss = loss / global_grad_tokens * dp_size # for sum loss

            # gradient accumulation: backward micronum times, pass the gradient back, but the parameters will not be updated
            # finally, optimizer.step()
            loss.backward()

            step_loss += loss.item()

            step_consumed_tokens += num_tokens.sum() / sp_size / tp_size

            train_state.step_count += 1
        grad_norm = clip_grad_norm_(
            required_grad_params, fsdp_mesh, args.max_grad_norm)

        if grad_norm.isnan() or grad_norm.isinf():
            train_state.inf_nan_skip_batches += 1
            logger.warning(
                f"Step {train_state.batch_count+1}/{total_steps}: The grad norm is NaN={grad_norm.isnan()} or Inf={grad_norm.isinf()}, skip this batch.")
            optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()

        '''tgs end timing'''
        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - train_state.batch_count - 1)
        eta = timedelta(seconds=int(eta))
        tgs = int(step_consumed_tokens / step_time)
        max_memory = torch.cuda.max_memory_allocated()
        total_consumed_tokens += step_consumed_tokens
        end2end_tgs = int(total_consumed_tokens /
                          (time.time() - start_train_t))

        # log to tensorboard
        tensorboard_start_time = time.time()
        if log_tb and is_interval(train_state.batch_count, total_steps, args.tb_interval):
            tbwriter.add_data_infos(type_ids, train_state, batch_count+1)
            tbwriter.add_train_dynamics(
                step_loss, unreduced_losses, type_ids, batch_count+1)
            '''add one line to record the learning rate'''
            tbwriter.add_optimize_info(
                grad_norm.detach().clone(), train_state, cur_lr, batch_count+1)
            tbwriter.add_speed_info(tgs, end2end_tgs, batch_count+1)

        if log_wandb and is_interval(
                train_state.batch_count, total_steps, args.tb_interval):
            wandbwriter.add_data_infos(type_ids, train_state, batch_count+1)
            wandbwriter.add_train_dynamics(
                step_loss, unreduced_losses, type_ids, batch_count+1)
            '''add one line to record the learning rate'''
            wandbwriter.add_optimize_info(
                grad_norm.detach().clone(), train_state, cur_lr, batch_count+1)
            wandbwriter.add_speed_info(tgs, end2end_tgs, batch_count+1)
        # 注意这里要 barrier
        dist.barrier()
        tensorboard_time = time.time() - tensorboard_start_time

        if is_interval(train_state.batch_count, total_steps, args.log_interval):
            logger.info(f'[Train] (Epoch ?) Step '
                        f'{train_state.batch_count+1}/{total_steps}  '
                        f'lr: {cur_lr:.7f}  loss: {step_loss:.4f}  '
                        f'grad_norm: {grad_norm:.4f}  '
                        f'max_memory: {(max_memory / 1024 ** 3):.1f}GB  '
                        f'text_tokens: {step_consumed_tokens}  '
                        f'total_skip_nan_inf: {train_state.inf_nan_skip_batches}  '
                        f'tgs: {tgs} e2e_tgs: {end2end_tgs} data_time: {step_data_time:.2f}s  '
                        f'time: {step_time:.2f}s tb_time: {tensorboard_time:.2f} '
                        f'eta: {eta}')

        num_digits = len(str(abs(total_steps)))

        if args.abnormal_detect:
            # if (grad_norm - last_grad_norm > 5) or (step_loss - last_step_loss > 0.45) or (
            #         train_state.batch_count+1 == args.abnormal_steps):
            if train_state.batch_count+1 in args.abnormal_steps or grad_norm.isnan() or grad_norm.isinf():
                logger.warning(f'[Abnornmal Train] Step '
                               f'{train_state.batch_count+1}/{total_steps}  '
                               f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  '
                               f'last_step_loss: {last_step_loss:.3f}  '
                               f'grad_norm: {grad_norm:.4f}  '
                               f'last_grad_norm: {last_grad_norm:.4f}')
                txt_file = os.path.join(
                    abnormal_log_dir, f'abnormal_rank{dist.get_rank()}.txt')
                data_state_dict = train_state.data_state_dict

                with open(txt_file, 'a') as f:
                    f.write(
                        f'[Abnormal Train] Step {train_state.batch_count+1}/{total_steps}\n')
                    f.write(
                        f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  last_step_loss: {last_step_loss:.3f}\n')
                    f.write(
                        f'grad_norm: {grad_norm:.4f}  last_grad_norm: {last_grad_norm:.4f}\n')
                    # f.write(f'input text: {tokenizer.decode(input_ids.ravel())}\n')
                    f.write(
                        f'multiple_packed_states: {data_state_dict["multiple_packed_states"]}\n')
                    f.write(
                        f'consumed_samples: {data_state_dict["consumed_samples"]}\n')
                    f.write('\n' + '='*30 + '\n')  # separator, to distinguish each record

        last_grad_norm, last_step_loss = grad_norm, step_loss

        if is_interval(train_state.batch_count, total_steps, hf_interval):
            DEVICE_MODULE.empty_cache()
            hf_dir = os.path.join(
                args.work_dir, f'hf-{train_state.batch_count+1:0{num_digits}}')

            with profile_time_and_memory('[HF Checkpoint]'):

                from torch.distributed._tensor import DTensor

                if rank == 0:
                    llm_state_dict = {}

                for name, param in llm.state_dict().items():
                    if isinstance(param, DTensor):
                        with torch.no_grad():
                            full_param = param.full_tensor().cpu()
                    else:
                        full_param = param.cpu()

                    if rank == 0:
                        llm_state_dict[name] = full_param

                if rank == 0:
                    rank0_llm.load_state_dict(llm_state_dict)
                    rank0_llm.save_pretrained(hf_dir)
                    # tokenizer.save_pretrained(hf_dir)

                dist.barrier()

            if dist.get_rank() == 0:
                save_hf_ckpt_names.append(hf_dir)
                if len(save_hf_ckpt_names) > max_keep_ckpts:
                    remove_hf_ckpt_name = save_hf_ckpt_names.pop(0)
                    os.system(f'rm -rf {remove_hf_ckpt_name}')

            max_memory = torch.cuda.max_memory_allocated()
            logger.info('[HF Checkpoint] During saving HF checkpoint, the peak GPU '
                        f'memory is {max_memory / 1024 ** 3:.1f}GB.')

        if is_interval(train_state.batch_count, total_steps, checkpoint_interval):
            if args.checkpoint_drop_optimizer:
                logger.warning('The saved checkpoint cannot be resumed. '
                               'If you want to save a resumable checkpoint, '
                               'please remove `--checkpoint-drop-optimizer` '
                               'from the command.')
            else:
                with profile_time_and_memory('[PT Checkpoint]'):
                    ckpt_id = f'{train_state.batch_count+1:0{num_digits}}-of-{total_steps:0{num_digits}}'
                    ckpt_dir = os.path.join(args.work_dir, f'ckpt-{ckpt_id}')
                    if dp_rank == 0:
                        mkdir_or_exist(ckpt_dir)
                    dist.barrier()

                    if hasattr(train_state, "data_state_dict"):  # TODO:  tp/sp/pp
                        assert hasattr(train_state, "batch_sampler")
                        torch.save(train_state.data_state_dict, os.path.join(
                            ckpt_dir, f"dataset_{dp_rank}.pt"))

                        if dp_rank == 0:
                            if hasattr(train_state, "batch_sampler") and not isinstance(
                                    train_state.batch_sampler, torch.utils.data.sampler.BatchSampler
                            ):
                                sampler_state = train_state.batch_sampler.state_dict()
                                torch.save(sampler_state, os.path.join(
                                    ckpt_dir, "sampler.pt"))
                        else:
                            train_state.data_state_dict["dataset_consumed_tokens"] = defaultdict(
                                int)

                    dist.barrier()

                    future = None

                    with profile_time_and_memory('[PT Checkpoint Wait]'):
                        if future is not None:
                            wait([future])

                    with profile_time_and_memory('[PT Checkpoint of DCP ASYNC]'):
                        # FSDP cannot be saved via torch.save
                        # Refer to https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html  # noqa: E501
                        _options = StateDictOptions(
                            cpu_offload=True, ignore_frozen_params=True)
                        (shard_model_state_dict,
                         shard_optimizer_state_dict) = get_state_dict(
                            llm, optimizer, options=_options)

                        state_dict = {
                            'model': shard_model_state_dict,
                            'optimizer': shard_optimizer_state_dict,
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'train_state': train_state.state_dict(),
                        }
                        future = dcp.async_save(
                            state_dict, checkpoint_id=ckpt_dir, process_group=group_gloo)

                        def send_to_oss_and_remove(future):
                            # send to oss and remove local file
                            # TODO: send to oss

                            if dist.get_rank() == 0:
                                save_pt_ckpt_names.append(ckpt_dir)
                                if len(save_pt_ckpt_names) > max_keep_ckpts:
                                    remove_pt_ckpt_name = save_pt_ckpt_names.pop(
                                        0)
                                    os.system(f'rm -rf {remove_pt_ckpt_name}')
                            # print('============send_to_oss_and_remove callback==================')

                        future.add_done_callback(send_to_oss_and_remove)

            max_memory = torch.cuda.max_memory_allocated()
            logger.info('[Checkpoint] During saving checkpoint, the peak GPU '
                        f'memory is {max_memory / 1024 ** 3:.1f}GB.')

    train_cost_time = time.time() - start_train_t
    logger.info(f'[Train] Cost {timedelta(seconds=int(train_cost_time))}')
    # ------------------------    Training  End  ---------------------------- #

    print("==========================")
    print("===========end===========")
    print("==========================\n")
    return 0


if __name__ == '__main__':
    args = parse_args()
    pretrain(args)
