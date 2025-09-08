import os
import json
import math
from torch.optim.lr_scheduler import _LRScheduler
# from internlm.config.config import read_base
from functools import partial

# with read_base():
#     from configs._base_.default_runtime import *  # pylint: disable=wildcard-import,unused-wildcard-import
#     from configs._base_.models.internlm.internlm3_moe_20B_A3B import *  # pylint: disable=wildcard-import,unused-wildcard-import
#     from configs._base_.monitors.base import *  # pylint: disable=wildcard-import,unused-wildcard-import
# model['extra_pred_tokens'] = N
# note off the script above will envoke training with multi-token-prediction: predict next N+1 tokens.

if "JOB_NAME" in os.environ:
    JOB_NAME = os.environ["JOB_NAME"]
else:
    JOB_NAME = os.path.basename(__file__).split(".py")[0]

DO_FEISHU_ALERT = True
# monitor = dict(
#     # feishu alert configs
#     alert=dict(
#         # light_monitor address to send heartbeat
#         light_monitor_address=os.getenv("LIGHT_MONITORING_ADDRESS"),
#         heartbeat_interval=500,  # the interval steps for sending heartbeat
#         alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
#         alert_keys=alert_keys,
#     ),
# )

VOCAB_FILE = './Llama-2-7b-chat-hf/tokenizer.model'
VOCAB_SIZE = 32000
model = dict(
    checkpoint=0.6,
    num_chunks=1,
    # vocab_size=128133,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    attention_type="GQA",
    hidden_size=2048,
    num_attention_heads=32,
    num_kv_attention_heads=4,
    mlp_layer_fusion=True,
    num_layers=32,
    multiple_of=1,
    no_bias=True,
    apply_post_layer_norm=False,
    dtype="torch.bfloat16",
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-6,
    qk_interleaved=False,
    rope_base=500000.0,
    # MoE part
    num_experts=64,
    top_k=8,
    num_shared_experts=0,
    moe_type="Dropless",
    residual_type="deepseek",
    first_k_dense_replace=0,
    moe_layer_freq=1,
    moe_intermediate_size=1536,
    enable_qk_norm=False,
    moe_layer_kwargs=dict(
        capacity_factor=None,
        normalize_expert_weights=True,
    ),
)

model["deepnorm_scale_cfg"] = dict(
    init_scale_factor=math.pow(2.0 * 32, 0.5),
    residual_scale_factor=math.pow(2.0 * 32, 0.5),
)


parallel = dict(
    zero1=dict(size=1, fsdp=False),  # 32
    tensor=dict(size=1, mode="isp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=1, overlap=True, memory_pool=False),
    expert=dict(size=1),
    expert_weight=dict(size=1, overlap=True, memory_pool=False),  # 32
    expert_zero1=dict(size=1),
)


# If set, will enable debug mode
# In non-debug mode, all the local changes are requested to be committed, otherwise the training will not start
DEBUG = 1
DO_SFT = False

skip_check_parallel_statistic = False
# Whether to enble `spawn` mode for pytorch multiprocessing. If set to False, will use `fork` mode during training.
MP_SPAWN = False

ENABLE_SAVE_CKPT = True

TRAIN_FOLDER: str = f"<YOUR_PATH>"
VALID_FOLDER = None
tensorboard_folder: str = f"<YOUR_PATH>"

# minicpm GBS = 3.93M (120 gpus, packed_len=32768)
# STEP_1B = 960 # gpu=64, 4, 4096
# STEP_1B = 640 # gpu=96, 4, 4096
# STEP_1B = 320 # gpu=96, 8, 4096
STEP_1B = 955 # gpu=64; 4, 4096
WARMUP_STEP = int(STEP_1B*1) # WARMUP
WS_STEP = int(STEP_1B*1)  # warmup+stable
DECAY_STEP = int(STEP_1B*99)
# 'cosine', 'miror_cosine', 'linear', 'exp', 'square', 'sqrt'
DECAY_TYPE = "cosine"

TOTAL_STEP = WS_STEP + DECAY_STEP # warmup+stable + decay

VALID_EVERY = TOTAL_STEP + 99999

MICRO_NUM = 1
VALID_MICRO_NUM = 1
GRADIENT_ACCUMULATION = MICRO_NUM

MICRO_BATCH_SIZE = 1 # should always be 1
DIFFERENT_BATCH_SIZE = 4  # packed_length = DIFFERENT_BATCH_SIZE * seq_len
SEQ_LEN = 4096
GLOBAL_BATCH_SIZE = 0
MIN_LENGTH = 50

# Two settings: "streaming" and "tokenized"
# If set to "streaming", will use streaming dataset (on-the-fly tokenize)
# If set to "tokenized", will use pre-tokenized dataset
DATASET_TYPE = "streaming"
# If the above DATASET_TYPE is "preprocessed", ignore this param.
# If set to 1, will use streaming dataset v1
# If set to 2, will use streaming dataset v2
STREAMING_DATASET_VERSION = 2
# Truncation rules for the pack process.
# It is recommended to set it to :
# `none`(streaming dataset v1)     | "cut"(streaming dataset v2) for pre-training
# `complete`(streaming dataset v1) | "pad"(streaming dataset v2) for fine-tuning tasks to keep the context intact.
# "pad"(tokenized dataset) | "cut"(tokenized dataset)
PACK_DATASET_BREAK_MODE = "cut"

# if VALID_PACK_DATASET_BREAK_MODE set to None, won't use packed validation
# It is recommended to set it to :
# "cut"(streaming dataset v2) for pre-training
# "pad"(streaming dataset v2) for fine-tuning tasks to keep the context intact.
# only v2 version is supported.
# when set to packed mode, we will disable VALID_MICRO_NUM.
VALID_PACK_DATASET_BREAK_MODE = None

# You might change it for better tgs
VALID_PACKED_LENGTH = SEQ_LEN

# If set to -1, will use SEQ_LEN as the max length of each sample
MAX_LENGTH_PER_SAMPLE = SEQ_LEN  # Or set as -1

# There are four tokenizer_wrapper types: "pretrain", "sft", "sft_multi_round", "fim", "online_prompt"
TOKENIZER_WRAPPER_TYPE = "pretrain"

LEARNING_RATE = 4e-4 # 1.8B: 3e-4
MIN_LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.1 

WARMUP_RATIO = 0.01 

CHECKPOINT_EVERY = 1000
SNAPSHOT_FREQ = CHECKPOINT_EVERY // 4


# The following configs are OSS_NAME & OSS_IP in different clusters.
OSS_HEAD = "boto3:s3"
OSS_NAME = "checkpoints_ssd_02"
OSS_IP = "10.135.7.249"  # P/T cluster


# Ckpt folder format:
#  fs: 'local: /mnt/nfs/XXX'
# oss: 'boto3: s3://model_weights/XXX'

SAVE_CKPT_FOLDER = f"local:./ckpts/{JOB_NAME}"

# If you want to train from scratch, set LOAD_CKPT_FOLDER to None.
LOAD_CKPT_FOLDER = None  
# LOAD_CKPT_FOLDER = f"boto3:s3://{OSS_NAME}.{OSS_IP}/{JOB_NAME}/{CHECKPOINT_EVERY}"

# NOTE: there are 2 params in LOAD_CKPT_FOLDER_INFO: content and ckpt_type.
# content should be in "all", "model", "sampler", "optimizer", "scheduler"
# ckpt_type should be in "internlm", "llama", "fuxi", "newton", "maibao", "plato", "to_internlm2", "internlm2"
LOAD_CKPT_FOLDER_INFO = dict(path=LOAD_CKPT_FOLDER, content=[
                             "all"], ckpt_type="internevo")

'''
ZH_WEIGHTS = {k: (v / sum(ZH_WEIGHTS.values())) *
              0.2 for k, v in ZH_WEIGHTS.items()}
EN_WEIGHTS = {k: (v / sum(EN_WEIGHTS.values())) *
              0.5 for k, v in EN_WEIGHTS.items()}
CODE_WEIGHTS = {k: (v / sum(CODE_WEIGHTS.values())) *
                0.2 for k, v in CODE_WEIGHTS.items()}
MATH_WEIGHTS = {k: (v / sum(MATH_WEIGHTS.values())) *
                0.1 for k, v in MATH_WEIGHTS.items()}
DATASET_WEIGHTS = {**ZH_WEIGHTS, **EN_WEIGHTS, **
                   CODE_WEIGHTS, **MATH_WEIGHTS}
'''

DATASET_WEIGHTS = {"finewebedu-sample-100BT-jsonl": 1.0}

DATASET_WEIGHTS = {k: v for k, v in DATASET_WEIGHTS.items() if v > 0}

ckpt = dict(
    # Save ckpt settings
    # If set to True, will save ckpt to save_ckpt_folder.
    enable_save_ckpt=ENABLE_SAVE_CKPT,
    save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save ckpt.
    # Save ckpt frequency
    checkpoint_every=CHECKPOINT_EVERY,
    oss_snapshot_freq=SNAPSHOT_FREQ,
    # Load ckpt settings
    # If set to True, will auto-load the latest checkpoint in save_ckpt_folder.
    auto_resume=True,
    # If auto_resume is False, will load checkpoint from load_ckpt_folder.
    load_ckpt_info=LOAD_CKPT_FOLDER_INFO,
    # Other infos
    async_upload=True,
    async_upload_tmp_folder=f"./internlm_tmp_ckpt_{JOB_NAME}/",
    stop_file_path=f"llm_alter/{JOB_NAME}.log",
)

# if set to True, model's norm will use fp32 instead of model.dtype
use_fp32_norm = False

# Frequency of checking the model weights under different process groups
# parallel_check_freq = 100
parallel_check_freq = 1000
data_check_counts = 0
empty_cache_and_diag_interval = 500
metric = dict(
    tensorboard_interval=100,
    metric_interval=1,
    skip_timer=False,
    tensorboard_max_queue=100 * 1024,
    tensorboard_flush_secs=60
)
compute_subset_loss = False
# if you want to filter something or use online prompt by generating dynamic attibutes:
# folder that save meta info for attibutes dataset
META_FOLDER = None

# Use mmap version of meta data. Enabling this option helps prevent memory overflow,
# but you need to use tools/scripts/make_mmap_meta_file.py to pre-process meta data.
# '0' means that the mmap mode is not applicable, '1' means that the mmap mode is
# started for the *.npy file nearby, and '2' means that the mmap mode is further
# started for the *.npz file, which is more time-consuming but saves more memory.
MMAP_META_STAGE = 2

# field that you want use in meta infos
# for example ATTRIBUTES = {"advertisement", "influency"}
ATTRIBUTES = None

# function name that reflects from train_internlm/data/attribute_manager/attibute_filter_ops.py
# use this if only you can decide wether to filter by only extra attributes
# you may create your own function in train_internlm/data/attribute_manager/attibute_filter_ops.py
# for example PREV_FILTER_FUNC = "prev_filter_op"
PREV_FILTER_FUNC = None

# function name that reflects from train_internlm/data/attribute_manager/attibute_filter_ops.py
# use this if you want to filter with content info or you want generate dynamic attibutes
# you may create your own function in train_internlm/data/attribute_manager/attibute_filter_ops.py
# for example DYNAMIC_ATTRIBUTE_FUNC = "dynamic_attr_op"
# DYNAMIC_ATTRIBUTE_FUNC = "dynamic_attr_op_v4"
DYNAMIC_ATTRIBUTE_FUNC = None

# if you want do online prompting on jsonl samples, you should do like we do in below:
# according to data filed "data_type" generating online prompt
# most of the time it is a must to use DYNAMIC_ATTRIBUTE_FUNC with it
# You should also set tokenizer_wrapper to `online_prompt`
# ONLINE_PROMPT = {
#     "porn":{"This is a porn sample:\n":0.3, "This is a good sample with porn information:\n":0.6},
#     "politic":{"This is a politic sample:\n":0.7},
# }

# individualy you may use fim_conf to select run pretraining with FIM Loss:
# You should also set tokenizer_wrapper to `fim`
# FIM_CONF = {
#     'fim_rate':0.5, # how much proportion of sample should use FIM, the rest remain CE
#     'pre_token_id':103027, # token that indicates start of the first part of FIM
#     'mid_token_id':103028, # token that indicates start of the middle part of FIM
#     'suf_token_id':103029, # token that indicates start of the rest part of FIM
#     'eot_token_id':103030, # token that indicates the end of FIM prediction
# }
FIM_CONF = None

SUBSET_PARAMS = {}

data = dict(
    seed=1024,
    type=DATASET_TYPE,
    version=STREAMING_DATASET_VERSION,
    sft=DO_SFT,  # if do sft training, set True
    tokenizer_wrapper=TOKENIZER_WRAPPER_TYPE,
    train_folder=TRAIN_FOLDER,
    valid_folder=VALID_FOLDER,
    vocab_file=VOCAB_FILE,  # pylint: disable=undefined-variable
    seq_len=SEQ_LEN,
    # Datasets with less than `MIN_LENGTH` will be discarded
    min_length=MIN_LENGTH,
    # micro_num means the number of micro_batch contained in one gradient update
    micro_num=MICRO_NUM,
    micro_bsz=MICRO_BATCH_SIZE,
    different_bsz=DIFFERENT_BATCH_SIZE,
    # defaults to the value of micro_num
    valid_micro_num=VALID_MICRO_NUM,
    total_steps=TOTAL_STEP,
    decay_steps=DECAY_STEP,
    warmup_steps=WARMUP_STEP,
    decay_type=DECAY_TYPE,
    # defaults to 0, means disable evaluate
    valid_every=VALID_EVERY,
    pack_sample_into_one=False,
    skip_batches="",
    rampup_batch_size="",
    num_worker=12, 
    gradient_accumulation=GRADIENT_ACCUMULATION,
    text_field="text", 
    val_text_field="text",
    prompt_text_field="prompt",
    output_text_field="output",  
    dataset_weights=DATASET_WEIGHTS,  # sample_data_weights
    # dataset_weights=None,
    break_mode=PACK_DATASET_BREAK_MODE,
    drop_last=False,
    max_length_per_sample=MAX_LENGTH_PER_SAMPLE,
    meta_folder=META_FOLDER,
    mmap_meta_stage=MMAP_META_STAGE,
    attributes=ATTRIBUTES,
    prev_filter_func_str=PREV_FILTER_FUNC,
    dynamic_attributes_func_str=DYNAMIC_ATTRIBUTE_FUNC,
    # which means loading arributes on the fly
    lazy_load_attribute=True,
    online_prompt=None,
    fim_conf=FIM_CONF,
    empty_cache_and_diag_interval=500,
    diag_outlier_ratio=1.1,
    # only work for TOKENIZER_WRAPPER_TYPE = "pretrain"
    # if set to True, each splited sample will have bos
    always_bos=False,
    valid_pack_mode=VALID_PACK_DATASET_BREAK_MODE,
    valid_packed_length=VALID_PACKED_LENGTH,
    valid_drop_last=False,
    subset_params=SUBSET_PARAMS,
    probe_size=-1, 
    # probe_size=int(1e6),  
    tokenizer_chunk_num=8,
    # skip_resuming_dataset=None,
    # inject_dyname_word_prompt=True,
)

grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**14,
        # the minimum loss scale, defaults to None
        min_scale=1,
        # the number of steps to increase loss scale when no overflow occurs
        growth_interval=1000,
    ),
    # the multiplication factor for increasing loss scale, defaults to 2
    growth_factor=2,
    # the multiplication factor for decreasing loss scale, defaults to 0.5
    backoff_factor=0.5,
    # the maximum loss scale, defaults to None
    max_scale=2**24,
    # the number of overflows before decreasing loss scale, defaults to 2
    hysteresis=2,
)

loss = dict(label_smoothing=0.0)

adam = dict(
    lr=LEARNING_RATE,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=WEIGHT_DECAY,
)


beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

cudnn_deterministic = False
cudnn_benchmark = False

enable_tb = True
