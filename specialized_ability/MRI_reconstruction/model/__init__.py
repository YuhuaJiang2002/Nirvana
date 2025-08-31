# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.transformer.configuration_transformer import TransformerConfig
from fla.models.transformer.modeling_transformer import (
    TransformerForCausalLM, TransformerModel)

from .configuration_transformer_rnn import TransformerConfig_rnn
from .modeling_transformer_rnn import TransformerForCausalLM_rnn, TransformerModel_rnn
from .task_aware_delta_net import Task_Aware_Delta_Net
from .ttt_cross_layer import TTT_Cross_Layer
from .varnet_nirvana_custom import CustomNirvanaModel
from .image_decoder import create_image_decoder
from .k_space_encoder import KSpaceEncoder

AutoConfig.register(TransformerConfig.model_type, TransformerConfig)
AutoModel.register(TransformerConfig, TransformerModel)
AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM)

AutoConfig.register(TransformerConfig_rnn.model_type, TransformerConfig_rnn)
AutoModel.register(TransformerConfig_rnn, TransformerModel_rnn)
AutoModelForCausalLM.register(TransformerConfig_rnn, TransformerForCausalLM_rnn)

__all__ = ['TransformerConfig', 'TransformerForCausalLM', 'TransformerModel',
           'TransformerConfig_rnn', 'TransformerForCausalLM_rnn', 'TransformerModel_rnn',
           'Task_Aware_Delta_Net', 'TTT_Cross_Layer', 'CustomNirvanaModel', 'create_image_decoder', 'KSpaceEncoder']
