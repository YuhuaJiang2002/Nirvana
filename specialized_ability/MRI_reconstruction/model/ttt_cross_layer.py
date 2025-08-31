import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.autograd import Function

class TTT_Cross_Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_size = config.concept_dim   # 128
        self.concept_dim = config.concept_dim  # 128
        # self.linear = nn.Linear(self.input_size, self.hidden_size)
        # self.ln = nn.LayerNorm(self.hidden_size)

        # self.logit_dim = 32
        self.logit_dim = config.logit_dim

        self.weight_linear = nn.Parameter(torch.empty(self.concept_dim, self.input_size, self.logit_dim))
        self.weight_ln = nn.Parameter(torch.empty(self.concept_dim, self.logit_dim))
        self.bias_linear = nn.Parameter(torch.empty(self.concept_dim, self.logit_dim))
        self.bias_ln = nn.Parameter(torch.empty(self.concept_dim, self.logit_dim))

        # self.weight_linear_tmp = torch.empty_like(self.weight_linear)
        # self.weight_ln_tmp = torch.empty_like(self.weight_ln)
        # self.bias_linear_tmp = torch.empty_like(self.bias_linear)
        # self.bias_ln_tmp = torch.empty_like(self.bias_ln)
        
        self.config = config
        self.init_weights()
    # def init_tmp_weights(self):
    #     weight_linear_tmp = self.weight_linear.clone().to(self.weight_linear.device).to(self.weight_linear.dtype)
    #     weight_ln_tmp = self.weight_ln.clone().to(self.weight_ln.device).to(self.weight_ln.dtype)
    #     bias_linear_tmp = self.bias_linear.clone().to(self.bias_linear.device).to(self.bias_linear.dtype)
    #     bias_ln_tmp = self.bias_ln.clone().to(self.bias_ln.device).to(self.bias_ln.dtype)
    #     params = {
    #         'weight_linear_tmp': weight_linear_tmp,
    #         'weight_ln_tmp': weight_ln_tmp,
    #         'bias_linear_tmp': bias_linear_tmp,
    #         'bias_ln_tmp': bias_ln_tmp
    #     }
    #     return params
    
    def init_params_as_logits(self, batch_size, sequence_length):
        weight_linear_tmp = torch.ones(batch_size, sequence_length, self.logit_dim).to(self.weight_linear.device).to(self.weight_linear.dtype)
        weight_ln_tmp = torch.ones(batch_size, sequence_length, self.logit_dim).to(self.weight_linear.device).to(self.weight_linear.dtype)
        bias_linear_tmp = torch.ones(batch_size, sequence_length, self.logit_dim).to(self.weight_linear.device).to(self.weight_linear.dtype)
        bias_ln_tmp = torch.ones(batch_size, sequence_length, self.logit_dim).to(self.weight_linear.device).to(self.weight_linear.dtype)
        
        params = {
            'weight_linear_tmp': weight_linear_tmp,
            'weight_ln_tmp': weight_ln_tmp,
            'bias_linear_tmp': bias_linear_tmp,
            'bias_ln_tmp': bias_ln_tmp
        }
        return params

    def init_weights(self):
        # torch.manual_seed(42) 
        nn.init.normal_(self.weight_linear, mean=0.0, std=self.config.initializer_range)
        nn.init._no_grad_fill_(self.weight_ln, 1.0 / self.logit_dim)
        # nn.init.zeros_(self.bias_linear)
        # nn.init.zeros_(self.bias_ln)
        nn.init.normal_(self.bias_linear, mean=0.0, std=self.config.initializer_range / self.logit_dim)
        nn.init.normal_(self.bias_linear, mean=0.0, std=self.config.initializer_range / self.logit_dim)

    def get_weight_per_token(self, params):
        
        weight_linear_tmp = torch.einsum('iol,bsl->bsio', self.weight_linear, params['weight_linear_tmp'])
        weight_ln_tmp = torch.einsum('ol,bsl->bso', self.weight_ln, params['weight_ln_tmp'])
        bias_linear_tmp = torch.einsum('ol,bsl->bso', self.bias_linear, params['bias_linear_tmp'])
        bias_ln_tmp = torch.einsum('ol,bsl->bso', self.bias_ln, params['bias_ln_tmp'])

        return weight_linear_tmp, weight_ln_tmp, bias_linear_tmp, bias_ln_tmp

    def learn(self, k, v, params, lr_linear=1, lr_ln=1, eps=1e-6):
        # k v size: [batch_size, length, channel_dim]
        # batch_size, seq_length, channel_dim = k.shape
        # weight_linear_tmp = params['weight_linear_tmp']
        # weight_ln_tmp = params['weight_ln_tmp']
        # bias_linear_tmp = params['bias_linear_tmp']
        # bias_ln_tmp = params['bias_ln_tmp']
        weight_linear_tmp, weight_ln_tmp, bias_linear_tmp, bias_ln_tmp = self.get_weight_per_token(params)
        # 1. reshape
        # k_reshaped = k.reshape(-1, channel_dim)  # [batch_size*length, channel_dim]
        
        # output_reshaped = self.predict(k_reshaped, params)  # [batch_size*length, channel_dim]
        # z = F.linear(k_reshaped, params['weight_linear_tmp'], params['bias_linear_tmp'])
        # mu = z.mean(dim=-1, keepdim=True)
        # var = z.var(dim=-1, keepdim=True, unbiased=False)

        z = torch.einsum('bsi,bsio->bso', k, weight_linear_tmp) + bias_linear_tmp
        mu = z.mean(dim=-1, keepdim=True)
        var = z.var(dim=-1, keepdim=True, unbiased=False)

        # Normalization
        
        std = torch.sqrt(var + eps)
        z_hat = (z - mu) / std     
        # output_reshaped = params['weight_ln_tmp'] * z_hat + params['bias_ln_tmp'] + k
        output_reshaped = weight_ln_tmp * z_hat + bias_ln_tmp + k

        # v_reshaped = v.reshape(-1, channel_dim)
        # error_reshaped = output_reshaped - v_reshaped  # [batch_size*length, channel_dim]
        error_reshaped = output_reshaped - v

        # ln_rate = learning_rate * 0.1  
        grad_weight_ln_temp = error_reshaped * z_hat
        # grad_weight_ln = grad_weight_ln_temp.mean(dim=0) # 
        # weight_ln_tmp = weight_ln_tmp - ln_rate * grad_weight_ln # sequence length, channel_dim
        grad_weight_ln = grad_weight_ln_temp
        # batch_size, sequence length, logit_dim
        params0 = params['weight_ln_tmp'] - lr_ln * torch.einsum('ol,bso->bsl', self.weight_ln, grad_weight_ln)
        
        # bias_update = ln_rate * error_reshaped # .mean(dim=0)
        # bias_ln_tmp = bias_ln_tmp - bias_update # batch_size, sequence length, concept_dim
        grad_bias_ln = error_reshaped
        params1 = params['bias_ln_tmp'] - lr_ln * torch.einsum('ol,bso->bsl', self.bias_ln, grad_bias_ln)

        # linear weight: [out_dim, in_dim]
        # grad_linear_temp = error_reshaped - error_reshaped.mean(dim=-1, keepdim=True) - z_hat * grad_weight_ln_temp.mean(dim=-1, keepdim=True)
        grad_linear = weight_ln_tmp * error_reshaped / std # batch_size, sequence length, concept_dim
        # grad_weight_linear = grad_linear.t() @ k  # [channel_dim, channel_dim]
        grad_weight_linear = torch.einsum('bsi,bso->bsio', k, grad_linear)
        # weight_linear_tmp = weight_linear_tmp - learning_rate * grad_weight_linear.mean(dim=0)
        params2 = params['weight_linear_tmp'] - lr_linear * torch.einsum('iol,bsio->bsl', self.weight_linear, grad_weight_linear)

        grad_b = grad_linear #.mean(dim=0)  # [channel_dim]
        # bias_linear_tmp = bias_linear_tmp - learning_rate * grad_b
        params3 = params['bias_linear_tmp'] - lr_linear * torch.einsum('ol,bso->bsl', self.bias_linear, grad_b)
        
        params_new = {
            'weight_linear_tmp': params2,
            'weight_ln_tmp': params0,
            'bias_linear_tmp': params3,
            'bias_ln_tmp': params1
        }

        return params_new

    def predict(self, q, params):
        weight_linear_tmp, weight_ln_tmp, bias_linear_tmp, bias_ln_tmp = self.get_weight_per_token(params)
        z = torch.einsum('bsi,bsio->bso', q, weight_linear_tmp) + bias_linear_tmp
        mu = z.mean(dim=-1, keepdim=True)
        var = z.var(dim=-1, keepdim=True, unbiased=False)

        # Normalization
        eps = 1e-6
        std = torch.sqrt(var + eps)
        z_hat = (z - mu) / std     
        # output_reshaped = params['weight_ln_tmp'] * z_hat + params['bias_ln_tmp'] + k
        output = weight_ln_tmp * z_hat + bias_ln_tmp + q

        return output
    



