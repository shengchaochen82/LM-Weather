import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from flcore.trainmodel.layers.Embed import DataEmbedding
from peft import get_peft_config, get_peft_model, LoraConfig, PrefixTuningConfig, TaskType, IA3Config, AdaLoraConfig, AdaLoraModel

from utils.data_utils import RevIN


class GPT4TS(nn.Module):
    def __init__(self, configs):
        super(GPT4TS, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff

        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        # Reverser Instance Normalization
        self.revin_layer = RevIN(num_features=configs.enc_in)

        if configs.pretrain:
            self.gpt2 = GPT2Model.from_pretrained('llm/gpt2', output_attentions=True, output_hidden_states=True)
        else:
            self.gpt2 = GPT2Model(GPT2Config())

        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]

        if configs.is_peft:
            print('loading peft into the pre-trained language model....')
            self.peft_(configs)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
            self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
            self.time_proj_layer = nn.Linear(configs.enc_in, configs.c_out)

        if self.task_name == 'imputation':
            self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
            self.ln_proj = nn.LayerNorm(configs.d_model)
            self.out_layer = nn.Linear(
                configs.d_model, 
                configs.c_out, 
                bias=True)
            
    def peft_(self, configs):
        """
        Function for parameters-efficient fine-tunning including lora, adaptlora, prefix, ia3, prompt-tuning...
        """
        if configs.peft == 'lora':
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=configs.rank, lora_alpha=32, lora_dropout=configs.dropout)
            self.gpt2 = get_peft_model(self.gpt2, peft_config)

        elif configs.peft == 'adaptlora':
            peft_config = AdaLoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=configs.rank, lora_alpha=32, lora_dropout=configs.dropout)
            self.gpt2 = AdaLoraModel(self.gpt2, peft_config, adapter_name='default')
            # self.gpt2 = get_peft_model(self.gpt2, peft_config)

        elif configs.peft == 'prefix':
            print('------------------P-Tuning------------------')
            peft_config = PrefixTuningConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                num_virtual_tokens=20, 
                encoder_hidden_size=768, 
                token_dim=768, 
                num_transformer_submodules=1, 
                num_attention_heads=12, 
                num_layers=12, 
                prefix_projection=False)
            self.gpt2 = get_peft_model(self.gpt2, peft_config)

        else:
            KeyError

        for name, param in self.named_parameters():
            if "enc_embedding" in name or "decomposition" in name or "layer" in name:
                param.requires_grad = True

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        B, L, M = x_enc.shape
        
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        x_enc = rearrange(x_enc, 'b l m -> b m l')
        
        x_enc = self.padding_patch_layer(x_enc)
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_enc = rearrange(x_enc, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x_enc)

        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        
        outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        dec_out = dec_out * stdev
        dec_out = dec_out + means
        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x_enc.shape
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        x_enc = self.revin_layer(x_enc, 'norm')
        # print(x_enc.shape)

        x_enc = rearrange(x_enc, 'b l m -> b m l')
        
        x_enc = self.padding_patch_layer(x_enc)
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_enc = rearrange(x_enc, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x_enc)
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        dec_out = self.time_proj_layer(outputs)

        # dec_out = dec_out * stdev
        # dec_out = dec_out + means
        dec_out = self.revin_layer(dec_out, 'denorm')
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        return None


    
