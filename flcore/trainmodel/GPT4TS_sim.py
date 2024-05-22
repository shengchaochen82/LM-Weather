import numpy as np
import torch
import torch.nn as nn

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from transformers import GPT2Tokenizer
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from peft import get_peft_config, get_peft_model, LoraConfig, PrefixTuningConfig, TaskType, IA3Config


class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        self.pred_len = configs.pred_len

        if configs.is_peft:
            print('loading peft into the pre-trained language model....')
            self.peft_(configs)

        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        self.time_proj_layer = nn.Linear(20, 1)
        
        for param in self.gpt2.parameters():
            param.requires_grad = True
 
        for layer in (self.gpt2, self.in_layer, self.out_layer, self.time_proj_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0

    def peft_(self, configs):
        """
        Function for parameters-efficient fine-tunning including lora, adaptlora, prefix, ia3, prompt-tuning...
        """
        if configs.peft == 'lora':
            print('------------------LoRA Tuning------------------')
            # target_modules = ['query', 'value', 'dense']
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=configs.rank, lora_dropout=configs.dropout)
            self.gpt2 = get_peft_model(self.gpt2, peft_config)
        elif configs.peft == 'prefix':
            print('------------------P-Tuning------------------')
            peft_config = PrefixTuningConfig(
                task_type=TaskType.FEATURE_EXTRACTION, num_virtual_tokens=20, encoder_hidden_size=768, 
                token_dim=768, num_transformer_submodules=1, num_attention_heads=12, num_layers=12, prefix_projection=False)
            self.gpt2 = get_peft_model(self.gpt2, peft_config)
        elif configs.peft == 'ia3':
            print('------------------IA3------------------')
            peft_config = IA3Config(
                task_type=TaskType.FEATURE_EXTRACTION)
            self.gpt2 = get_peft_model(self.gpt2, peft_config)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        B, L, M = x_enc.shape
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        
        outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    # def forecast(self, x, x_mark_enc, x_dec, x_mark_dec):
    #     B, L, M = x.shape

    #     means = x.mean(1, keepdim=True).detach()
    #     x = x - means
    #     stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
    #     x /= stdev

    #     x = rearrange(x, 'b l m -> b m l')
        
    #     x = self.padding_patch_layer(x)
    #     x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
    #     x = rearrange(x, 'b m n p -> (b m) n p')

    #     outputs = self.in_layer(x)
    #     if self.is_gpt:
    #         outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

    #     outputs = self.out_layer(outputs.reshape(B*M, -1))
    #     outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

    #     outputs = self.time_proj_layer(outputs)

    #     outputs = outputs * stdev
    #     outputs = outputs + means

    #     return outputs
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x_enc.shape
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        enc_out = torch.nn.functional.pad(enc_out, (0, 768-enc_out.shape[-1]))

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        # dec_out = dec_out.reshape(B, -1)
        
        # dec_out = self.ln(dec_out)
        dec_out = self.out_layer(dec_out)
        # print(dec_out.shape)
        # dec_out = dec_out.reshape(B, self.pred_len + self.seq_len, -1)
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        
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