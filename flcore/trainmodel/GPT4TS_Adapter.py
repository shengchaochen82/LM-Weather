import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from flcore.trainmodel.layers.Autoformer_EncDec import series_decomp, SeasonalTrendDecomp
from flcore.trainmodel.layers.Embed import DataEmbedding, TokenEmbedding, PositionalEmbedding
from peft import get_peft_config, get_peft_model, LoraConfig, PrefixTuningConfig, TaskType, IA3Config, AdaLoraConfig, AdaLoraModel
from utils.data_utils import RevIN

class GPT4TSPrompt(nn.Module):
    def __init__(self, configs):
        super(GPT4TSPrompt, self).__init__()
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

        self.enc_embedding_seasonal = DataEmbedding(configs.enc_in, configs.d_model // 2, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_embedding_trend = DataEmbedding(configs.enc_in, configs.d_model // 2, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_embedding_residual = DataEmbedding(configs.enc_in, configs.d_model // 2, configs.embed, configs.freq,
                                           configs.dropout)

        # self.decomposition = series_decomp(kernel_size=configs.moving_avg)
        self.decomposition = SeasonalTrendDecomp(kernel_size=configs.moving_avg, period=10)

        # Reverser Instance Normalization
        self.revin_layer = RevIN(num_features=configs.enc_in, affine=True)

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

            self.down_layer = nn.Sequential(
                nn.Linear(4096, configs.d_model),
                nn.LayerNorm(configs.d_model),
                nn.Linear(configs.d_model, 2560)
            )

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

        else:
            KeyError

        for name, param in self.named_parameters():
            # layer contains low-rank parameters
            if 'layer' in name:
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
        if 'OWD2' in self.configs.dataset:
            residual, seasonal, trend = residual[:, :, :M], seasonal[:, :, :M], trend[:, :, :M]
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
        
        residual, seasonal, trend = self.decomposition(x_enc)
        residual, seasonal, trend = self.revin_layer(residual, 'norm'), self.revin_layer(seasonal, 'norm'), self.revin_layer(trend, 'norm')
        # x_enc = self.revin_layer(x_enc, 'norm')

        residual = self.enc_embedding_residual(residual, x_mark_enc)
        seasonal = self.enc_embedding_seasonal(seasonal, x_mark_enc)
        trend = self.enc_embedding_trend(trend, x_mark_enc)

        # decomposition_init = torch.cat([residual, seasonal, trend], dim=0)
        
        residual = rearrange(residual, 'b l m -> b m l')
        residual = self.padding_patch_layer(residual)
        residual = residual.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        residual = rearrange(residual, 'b m n p -> (b m) n p')

        trend = rearrange(trend, 'b l m -> b m l')
        trend = self.padding_patch_layer(trend)
        trend = trend.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        trend = rearrange(trend, 'b m n p -> (b m) n p')

        seasonal = rearrange(seasonal, 'b l m -> b m l')
        seasonal = self.padding_patch_layer(seasonal)
        seasonal = seasonal.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        seasonal = rearrange(seasonal, 'b m n p -> (b m) n p')

        # outputs = self.in_layer(x_enc)
        decomposition_init = torch.cat([trend, seasonal, residual], dim=-1)

        decomposition_init = self.in_layer(decomposition_init)

        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        # dec_out = self.time_proj_layer(outputs)

        dec_out = outputs

        dec_out = self.revin_layer(dec_out, 'denorm')
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.forecast(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        return None


    
