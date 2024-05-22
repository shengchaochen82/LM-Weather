import torch
import torch.nn as nn
import numpy as np
from flcore.trainmodel.layers.Embed import DataEmbedding
from flcore.trainmodel.layers.Autoformer_EncDec import series_decomp
from einops import rearrange
from transformers.models.bert import BertModel, BertConfig, BertTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, PrefixTuningConfig, TaskType, PromptTuningConfig, AdaptionPromptConfig, AdaptionPromptModel, AdaLoraConfig, AdaLoraModel

class Bert4TS_Adapter(nn.Module):

    def __init__(self, configs):
        super(Bert4TS_Adapter, self).__init__()

        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff

        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        self.enc_embedding_seasonal = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_embedding_trend = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.decomposition = series_decomp(kernel_size=configs.moving_avg)

        if configs.pretrain:
            self.bert = BertModel.from_pretrained("llm/bert-base-uncased", torch_dtype=torch.float16,
                                                  output_attentions=True, output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained("llm/bert-base-uncased", torch_dtype=torch.float16)
        else:
            print('------------------no pretrain------------------')
            self.bert = BertModel(BertConfig())
        self.bert.encoder.layer = self.bert.encoder.layer[:1]
  
        if configs.is_peft:
            print('===========================PEFT========================')
            self.peft_(configs)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
            self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
            self.time_proj_layer = nn.Linear(20, 1)

        if self.task_name == 'imputation':
            self.ln_proj = nn.LayerNorm(configs.d_model)
            self.out_layer = nn.Linear(
                configs.d_model, 
                configs.c_out, 
                bias=True)

    def peft_(self, configs):
        """
        Function for parameters-efficient fine-tunning including lora, adaptlora, prefix, ia3, prompt-tuning
        """
        if configs.peft == 'lora':
            self.peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=configs.rank, lora_alpha=32, lora_dropout=configs.dropout)
            self.bert = get_peft_model(self.bert, self.peft_config)

        else:
            NotImplementedError

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

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        outputs = self.bert(inputs_embeds=enc_out).hidden_states
        
        outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        dec_out = dec_out * stdev
        dec_out = dec_out + means

        return dec_out
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x_enc.shape

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x_enc /= stdev

        seasonal_init, trend_init = self.decomposition(x_enc)

        seasonal_init = self.enc_embedding_seasonal(seasonal_init, x_mark_enc)
        trend_init = self.enc_embedding_trend(trend_init, x_mark_enc)

        x_enc = rearrange(x_enc, 'b l m -> b m l')
        x_enc = self.padding_patch_layer(x_enc)
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_enc = rearrange(x_enc, 'b m n p -> (b m) n p')
        
        outputs = self.in_layer(x_enc)

        seasonal_init, trend_init = seasonal_init.reshape(-1, 12, 768), trend_init.reshape(-1, 12, 768)
        outputs = torch.cat([seasonal_init, trend_init, outputs], dim=0)
        outputs = self.bert(inputs_embeds=outputs).last_hidden_state
        outputs = outputs[-2560:]
 
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = self.time_proj_layer(outputs)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        return None
