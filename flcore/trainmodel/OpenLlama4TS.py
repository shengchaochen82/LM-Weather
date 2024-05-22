from einops import rearrange
import torch
import torch.nn as nn
from flcore.trainmodel.layers.Embed import DataEmbedding
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from peft import get_peft_config, get_peft_model, LoraConfig, PrefixTuningConfig, TaskType, AdaLoraConfig, AdaLoraModel

class OpenLlama4TS(nn.Module):
    def __init__(self, configs):
        super(OpenLlama4TS, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        if configs.pretrain:
            self.openllama = LlamaForCausalLM.from_pretrained(
                'llm/open_llama_3b', torch_dtype=torch.float16, device_map='auto', 
                output_attentions=True, output_hidden_states=True)
        else:
            print("------------------no pretrain------------------")
            self.openllama = LlamaForCausalLM(LlamaConfig())
        self.openllama.model.layers = self.openllama.model.layers[:configs.gpt_layers]

        if configs.is_peft:
            print('loading peft into the pre-trained language model....')
            self.peft_(configs)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear_pre = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.ln = nn.LayerNorm(configs.d_ff)
            self.out_layer = nn.Linear(configs.d_ff, configs.c_out)

        if self.task_name == 'imputation':
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
            print('------------------LoRA Tuning------------------')
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=configs.rank, lora_alpha=32, lora_dropout=configs.dropout)
            self.openllama = get_peft_model(self.openllama, peft_config)

        else:
            NotImplementedError

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

        outputs = self.openllama(inputs_embeds=enc_out).hidden_state
        
        outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        dec_out = dec_out * stdev
        dec_out = dec_out + means

        return dec_out
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x_enc.shape
        
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

        dec_out = self.openllama(inputs_embeds=enc_out).hidden_state

        # hidden_state is embedding attention, other is each layer attention
        dec_out = dec_out[0][:, :, :self.d_ff]
        dec_out = self.out_layer(dec_out)

        dec_out = dec_out * stdev
        dec_out = dec_out + means
        
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
    







