from einops import rearrange
import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from peft import get_peft_config, get_peft_model, LoraConfig, PrefixTuningConfig, TaskType, IA3Config

class OpenLlama4TS(nn.Module):
    def __init__(self, configs):
        super(OpenLlama4TS, self).__init__()

        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        if configs.pretrain:
            self.openllama = LlamaForCausalLM.from_pretrained(
                'llm/open_llama_3b', torch_dtype=torch.float16, device_map='auto')
        else:
            print("------------------no pretrain------------------")
            self.openllama = LlamaForCausalLM(LlamaConfig())
        self.openllama.model.layers = self.openllama.model.layers[:configs.gpt_layers]
        self.pred_len = configs.pred_len

        if configs.is_peft:
            print('loading peft into the pre-trained language model....')
            self.peft_(configs)

        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        self.time_proj_layer = nn.Linear(20, 1)

    def peft_(self, configs):
            """
            Function for parameters-efficient fine-tunning including lora, adaptlora, prefix, ia3, prompt-tuning...
            """
            if configs.peft == 'lora':
                print('------------------LoRA Tuning------------------')
                # target_modules = ['query', 'value', 'dense']
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION, r=configs.rank, lora_dropout=configs.dropout)
                self.openllama = get_peft_model(self.openllama, peft_config)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x)

        outputs = self.openllama(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = self.time_proj_layer(outputs)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
    







