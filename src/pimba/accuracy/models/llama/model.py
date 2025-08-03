import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

from ... import configs
from .modeling_llama import LlamaForCausalLM


class LLaMA(HFLM):
    def __init__(self, name: str, revision: str):
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            revision=revision,
            trust_remote_code=True,
        )

        # model
        model = LlamaForCausalLM.from_pretrained(
            name,
            revision=revision,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map=configs.DEVICE,
        )

        super().__init__(pretrained=model, tokenizer=tokenizer, batch_size=1)
