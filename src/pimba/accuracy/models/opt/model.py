from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

from ... import configs
from .modeling_opt import OPTForCausalLM


class OPT(HFLM):
    def __init__(self, name: str, revision: str):
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            revision=revision,
            trust_remote_code=True,
        )

        # model
        model = OPTForCausalLM.from_pretrained(
            name,
            revision=revision,
            torch_dtype="auto",
            device_map=configs.DEVICE,
        )

        super().__init__(pretrained=model, tokenizer=tokenizer, batch_size=1)
