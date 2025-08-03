from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

from ... import configs
from .modeling_hgrn2 import HGRN2ForCausalLM


class HGRN2(HFLM):
    def __init__(self, name: str, revision: str):
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            revision=revision,
            trust_remote_code=True,
        )

        # model
        model = HGRN2ForCausalLM.from_pretrained(
            name,
            revision=revision,
            torch_dtype="auto",
            device_map=configs.DEVICE,
        )

        super().__init__(pretrained=model, tokenizer=tokenizer, batch_size=1)
