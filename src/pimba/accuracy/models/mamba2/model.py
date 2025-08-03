import torch
from huggingface_hub import snapshot_download
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

from ... import configs
from .models.mixer_seq_simple import MambaLMHeadModel


class Mamba2(HFLM):
    def __init__(self, name: str, revision: str):
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            # NOTE: Mamba2 repo does not contain tokenizer.
            # It uses gpt-neox's tokenizer, thus, we will use that.
            "EleutherAI/gpt-neox-20b",
            trust_remote_code=True,
        )

        # download model
        snapshot_path = snapshot_download(
            repo_id=name,
            revision=revision,
        )

        # model
        model = MambaLMHeadModel.from_pretrained(
            snapshot_path,
            dtype=torch.float16,
            device=configs.DEVICE,
        )
        model.device = configs.DEVICE

        super().__init__(
            pretrained=model, tokenizer=tokenizer, batch_size=1, backend="causal"
        )

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        for key in ("do_sample", "attention_mask"):
            if key in generation_kwargs:
                generation_kwargs.pop(key)

        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            **generation_kwargs,
        )
