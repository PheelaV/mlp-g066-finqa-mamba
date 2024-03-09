from transformers import Trainer
from trl import SFTTrainer
import torch
from torch.nn.functional import cross_entropy
from typing import Dict, Sequence
import os
from transformers import DataCollatorForSeq2Seq


def _compute_loss(self, model, inputs, return_outputs=False, prompt_mass_weight=1.0):
    input_ids = inputs.pop("input_ids")
    if "prompt_lens" in inputs:
        prompt_lens = inputs.pop("prompt_lens")
    outputs = model(input_ids)

    lm_logits = outputs.logits
    labels = input_ids.to(lm_logits.device)
    shift_logits = lm_logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()

    lm_loss = cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        labels.view(-1),
        reduction="none",
        ignore_index=self.padding_token_id,
    )

    if prompt_lens is not None:
        mask = torch.zeros_like(labels, dtype=torch.float)
        for i, last_idx in enumerate(prompt_lens):
            mask[i, : last_idx + 1] = 1
        flattened_mask = mask.view(-1)
        weighted_mask = (
            flattened_mask * self.prompt_mass_weight + (1 - flattened_mask)
        ).to(lm_loss.device)
        lm_loss *= weighted_mask

    loss = lm_loss.mean()
    return (loss, outputs) if return_outputs else loss


class MockTrainer(object):
    def __init__(
        self, model, tokenizer, prompt_mass_weight: float = 1.0, padding_token_id=-100
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_mass_weight = prompt_mass_weight
        self.padding_token_id = padding_token_id

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        return _compute_loss(self, model, inputs, return_outputs=return_outputs)


class CustomSFTTrainer(SFTTrainer):
    """
    Custom Self-supervised Fine Tuning Trainer class for CLM with prompt loss
    weight (PLW)

    To compute loss with CrossEntropyLoss it uses PLW to accordingly weight
    the prompt tokens. This is only applied if prompt lengths (prompt_lens) are
    passed in the input dict.

    \sum{L_{CE}(x, y) * (1 - mask) + L_{CE}(x, y) * mask * prompt_mass_weight

    where L_{CE} is the CrossEntropyLoss, x is the input, y is the target, and
    mask is a binary mask that is 1 for prompt tokens and 0 for non-prompt tokens.
    """

    def __init__(
        self, *args, prompt_mass_weight: float = 1.0, padding_token_id=-100, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.prompt_mass_weight = prompt_mass_weight
        self.padding_token_id = padding_token_id

    def compute_loss(self, model, inputs, return_outputs=False):
        return _compute_loss(self, model, inputs, return_outputs=return_outputs)


class CustomTrainer(Trainer):
    """
    Custom Trainer class for CLM with prompt loss weight (PLW)

    To compute loss with CrossEntropyLoss it uses PLW to accordingly weight
    the prompt tokens. This is only applied if prompt lengths (prompt_lens) are
    passed in the input dict.

    \sum{L_{CE}(x, y) * (1 - mask) + L_{CE}(x, y) * mask * prompt_mass_weight

    where L_{CE} is the CrossEntropyLoss, x is the input, y is the target, and
    mask is a binary mask that is 1 for prompt tokens and 0 for non-prompt tokens.
    """

    def __init__(
        self, *args, prompt_mass_weight: float = 1.0, padding_token_id=-100, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.prompt_mass_weight = prompt_mass_weight
        self.padding_token_id = padding_token_id

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        return _compute_loss(self, model, inputs, return_outputs=return_outputs)

    # this is for saving the full model
    def save_model(self, output_dir, _internal_call=False):
        if _internal_call:
            return super().save_model(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)


class CustomDataCollatorSeq2Seq(DataCollatorForSeq2Seq):
    """
    Custom DataCollatorForSeq2Seq class to add prompt_lens to the batch.

    This is then used for prompt mass weight (PMW) in the loss computation.
    """

    def __init__(
        self,
        tokenizer,
        model=None,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        label_pad_token_id=-100,
        return_tensors="pt",
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            return_tensors=return_tensors,
        )

    def __call__(self, features: Sequence[Dict]) -> Dict:
        batch = super().__call__(features, return_tensors="pt")
        # Add prompt_lens if it exists in any of the features.
        if "prompt_lens" in features[0]:
            batch["prompt_lens"] = torch.tensor(
                [feature.get("prompt_lens", 0) for feature in features]
            ).to(batch["input_ids"].device)
        return batch
