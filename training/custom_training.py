from transformers import Trainer

# # allows for ingestion of pre-tokenized datasets
from custom_sft_trainer import SFTTrainer
import torch
from typing import Optional, List, Tuple, Union, Any
from torch.nn.functional import cross_entropy
from typing import Dict, Sequence
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_pt_utils import nested_detach


def _compute_loss(self, model, inputs, return_outputs=False):
    input_ids = inputs.pop("input_ids")
    prompt_weighted_mask = inputs.pop("prompt_weighted_mask")
    outputs = model(input_ids)

    lm_logits = outputs.logits
    labels = input_ids

    shift_logits = lm_logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()

    lm_loss = cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        labels.view(-1),
        reduction="none",
        ignore_index=self.padding_token_id,
    )

    real_mass = prompt_weighted_mask.sum()
    lm_loss = lm_loss * prompt_weighted_mask.view(-1)
    current_mass = lm_loss.size(0)
    # this preserves correct scale of the loss
    loss = lm_loss.mean() * real_mass / current_mass
    
    del prompt_weighted_mask
    del shift_logits
    del labels
    del current_mass
    del real_mass
    del lm_loss
    
    return (loss, outputs) if return_outputs else loss


# Trainer  will fail on eval step as it tries to check if labels are present and we do not use labels
# 3452         logits and labels (each being optional).
#    3453     """
# -> 3454     has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
#    3455     # For CLIP-like models capable of returning loss values.
#    3456     # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
#    3457     # is `True` in `model.forward`.
#    3458     return_loss = inputs.get("return_loss", None)


# because we are passing around the metadata for the length of the prompt, we need are not going to use labels
# while labels are stil required for the computation somewhere and so this is an effective workaround
def _prediction_step(
    self,
    model: torch.nn.Module,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    prediction_loss_only: bool,
    ignore_keys: Optional[List[str]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"],
            # .to(model.device),
            labels=inputs["input_ids"],
            # .to(model.device),
        )

    logits = nested_detach(output.logits)
    labels = None
    if prediction_loss_only:
        return (output.loss, logits, labels)


# AttributeError: 'NoneType' object has no attribute 'get'
class MockTrainer(object):
    """For testing purposes"""

    def __init__(
        self, model, tokenizer, prompt_loss_weight: float = 1.0, padding_token_id=-100
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_loss_weight = prompt_loss_weight
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

    \sum{L_{CE}(x, y) * (1 - mask) + L_{CE}(x, y) * mask * prompt_loss_weight

    where L_{CE} is the CrossEntropyLoss, x is the input, y is the target, and
    mask is a binary mask that is 1 for prompt tokens and 0 for non-prompt tokens.
    """

    def __init__(
        self,
        *args,
        prompt_loss_weight: float = 1.0,
        padding_token_id=-100,
        **kwargs,
    ):
        # if ""
        super().__init__(*args, **kwargs)
        self.prompt_loss_weight = prompt_loss_weight
        self.padding_token_id = padding_token_id

    def compute_loss(self, model, inputs, return_outputs=False):
        return _compute_loss(self, model, inputs, return_outputs=return_outputs)

    # def prediction_step(self, model: Module, inputs: Dict[str, torch.Tensor | Any], prediction_loss_only: bool, ignore_keys: List[str] | None = None) -> Tuple[torch.Tensor | None]:
    #     # return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    #     return _prediction_step(self, model, inputs, prediction_loss_only, ignore_keys)


class CustomTrainer(Trainer):
    """
    Custom Trainer class for CLM with prompt loss weight (PLW)

    To compute loss with CrossEntropyLoss it uses PLW to accordingly weight
    the prompt tokens. This is only applied if prompt lengths (prompt_lens) are
    passed in the input dict.

    \sum{L_{CE}(x, y) * (1 - mask) + L_{CE}(x, y) * mask * prompt_loss_weight

    where L_{CE} is the CrossEntropyLoss, x is the input, y is the target, and
    mask is a binary mask that is 1 for prompt tokens and 0 for non-prompt tokens.
    """

    def __init__(
        self, *args, prompt_loss_weight: float = 1.0, padding_token_id=-100, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.prompt_loss_weight = prompt_loss_weight
        self.padding_token_id = padding_token_id

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        return _compute_loss(self, model, inputs, return_outputs=return_outputs)

    # def prediction_step(self, model: Module, inputs: Dict[str, torch.Tensor | Any], prediction_loss_only: bool, ignore_keys: List[str] | None = None) -> Tuple[torch.Tensor | None]:
    #     # return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    #     return _prediction_step(self, model, inputs, prediction_loss_only, ignore_keys)

    # this is for saving the full model
    # def save_model(self, working_dir, _internal_call=False):
    #     if _internal_call:
    #         return super().save_model(working_dir)
    #     if not os.path.exists(working_dir):
    #         os.makedirs(working_dir)

    #     torch.save(self.model.state_dict(), f"{working_dir}/pytorch_model.bin")
    #     self.tokenizer.save_pretrained(working_dir)


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
        prompt_loss_weight: float = 0.1,
    ):
        self.prompt_loss_weight = prompt_loss_weight
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
        # turns out this is not necessary if remove_unused_columns is set to False (as unused is determined by the partiuclar model signature) in trainer.py
        batch = super().__call__(features, return_tensors="pt")
        # if self.prompt_loss_weight == 1:
        #     return batch
        batch_size, seq_len = batch["input_ids"].size()
        # here the tensors are still on the CPU
        # `seq_len - 1` because we are using all of this for causal LM and this
        # accounts for the shift # we want the last prompt token to be included
        # in the loss as it corresponds to the  first predicted token of th
        # model.                         \/
        arange_mask = torch.arange(seq_len - 1).expand(batch_size, -1)
        # `batch["prompt_lens"] - 1` at the same time we shorten the patting by one to make the shapes
        # compatible later on.            \/
        mask = (arange_mask < (batch.pop("prompt_lens") - 1).unsqueeze(1)).float()
        batch["prompt_weighted_mask"] = (
            # this is the meask i.e. 0.1, 0.1, 1, 1, 1, 1 for prompt_lens = 2 and seq_len = 
            # 6 because of padding but answer might just be 2 
            mask * self.prompt_loss_weight + (1 - mask)
            # so we need to mask the mask to get
            # 0.1, 0.1, 1, 1, 0, 0 and has the correct weight (used for mean calculation)
        ) * (arange_mask < (batch.pop("input_lens") - 1).unsqueeze(1)).float()
        return batch
