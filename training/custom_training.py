from torch.nn.modules import Module
from transformers import Trainer, Seq2SeqTrainer
# from trl import SFTTrainer
from custom_sft_trainer import SFTTrainer
import torch
from typing import Optional, List, Tuple, Union, Any
from torch.nn.functional import cross_entropy
from typing import Dict, Sequence
import os
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_pt_utils import nested_detach
# from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat

# from transformers.utils.import_utils import is_sagemaker_mp_enabled

def _compute_loss(self, model, inputs, return_outputs=False):
    input_ids = inputs.pop("input_ids")
    if "prompt_lens" in inputs:
        prompt_lens = inputs.pop("prompt_lens")
    prompt_lens = inputs.get("prompt_lens", None)
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
            flattened_mask * self.prompt_loss_weight + (1 - flattened_mask)
        ).to(lm_loss.device)
        lm_loss *= weighted_mask

    loss = lm_loss.mean()
    return (loss, outputs) if return_outputs else loss

    # if len(logits) == 1:
    #     logits = logits[0]

    # return (loss, logits, labels)
#         outputs = model(**inputs)
#         if hasattr(model, "config") and hasattr(model.config, "use_cache"):
#             use_cache = model.config.use_cache
#         else:
#             use_cache = False

#         if not prediction_loss_only:
#             if use_cache:
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
            #.to(model.device),
            labels=inputs["input_ids"]
            #.to(model.device),
        )
            
    logits = nested_detach(output.logits)
    labels = None
    if prediction_loss_only:
        return (output.loss, logits, labels)


# def _prediction_step(
#     self,
#     model: torch.nn.Module,
#     inputs: Dict[str, Union[torch.Tensor, Any]],
#     prediction_loss_only: bool,
#     ignore_keys: Optional[List[str]] = None,
# ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
#     """
#     Perform an evaluation step on `model` using `inputs`.

#     Subclass and override to inject custom behavior.

#     Args:
#         model (`nn.Module`):
#             The model to evaluate.
#         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
#             The inputs and targets of the model.

#             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
#             argument `labels`. Check your model's documentation for all accepted arguments.
#         prediction_loss_only (`bool`):
#             Whether or not to return the loss only.
#         ignore_keys (`List[str]`, *optional*):
#             A list of keys in the output of your model (if it is a dictionary) that should be ignored when
#             gathering predictions.

#     Return:
#         Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
#         logits and labels (each being optional).
#     """
#     has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
#     # For CLIP-like models capable of returning loss values.
#     # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
#     # is `True` in `model.forward`.
#     return_loss = inputs.get("return_loss", None)
#     if return_loss is None:
#         return_loss = self.can_return_loss
#     loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

#     inputs = self._prepare_inputs(inputs)
#     if ignore_keys is None:
#         if hasattr(self.model, "config"):
#             ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
#         else:
#             ignore_keys = []

#     # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
#     if has_labels or loss_without_labels:
#         labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
#         if len(labels) == 1:
#             labels = labels[0]
#     else:
#         labels = None

#     with torch.no_grad():
#         if is_sagemaker_mp_enabled():
#             raw_outputs = smp_forward_only(model, inputs)
#             if has_labels or loss_without_labels:
#                 if isinstance(raw_outputs, dict):
#                     loss_mb = raw_outputs["loss"]
#                     logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
#                 else:
#                     loss_mb = raw_outputs[0]
#                     logits_mb = raw_outputs[1:]

#                 loss = loss_mb.reduce_mean().detach().cpu()
#                 logits = smp_nested_concat(logits_mb)
#             else:
#                 loss = None
#                 if isinstance(raw_outputs, dict):
#                     logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
#                 else:
#                     logits_mb = raw_outputs
#                 logits = smp_nested_concat(logits_mb)
#         else:
#             if has_labels or loss_without_labels:
#                 with self.compute_loss_context_manager():
#                     loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
#                 loss = loss.mean().detach()

#                 if isinstance(outputs, dict):
#                     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
#                 else:
#                     logits = outputs[1:]
#             else:
#                 loss = None
#                 with self.compute_loss_context_manager():
#                     outputs = model(**inputs)
#                 if isinstance(outputs, dict):
#                     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
#                 else:
#                     logits = outputs
#                 # TODO: this needs to be fixed and made cleaner later.
#                 if self.args.past_index >= 0:
#                     self._past = outputs[self.args.past_index - 1]

#     if prediction_loss_only:
#         return (loss, None, None)

#     logits = nested_detach(logits)
#     if len(logits) == 1:
#         logits = logits[0]

#     return (loss, logits, labels)


# AttributeError: 'NoneType' object has no attribute 'get'
class MockTrainer(object):
    """ For testing purposes """
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
        self, *args, prompt_loss_weight: float = 1.0, padding_token_id=-100, **kwargs
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
    
    def prediction_step(self, model: Module, inputs: Dict[str, torch.Tensor | Any], prediction_loss_only: bool, ignore_keys: List[str] | None = None) -> Tuple[torch.Tensor | None]:
        # return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        return _prediction_step(self, model, inputs, prediction_loss_only, ignore_keys)


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
