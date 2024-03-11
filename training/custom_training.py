from torch.nn.modules import Module
from transformers import Trainer, Seq2SeqTrainer
# from trl import SFTTrainer
from custom_sft_trainer import SFTTrainer
import torch
from typing import Optional, List, Tuple, Union, Any
from torch.nn.functional import cross_entropy
from typing import Dict, Sequence
import os
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers.trainer_pt_utils import nested_detach
# from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat

# from transformers.utils.import_utils import is_sagemaker_mp_enabled

def _compute_loss(self, model, inputs, return_outputs=False):
    input_ids = inputs.pop("input_ids")
    prompt_lens = inputs.get("prompt_lens", None)
    # print(inputs.get("prompt_lens", None))
    # prompt_lens = torch.ones(input_ids.size(0), dtype=torch.int) * int(input_ids.size(1) * 0.8)
    outputs = model(input_ids)
    
    lm_logits = outputs.logits
    labels = input_ids

    shift_logits = lm_logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    
    # loss = cross_entropy(
    #     shift_logits.view(-1, shift_logits.size(-1)),
    #     labels.view(-1),
    #     # reduction="none",
    #     ignore_index=self.padding_token_id,
    # )
    # return (loss, outputs) if return_outputs else loss

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
        )
        lm_loss *= weighted_mask
        
        del mask
        del flattened_mask
        del weighted_mask
        del shift_logits
        del labels

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
    
    # def prediction_step(self, model: Module, inputs: Dict[str, torch.Tensor | Any], prediction_loss_only: bool, ignore_keys: List[str] | None = None) -> Tuple[torch.Tensor | None]:
    #     # return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    #     return _prediction_step(self, model, inputs, prediction_loss_only, ignore_keys)


    # this is for saving the full model
    def save_model(self, output_dir, _internal_call=False):
        if _internal_call:
            return super().save_model(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)


# class CustomDataCollatorSeq2Seq(DataCollatorForLanguageModeling):
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
        # super().__init__(
        #     tokenizer=tokenizer,
        #     mlm=False,
        #     pad_to_multiple_of=pad_to_multiple_of,
        #     return_tensors=return_tensors,
        # )
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
        # Add prompt_lens if it exists in any of the features.
        # print("*" * 100)
        # print(type(features[0]))
        # print(type(batch))
        # print(features[0].keys())
        # print(batch.keys())
        # if "prompt_lens" in features[0]:
        #     batch["prompt_lens"] = torch.tensor(
        #         [feature.get("prompt_lens", 0) for feature in features]
        #     )
        # print(batch.keys())
        # print(batch)
        # exit()
        return batch
