import math
from dataclasses import dataclass, field


import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)

def get_regularization_loss(regularization):
    if regularization == "l1":
        return lambda x: torch.sum(torch.abs(x))
    elif regularization == "l2":
        return lambda x: torch.sum(x ** 2)
    else:
        raise ValueError(f"Unknown regularization type: {regularization}")

@dataclass
class LabelSmoothedCrossEntropySparseActivations(
    LabelSmoothedCrossEntropyCriterionConfig
):
    sparsity_weight: float = field(default=1e-7, metadata={"help": "weight for CTC loss"})
    regularization: str = field(default="l1", metadata={"help": "regularization type"})

@register_criterion(
    "label_smoothed_cross_entropy_sparse_activations",
    dataclass=LabelSmoothedCrossEntropySparseActivations,
)
class LabelSmoothedCrossEntropySparseActivationsCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size,
        report_accuracy,
        sparsity_weight,
        regularization,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.sparsity_weight = sparsity_weight
        self.regularization = get_regularization_loss(regularization)

    def forward(self, model, sample, reduce=True):
        hooks = {}
        activations = [] 
        def hook_fn(module, input, output):
            activations.append(output)

        for name, module in model.named_modules():
            if 'sparsing_fn' in name:
                hooks[name] = module.register_forward_hook(hook_fn)

        net_output = model(**sample["net_input"])
        
        for hook in hooks.values():
            hook.remove()

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        
        act_reg = torch.tensor(0.0).type_as(loss)
        for activation in activations:
            act_reg += self.regularization(activation.type_as(loss)) * self.sparsity_weight
        loss += act_reg

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "activations_regularization": act_reg.data, # TODO: add to logging
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if not model.training:
            enc_sparsity = model.encoder.sparse_stats
            for module in enc_sparsity.keys():
                logging_output[f'sparsity_{module}'] = enc_sparsity[module]

            dec_sparsity = model.decoder.sparse_stats
            for module in dec_sparsity.keys():
                logging_output[f'sparsity_{module}'] = dec_sparsity[module]

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        loss_sum = sum(log.get("activations_regularization", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "activations_regularization", loss_sum / sample_size / math.log(2), sample_size, round=6
        )
        

