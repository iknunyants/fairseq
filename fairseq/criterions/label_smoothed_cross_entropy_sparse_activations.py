import math
from dataclasses import dataclass, field
from typing import Optional

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
    elif regularization == "hoyer":
        return lambda x: torch.sum(torch.abs(x)) ** 2 / torch.sum(x ** 2)
    else:
        raise ValueError(f"Unknown regularization type: {regularization}")

def get_threshold_pen_func(threshold_pen_func):
    if threshold_pen_func == "inv_sqr":
        return lambda x: torch.sum((1 / x) ** 2)
    elif threshold_pen_func == "neg_log":
        return lambda x: -torch.sum(torch.log(x + 1e-3))
    elif threshold_pen_func == "neg_l1":
        return lambda x: -torch.sum(torch.abs(x))
    else:
        raise ValueError(f"Unknown threshold penalty function: {threshold_pen_func}")
@dataclass
class LabelSmoothedCrossEntropySparseActivationsCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    sparsity_weight: float = field(default=1e-7, metadata={"help": "weight for activation regularization loss"})
    regularization: str = field(default="l1", metadata={"help": "regularization type"})
    sparsing_fn: str = field(default="relu", metadata={"help": "sparsifying function"})
    fatrelu_reg: str = field(default="l1thres", metadata={"help": "regularization type for fatrelu"})
    clamp_thresholds: bool = field(default=False, metadata={"help": "clamp negative thresholds of fatrelu to 0.0"})
    threshold_pen_func: Optional[str] = field(default=None, metadata={"help": "function for threshold penalty"})

@register_criterion(
    "label_smoothed_cross_entropy_sparse_activations",
    dataclass=LabelSmoothedCrossEntropySparseActivationsCriterionConfig,
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
        sparsing_fn, 
        fatrelu_reg, 
        threshold_pen_func
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.sparsity_weight = sparsity_weight
        self.regularization_name = regularization
        self.regularization = get_regularization_loss(regularization)
        self.sparsing_fn = sparsing_fn
        self.fatrelu_reg = fatrelu_reg
        if self.sparsing_fn == 'fatrelu' and self.fatrelu_reg == 'l1thres':
            self.threshold_pen_func = get_threshold_pen_func(threshold_pen_func)
        
    def forward(self, model, sample, reduce=True):
        hooks = {}
        activations = [] 
        def hook_fn(module, input, output):
            activations.append(output)

        for name, module in model.named_modules():
            name_split = name.split('.')
            if ('sparsing_fn' in name_split[-1] and (self.fatrelu_reg != 'l1thres' or self.sparsing_fn != 'fatrelu')) or \
                ('relu' in name_split[-1] and self.fatrelu_reg == 'l1thres' and self.sparsing_fn == 'fatrelu'):
                hooks[name] = module.register_forward_hook(hook_fn)

        net_output = model(**sample["net_input"])
        
        for hook in hooks.values():
            hook.remove()

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        act_reg = torch.tensor(0.0).type_as(loss)
        for activation in activations:
            act_reg += self.regularization(activation.type_as(loss))

        logging_output = {'activations_regularization': act_reg.data.clone()}

        if self.sparsing_fn == 'fatrelu':
            if self.fatrelu_reg == 'l1thres':
                threshold_pen = torch.tensor(0.0).type_as(loss)
                for name, module in model.named_modules():
                    if 'FATReLU' in module.__class__.__name__:
                        threshold_pen += self.threshold_pen_func(module.threshold.type_as(loss))

                logging_output['threshold_pen'] = threshold_pen.data
                act_reg += threshold_pen 
            
            # add threshold values TODO: add to logging_output
            for name, module in model.named_modules():
                if 'FATReLU' in module.__class__.__name__:
                    name_split = name.split('.')
                    folder = name_split[0] + '_thresholds/' 
                    if len(name_split) > 2:
                        plot_name = 'layer_' + name_split[2]
                        if 'fc' in name_split[3]:
                            plot_name += '_fc'
                        elif 'proj' in name_split[3]: 
                            plot_name += '_projection'
                        else:
                            plot_name += '_cross'
                    else:
                        plot_name = 'out_projection'
                    logging_output[folder + plot_name] = module.threshold.data.mean()


        act_reg *= self.sparsity_weight

        loss += act_reg

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output.update({
            "loss": loss.data,
            "full_regularization": act_reg.data, 
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        })
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
        full_act_reg = sum(log.get("full_regularization", 0) for log in logging_outputs)
        act_reg = sum(log.get("activations_regularization", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        if 'threshold_pen' in logging_outputs[0]:
            threshold_pen_sum = sum(log.get("threshold_pen", 0) for log in logging_outputs)
            metrics.log_scalar(
                "threshold_pen", threshold_pen_sum / sample_size, sample_size, round=6
            )
        
        metrics.log_scalar( 
            "activations_regularization", act_reg / sample_size, sample_size, round=6
        )

        metrics.log_scalar(
            "full_regularization", full_act_reg / sample_size, sample_size, round=6
        )

        for threshold_type in [s for s in logging_outputs[0] if 'thresholds' in s]:
            threshold_sum = sum(log.get(threshold_type, 0) for log in logging_outputs)
            metrics.log_scalar(threshold_type, threshold_sum / len(logging_outputs), round=6)
