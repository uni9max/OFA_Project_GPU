# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from argparse import Namespace
from itertools import zip_longest
from collections import OrderedDict

import numpy as np
import sacrebleu
import string
from fairseq import metrics, utils
from fairseq.tasks import register_task
from utils.cider.pyciderevalcap.spice.spice import Spice

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.caption_dataset import CaptionDataset
from data.file_dataset import FileDataset
from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


@dataclass
class CaptionConfig(OFAConfig):
    eval_bleu: bool = field(default=False, metadata={"help": "evaluation with BLEU scores"})
    eval_cider: bool = field(default=False, metadata={"help": "evaluation with CIDEr scores"})
    eval_args: Optional[str] = field(
        default='{}',
        metadata={"help": 'generation args for BLUE or CIDEr scoring, e.g., "{\"beam\": 4, \"lenpen\": 0.6}", as JSON string'}
    )
    eval_print_samples: bool = field(default=False, metadata={"help": "print sample generations during validation"})
    eval_cider_cached_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "path to cached cPickle file used to calculate CIDEr scores"}
    )
    eval_spice: bool = field(default=False, metadata={"help": "evaluation with SPICE scores"})
    scst: bool = field(default=False, metadata={"help": "Self-critical sequence training"})
    scst_args: str = field(default='{}', metadata={"help": 'generation args for Self-critical sequence training, as JSON string'})


@register_task("caption", dataclass=CaptionConfig)
class CaptionTask(OFATask):
    def __init__(self, cfg: CaptionConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        file_path = paths[(epoch - 1) % (len(paths) - 1)] if split == 'train' else paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = CaptionDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            scst=getattr(self.cfg, 'scst', False)
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_bleu or self.cfg.eval_cider:
            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
            if self.cfg.eval_cider:
                self.CiderD_scorer = None
                if self.cfg.eval_cider_cached_tokens is not None:
                    self.CiderD_scorer = CiderD(df=self.cfg.eval_cider_cached_tokens)
                else:
                    logger.warning("eval_cider is True but eval_cider_cached_tokens is None. Skipping CIDEr scorer initialization.")
        if self.cfg.eval_spice:
            self.Spice_scorer = Spice()

        if self.cfg.scst:
            scst_args = json.loads(self.cfg.scst_args)
            self.scst_generator = self.build_generator([model], Namespace(**scst_args))

        return model

    def _calculate_cider_scores(self, gen_res, gt_res):
        res = OrderedDict((i, [gen_res[i].strip()]) for i in range(len(gen_res)))
        gts = OrderedDict((i, [gt.strip() for gt in gt_res[i]]) for i in range(len(gen_res)))
        res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
        _, scores = self.CiderD_scorer.compute_score(gts, res_)
        return scores

    def _calculate_spice_scores(self, gen_res, gt_res):
        res = OrderedDict((i, [gen_res[i].strip()]) for i in range(len(gen_res)))
        gts = OrderedDict((i, [gt.strip() for gt in gt_res[i]]) for i in range(len(gen_res)))
        res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
        _, scores = self.Spice_scorer.compute_score(gts, res_)
        return scores

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample)
        model.eval()

        if self.cfg.eval_bleu or self.cfg.eval_cider:
            hyps, refs = self._inference(self.sequence_generator, sample, model)
            if self.cfg.eval_bleu:
                bleu = sacrebleu.corpus_bleu(hyps, list(zip_longest(*refs)), tokenize="none") if self.cfg.eval_tokenized_bleu else sacrebleu.corpus_bleu(hyps, list(zip_longest(*refs)))
                logging_output.update({
                    "_bleu_sys_len": bleu.sys_len,
                    "_bleu_ref_len": bleu.ref_len,
                    **{f"_bleu_counts_{i}": bleu.counts[i] for i in range(EVAL_BLEU_ORDER)},
                    **{f"_bleu_totals_{i}": bleu.totals[i] for i in range(EVAL_BLEU_ORDER)}
                })
            if self.cfg.eval_spice:
                scores = self._calculate_spice_scores(hyps, refs)
                logging_output["_spice_score_sum"] = scores.sum()
                logging_output["_spice_cnt"] = scores.size
            if self.cfg.eval_cider and self.CiderD_scorer is not None:
                scores = self._calculate_cider_scores(hyps, refs)
                logging_output["_cider_score_sum"] = scores.sum()
                logging_output["_cider_cnt"] = scores.size

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            return result.cpu() if torch.is_tensor(result) else result

        if self.cfg.eval_spice:
            if sum_logs("_spice_cnt") > 0:
                metrics.log_scalar("_spice_score_sum", sum_logs("_spice_score_sum"))
                metrics.log_scalar("_spice_cnt", sum_logs("_spice_cnt"))
                metrics.log_derived("spice", lambda meters: round(meters["_spice_score_sum"].sum / meters["_spice_cnt"].sum, 3))

        if self.cfg.eval_bleu:
            counts = [sum_logs(f"_bleu_counts_{i}") for i in range(EVAL_BLEU_ORDER)]
            totals = [sum_logs(f"_bleu_totals_{i}") for i in range(EVAL_BLEU_ORDER)]
            if max(totals) > 0:
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    smooth = {"smooth_method": "exp"} if "smooth_method" in fn_sig else {"smooth": "exp"}
                    return round(sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    ).score, 2)

                metrics.log_derived("bleu", compute_bleu)

        if self.cfg.eval_cider:
            if sum_logs("_cider_cnt") > 0:
                metrics.log_scalar("_cider_score_sum", sum_logs("_cider_score_sum"))
                metrics.log_scalar("_cider_cnt", sum_logs("_cider_cnt"))
                metrics.log_derived("cider", lambda meters: round((
                    meters["_cider_score_sum"].sum / meters["_cider_cnt"].sum if not isinstance(
                        meters["_cider_score_sum"].sum, float
                    ) else meters["_cider_score_sum"].sum), 3))

    def _inference(self, generator, sample, model):
        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(toks.int().cpu(), unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"))
            return self.bpe.decode(s) if self.bpe else s

        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        transtab = str.maketrans({key: None for key in string.punctuation})
        for i in range(len(gen_out)):
            decode_tokens = decode(gen_out[i][0]["tokens"])
            hyps.append(decode_tokens.translate(transtab).strip())
            refs.append([
                sent.translate(transtab).strip() for sent in decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()), escape_unk=True
                ).split('&&')
            ])
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + ' && '.join(refs[0]))
        return hyps, refs
