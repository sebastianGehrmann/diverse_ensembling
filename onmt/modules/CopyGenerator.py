import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable


import onmt
from onmt.Utils import aeq


class CopyGenerator(nn.Module):
    """
    Generator module that additionally considers copying
    words directly from the source.
    """
    def __init__(self, opt, src_dict, tgt_dict):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(opt.rnn_size, len(tgt_dict))
        self.linear_copy = nn.Linear(opt.rnn_size, 1)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    def forward(self, hidden, attn, src_map):
        """
        Computes p(w) = p(z=1) p_{copy}(w|z=0)  +  p(z=0) * p_{softmax}(w|z=0)
        """
        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.tgt_dict.stoi[onmt.IO.PAD_WORD]] = -float('inf')
        prob = F.softmax(logits)

        # Probability of copying p(z=1) batch.
        copy = F.sigmoid(self.linear_copy(hidden))

        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob,  1 - copy.expand_as(prob))
        mul_attn = torch.mul(attn, copy.expand_as(attn))
        copy_prob = torch.bmm(mul_attn.view(-1, batch, slen)
                              .transpose(0, 1),
                              src_map.transpose(0, 1)).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


class CopyGeneratorCriterion(object):
    def __init__(self, vocab_size, force_copy, pad, eps=1e-20, summed=True):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size
        self.pad = pad
        self.summed = summed

    def __call__(self, scores, align, target):
        align = align.view(-1)

        # Copy prob.
        out = scores.gather(1, align.view(-1, 1) + self.offset) \
                    .view(-1).mul(align.ne(0).float())
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            out = out + self.eps + tmp.mul(target.ne(0).float()) + \
                tmp.mul(align.eq(0).float()).mul(target.eq(0).float())
        else:
            # Forced copy.
            out = out + self.eps + tmp.mul(align.eq(0).float())

        # Drop padding.
        loss = -out.log().mul(target.ne(self.pad).float())
        if self.summed:
            loss = loss.sum()
        return loss


class CopyGeneratorLossCompute(onmt.Loss.LossComputeBase):
    """
    Copy Generator Loss Computation.
    """
    def __init__(self, generator, tgt_vocab, dataset,
                 force_copy, eps=1e-20):
        super(CopyGeneratorLossCompute, self).__init__(generator, tgt_vocab)

        self.dataset = dataset
        self.force_copy = force_copy
        self.criterion = CopyGeneratorCriterion(len(tgt_vocab), force_copy,
                                                self.padding_idx)

    def make_shard_state(self, batch, output, range_, attns):
        """ See base class for args description. """
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]]
        }

    def compute_loss(self, batch, output, target, copy_attn, align):
        """
        Compute the loss. The args must match self.make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(self.bottle(output),
                                self.bottle(copy_attn),
                                batch.src_map)

        loss = self.criterion(scores, align, target)

        scores_data = scores.data.clone()
        scores_data = self.dataset.collapse_copy_scores(
                self.unbottle(scores_data, batch.batch_size),
                batch, self.tgt_vocab)
        scores_data = self.bottle(scores_data)

        # Correct target is copy when only option.
        # TODO: replace for loop with masking or boolean indexing
        target_data = target.data.clone()
        for i in range(target_data.size(0)):
            if target_data[i] == 0 and align.data[i] != 0:
                target_data[i] = align.data[i] + len(self.tgt_vocab)

        # Coverage loss term.
        loss_data = loss.data.clone()

        stats = self.stats(loss_data, scores_data, target_data)

        return loss, stats


class MCLCopyGeneratorLossCompute(onmt.Loss.LossComputeBase):
    """
    Copy Generator Loss Computation.
    """
    def __init__(self, generator,
                 tgt_vocab, dataset,
                 force_copy, mcl_k=1,
                 ensemble_num=2,
                 teacher_model=False,
                 em_type='hard',
                 eps=1e-20):
        super(MCLCopyGeneratorLossCompute, self).__init__(generator, tgt_vocab)

        self.dataset = dataset
        self.force_copy = force_copy
        self.criterion = CopyGeneratorCriterion(len(tgt_vocab), force_copy,
                                                self.padding_idx, summed=False)

        self.ensemble_num = ensemble_num
        self.k = mcl_k

        self.use_mask = False
        self.teacher_model = teacher_model
        self.em_type = em_type

    def compute_loss(self, batch, output, target, copy_attn, align):
        """
        Compute the loss. The args must match self.make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        losses = []
        all_stats = []
        target = target.view(-1)
        align = align.view(-1)

        # Correct target is copy when only option.
        target_data = target.data.clone()
        for i in range(target_data.size(0)):
            if target_data[i] == 0 and align.data[i] != 0:
                target_data[i] = align.data[i] + len(self.tgt_vocab)

        for ix in range(self.ensemble_num):
            scores = self.generator.models[ix].generator(
                self.bottle(output[ix]),
                self.bottle(copy_attn[ix].get("copy")),
                batch.src_map)
            loss = self.criterion(scores, align, target)

            loss_data = loss.sum().data.clone()

            # reformat to get batches back
            logpy = loss.view(-1, batch.batch_size)
            logpy = logpy.sum(0).squeeze().unsqueeze(1)
            losses.append(logpy)

            scores_data = scores.data.clone()
            scores_data = self.dataset.collapse_copy_scores(
                self.unbottle(scores_data, batch.batch_size),
                batch, self.tgt_vocab)
            scores_data = self.bottle(scores_data)

            # Coverage loss term.
            all_stats.append(self.stats(loss_data, scores_data, target_data))
        losses = torch.cat(losses, 1)
        topk, indices = torch.topk(losses, self.k, dim=1, largest=False)
        topk.data.fill_(1)
        if self.use_mask:
            if self.em_type == 'hard':
                mask = torch.zeros(losses.size()).cuda()
                mask.scatter_(1, indices.data, 1.)
                if self.teacher_model:
                    mask[:, 0].fill_(1)
                mask = Variable(mask)
                losses = losses * mask
            if self.em_type == 'soft':
                pxz = losses.clone()
                lossum = losses.sum(dim=1).unsqueeze(1).expand_as(pxz)
                pzx = pxz.div(lossum)
                losses = losses * pzx
        loss = losses.sum()

        return loss, all_stats, indices

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size):
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        # Not implemented as sharded!
        range_ = (0, batch.tgt.size(0))
        loss, stats, indices = self.compute_loss(
            batch,
            output,
            batch.tgt[range_[0] + 1: range_[1]],
            attns,
            batch.alignment[range_[0] + 1: range_[1]])
        loss.div(batch.batch_size).backward()
        return (stats, indices)

    def monolithic_compute_loss(self, batch, output, attns):
        range_ = (0, batch.tgt.size(0))
        loss, stats, indices = self.compute_loss(
            batch,
            output,
            batch.tgt[range_[0] + 1: range_[1]],
            attns,
            batch.alignment[range_[0] + 1: range_[1]])
        return stats

