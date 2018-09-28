from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import numpy as np
import torch
import torch.nn as nn

import onmt
import onmt.modules


class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        # print(self.loss)
        # print(self.n_words)
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start, ix=False):
        t = self.elapsed_time()
        if ix:
            print(("M%2d Ep. %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
                   "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
                  (ix, epoch, batch, n_batches,
                   self.accuracy(),
                   self.ppl(),
                   self.n_src_words / (t + 1e-5),
                   self.n_words / (t + 1e-5),
                   time.time() - start))
        else:
            print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
                   "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
                  (epoch, batch,  n_batches,
                   self.accuracy(),
                   self.ppl(),
                   self.n_src_words / (t + 1e-5),
                   self.n_words / (t + 1e-5),
                   time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


class Trainer(object):
    def __init__(self, model, train_iter, valid_iter,
                 train_loss, valid_loss, optim,
                 trunc_size, shard_size,
                 ensemble=False, ensemble_num=2,
                 pretrain_for=2):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            valid_iter: the validate data iterator.
            train_loss: the train side LossCompute object for computing loss.
            valid_loss: the valid side LossCompute object for computing loss.
            optim: the optimizer responsible for lr update.
            trunc_size: a batch is divided by several truncs of this size.
            shard_size: compute loss in shards of this size for efficiency.
        """
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size

        self.ensemble = ensemble
        self.ensemble_num = ensemble_num
        self.pretrain_for = pretrain_for

        # Set model in training mode.
        self.model.train()

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        if self.ensemble:
            total_stats = [Statistics() for i in range(self.ensemble_num)]
            report_stats = [Statistics() for i in range(self.ensemble_num)]
            self.total_counts = {i: 0 for i in range(self.ensemble_num)}
            if epoch == self.pretrain_for + 1:
                print("STARTING MCL LOSS...")
            if epoch > self.pretrain_for:
                self.train_loss.use_mask = True
        else:
            total_stats = Statistics()
            report_stats = Statistics()

        for i, batch in enumerate(self.train_iter):
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            trunc_size = self.trunc_size if self.trunc_size else target_size

            dec_state = None
            _, src_lengths = batch.src

            src = onmt.IO.make_features(batch, 'src')
            tgt_outer = onmt.IO.make_features(batch, 'tgt')
            if self.ensemble:
                for r in report_stats:
                    r.n_src_words += src_lengths.sum()
            else:
                report_stats.n_src_words += src_lengths.sum()

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                self.model.zero_grad()
                outputs, attns, dec_state = \
                    self.model(src, tgt, src_lengths, dec_state)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size)

                # 4. Update the parameters and statistics.
                self.optim.step()
                if self.ensemble:
                    indices = batch_stats[1]
                    for ix in range(self.ensemble_num):
                        self.total_counts[ix] += indices.eq(ix).sum().data[0]
                    batch_stats = batch_stats[0]
                    for ix, s in enumerate(batch_stats):
                        total_stats[ix].update(s)
                        report_stats[ix].update(s)
                        if report_func is not None:
                            report_stats[ix] = report_func(
                                    epoch, i, len(self.train_iter),
                                    total_stats[0].start_time, self.optim.lr,
                                    report_stats[ix], ix+1)
                else:
                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)
                    if report_func is not None:
                        report_stats = report_func(
                                epoch, i, len(self.train_iter),
                                total_stats.start_time, self.optim.lr,
                                report_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    if self.ensemble:
                        for d in dec_state:
                            d.detach()
                    else:
                        dec_state.detach()
            # if i > 20:
            #     break
        return total_stats

    def validate(self):
        """ Called for each epoch to validate. """
        # Set model in validating mode.
        self.model.eval()
        if self.ensemble:
            stats = [Statistics() for i in range(self.ensemble_num)]
        else:
            stats = Statistics()

        for batch in self.valid_iter:
            _, src_lengths = batch.src
            src = onmt.IO.make_features(batch, 'src')
            tgt = onmt.IO.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            if self.ensemble:
                for ix, s in enumerate(batch_stats):
                    stats[ix].update(s)
            else:
                stats.update(batch_stats)
            # break

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        """ Called for each epoch to update learning rate. """
        return self.optim.updateLearningRate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Called conditionally each epoch to save a snapshot. """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()}
        checkpoint = {
            'model': model_state_dict,
            'vocab': onmt.IO.ONMTDataset.save_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim
        }
        if self.ensemble:
            valid_ppl = "_".join(["%.2f" % s.ppl() for s in valid_stats])
            valid_acc = "_".join(["%.2f" % s.accuracy() for s in valid_stats])
        else:
            valid_ppl = "%.2f" % valid_stats.ppl()
            valid_acc = "%.2f" % valid_stats.accuracy()
        torch.save(checkpoint,
                   '%s_acc_%s_ppl_%s_e%d.pt'
                   % (opt.save_model, valid_acc,
                      valid_ppl, epoch))
