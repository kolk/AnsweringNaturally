# -*- coding: utf-8 -*-

import torch
import random
import codecs
from itertools import islice
import numpy as np
#from torch.nn._functions.packing import PackPadded
#from torch.nn.utils.rnn import PackedSequence


def split_corpus(path, shard_size):
    print(path)
    with codecs.open(path, "r", encoding="utf-8") as f:
        if shard_size <= 0:
            yield f.readlines()
        else:
            while True:
                shard = list(islice(f, shard_size))
                if not shard:
                    break
                yield shard


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpu_ranks') and len(opt.gpu_ranks) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)


def set_random_seed(seed, is_cuda):
    """Sets the random seed."""
    if seed > 0:
        torch.manual_seed(seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if is_cuda and seed > 0:
        # These ensure same initialization in multi gpu mode
        torch.cuda.manual_seed(seed)


'''
def pack_padded_sequence_ans(input, lengths, batch_first=False):
    r"""Packs a Tensor containing padded sequences of variable length.

    Input can be of size ``T x B x *`` where `T` is the length of the longest sequence
    (equal to ``lengths[0]``), `B` is the batch size, and `*` is any number of
    dimensions (including 0). If ``batch_first`` is True ``B x T x *`` inputs are
    expected.

    The sequences should be sorted by length in a decreasing order, i.e.
    ``input[:,0]`` should be the longest sequence, and ``input[:,B-1]`` the
    shortest one.

    Note:
        This function accepts any input that has at least two dimensions. You
        can apply it to pack the labels, and use the output of the RNN with
        them to compute the loss directly. A Tensor can be retrieved from
        a :class:`PackedSequence` object by accessing its ``.data`` attribute.

    Arguments:
        input (Tensor): padded batch of variable length sequences.
        lengths (Tensor): list of sequences lengths of each batch element.
        batch_first (bool, optional): if ``True``, the input is expected in ``B x T x *``
            format.

    Returns:
        a :class:`PackedSequence` object
    """

    ## sort the answer vector ############
    sorted_lengths = sorted(lengths, reverse=True)
    #indices = np.argsort(lengths)[::-1]
    indices = torch.argsort(lengths, descending=True)

    if isinstance(sorted_lengths, list):
        sorted_lengths = torch.LongTensor(sorted_lengths)

    data_, batch_sizes_ = PackPadded.apply(input, sorted_lengths, batch_first)
    data = data_.clone()
    batch_sizes = batch_sizes_.clone()
    for i, index in enumerate(indices):
        data[index] = data_[i, :]
        #batch_sizes[index] = batch_sizes_[i]

    return PackedSequence(data, batch_sizes)
'''
