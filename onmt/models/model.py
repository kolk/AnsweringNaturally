""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch
from onmt.utils.logging import logger

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, ans, tgt, lengths, ans_lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            bptt (:obj:`Boolean`):
                a flag indicating if truncated bptt is set. If reset then
                init_state

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state_ques, memory_bank_ques, ques_lengths = self.encoder(src, lengths, "ques")
        enc_state_ans, memory_bank_ans, ans_lengths = self.encoder(ans, ans_lengths, "ans")

        enc_state_final =  tuple(torch.add(enc_q, enc_ans) for enc_q, enc_ans in zip(enc_state_ques, enc_state_ans))
        memory_bank_final = torch.cat([memory_bank_ques, memory_bank_ans], 0)
        memory_lengths_final = torch.add(ques_lengths, ans_lengths)
        
        if bptt is False:
            self.decoder.init_state(src, memory_bank_final, enc_state_final)
        dec_out, attns = self.decoder(tgt, memory_bank_final, memory_lengths=memory_lengths_final)
        return dec_out, attns
