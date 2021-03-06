""" Translation main class """
from __future__ import unicode_literals, print_function

import torch
from onmt.inputters.text_dataset import TextDataset


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (DataSet):
       fields (dict of Fields): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, data, fields, n_best=1, replace_unk=False,
                 has_tgt=False):
        self.data = data
        self.fields = fields
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.has_tgt = has_tgt

    def _build_target_tokens(self, src, src_vocab, src_raw, ans_raw, pred, attn):
        tgt_field = self.fields["tgt"][0][1].base_field
        vocab = tgt_field.vocab
        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                #assert src_vocab.itos[tok - len(vocab)] == ans_vocab.itos[tok - len(vocab)]
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == tgt_field.eos_token:
                tokens = tokens[:-1]
                break
        if self.replace_unk and attn is not None and src is not None:
            for i in range(len(tokens)):
                if tokens[i] == tgt_field.unk_token:
                    _, max_index = attn[i].max(0)
                    raw = src_raw + ans_raw # append raw words as attention = [source attention; ans attention]
                    tokens[i] = raw[max_index.item()]
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, attn, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch.indices)
        if isinstance(self.data, TextDataset):
            src = batch.src[0][:, :, 0].index_select(1, perm)
            ans = batch.ans[0][:, :, 0].index_select(1, perm)
        else:
            src = None
            ans = None
       
        tgt = batch.tgt[:, :, 0].index_select(1, perm) \
            if self.has_tgt else None

        translations = []
        for b in range(batch_size):
            if isinstance(self.data, TextDataset):
                src_vocab = self.data.src_vocabs[inds[b]] \
                    if self.data.src_vocabs else None
                src_raw = self.data.examples[inds[b]].src[0]
                #ans_vocab = self.data.ans_vocabs[inds[b]] \
                #    if self.data.ans_vocabs else None
                ans_raw = self.data.examples[inds[b]].ans[0]
            else:
                src_vocab = None
                src_raw = None
                #ans_vocab = None
                ans_raw = None
            pred_sents = [self._build_target_tokens(
                src[:, b] if src is not None else None,
                src_vocab, src_raw,
                ans_raw,
                preds[b][n], attn[b][n])
                for n in range(self.n_best)]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab, src_raw,
                    ans_raw,
                    tgt[1:, b] if tgt is not None else None, None)

            translation = Translation(
                src[:, b] if src is not None else None,
                ans[:, b] if ans is not None else None,
                src_raw, ans_raw, pred_sents, attn[b], pred_score[b],
                gold_sent, gold_score[b]
            )
            translations.append(translation)

        return translations


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, src, ans, src_raw, ans_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.src = src
        self.src_raw = src_raw
        self.ans = ans
        self.ans_raw = ans_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nQUES {}: {}\n ANS {}: {}\n'.format(sent_number, self.src_raw, sent_number, self.ans_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
