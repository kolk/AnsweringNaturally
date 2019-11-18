"""  Attention and normalization modules  """
from onmt.modules.util_class import Elementwise
from onmt.modules.gate import context_gate_factory, ContextGate
from onmt.modules.global_attention import GlobalAttention
from onmt.modules.copy_generator import CopyGenerator, CopyGeneratorLoss, \
    CopyGeneratorLossCompute
from onmt.modules.embeddings import Embeddings, PositionalEncoding
from onmt.modules.weight_norm import WeightNormConv2d

__all__ = ["Elementwise", "context_gate_factory", "ContextGate",
           "GlobalAttention", "CopyGenerator", "CopyGeneratorLoss", "CopyGeneratorLossCompute",
           "Embeddings", "PositionalEncoding", "WeightNormConv2d"]
