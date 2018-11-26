"""Beam Decoder Module.

    TODO: 
        * write beam search decoder
        * use KenLM langauge model to aid decoding
        * add WER and CER metrics
"""
from torch import topk

def GreedyDecoder(ctc_matrix, blank_label=0):
    """Greedy Decoder. Returns highest probability of
        class labels for each timestep

        # TODO: collapse blank labels

    Args:
        ctc_matrix (torch.Tensor): 
            shape (1, num_classes, output_len)
        blank_label (int): blank labels to collapse
    
    Returns:
        torch.Tensor: class labels per time step.
         shape (ctc timesteps)
    """
    top = topk(ctc_matrix, k=1, dim=1)
    return top[1][0][0]
