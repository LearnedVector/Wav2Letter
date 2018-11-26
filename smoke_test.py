"""Quick smoke test example using randomly generating data.
    Used to quickly verify if the model can run without errors
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Wav2Letter.model import Wav2Letter
from Wav2Letter.decoder import GreedyDecoder

def smoke_test():
    """Smoke test for training Wav2Letter Model using
        randomly generated data. This is used just to quickly
        verify if the model can run without errors.
        expects the model to perform poorly.
    """
    # 26 letters in the english alphabet + blank token
    grapheme_count = 26 + 1
    in_frame_len = 500  # arbitrary frame length
    sample_size = 50  # arbitrary sample size
    mfcc_features = 13  # 13 mfcc features, discard 13 - 29
    batch_size = 25  # arbitrary batch size
    seq_length = 20  # arbitrary max sequence length

    print("Randomly generating input and output data...")

    # create dummy X inputs data
    inputs = torch.randn(sample_size, in_frame_len, mfcc_features)

    # create dummy Y target data of class labels
    # from 1 - 26 (0 reservered for blank)
    targets = torch.randint(1, grapheme_count, (sample_size, seq_length))

    print("inputs shape", inputs.shape)
    print("target shape", targets.shape)

    model = Wav2Letter(mfcc_features, grapheme_count)
    print(model.layers)

    ctc_loss = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters())

    # Each mfcc feature is a channel
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
    # transpose (sample_size, in_frame_len, mfcc_features)
    # to      (sample_size, mfcc_features, in_frame_len)
    inputs = inputs.transpose(1, 2)

    model.fit(inputs, targets, optimizer, ctc_loss, batch_size, epoch=1, print_every=1)
    log_probs = model.eval(inputs[0])
    out_put = GreedyDecoder(log_probs)

    # print class labels per time step
    print("output labels", out_put)
    # print true labels
    print("true", targets[0])

if __name__ == '__main__':
    smoke_test()
