from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Wav2Letter.decoder import GreedyDecoder


class Wav2Letter(nn.Module):
    """Wav2Letter Speech Recognition model
        Architecture is based off of Facebooks AI Research paper
        https://arxiv.org/pdf/1609.03193.pdf
        This specific architecture accepts mfcc or
        power spectrums speech signals

        TODO: use cuda if available

        Args:
            num_features (int): number of mfcc features
            num_classes (int): number of unique grapheme class labels
    """

    def __init__(self, num_features, num_classes):
        super(Wav2Letter, self).__init__()

        # Conv1d(in_channels, out_channels, kernel_size, stride)
        self.layers = nn.Sequential(
            nn.Conv1d(num_features, 250, 48, 2),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 2000, 32),
            torch.nn.ReLU(),
            nn.Conv1d(2000, 2000, 1),
            torch.nn.ReLU(),
            nn.Conv1d(2000, num_classes, 1),
        )

    def forward(self, batch):
        """Forward pass through Wav2Letter network than 
            takes log probability of output

        Args:
            batch (int): mini batch of data
             shape (batch, num_features, frame_len)

        Returns:
            log_probs (torch.Tensor):
                shape  (batch_size, num_classes, output_len)
        """
        # y_pred shape (batch_size, num_classes, output_len)
        y_pred = self.layers(batch)

        # compute log softmax probability on graphemes
        log_probs = F.log_softmax(y_pred, dim=1)

        return log_probs

    def fit(self, inputs, output, optimizer, ctc_loss, batch_size, epoch, print_every=50):
        """Trains Wav2Letter model.

        Args:
            inputs (torch.Tensor): shape (sample_size, num_features, frame_len)
            output (torch.Tensor): shape (sample_size, seq_len)
            optimizer (nn.optim): pytorch optimizer
            ctc_loss (ctc_loss_fn): ctc loss function
            batch_size (int): size of mini batches
            epoch (int): number of epochs
            print_every (int): every number of steps to print loss
        """

        total_steps = math.ceil(len(inputs) / batch_size)
        seq_length = output.shape[1]

        for t in range(epoch):

            samples_processed = 0
            avg_epoch_loss = 0

            for step in range(total_steps):
                optimizer.zero_grad()
                batch = \
                    inputs[samples_processed:batch_size + samples_processed]

                # log_probs shape (batch_size, num_classes, output_len)
                log_probs = self.forward(batch)

                # CTC_Loss expects input shape
                # (input_length, batch_size, num_classes)
                log_probs = log_probs.transpose(1, 2).transpose(0, 1)

                # CTC arguments
                # https://pytorch.org/docs/master/nn.html#torch.nn.CTCLoss
                # better definitions for ctc arguments
                # https://discuss.pytorch.org/t/ctcloss-with-warp-ctc-help/8788/3
                mini_batch_size = len(batch)
                targets = output[samples_processed: mini_batch_size + samples_processed]

                input_lengths = torch.full((mini_batch_size,), log_probs.shape[0], dtype=torch.long)
                target_lengths = torch.IntTensor([target.shape[0] for target in targets])

                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

                avg_epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                samples_processed += mini_batch_size

                if step % print_every == 0:
                    print("epoch", t + 1, ":" , "step", step + 1, "/", total_steps, ", loss ", loss.item())

            print("epoch", t + 1, "average epoch loss", avg_epoch_loss / total_steps)

    def eval(self, sample):
        """Evaluate model given a single sample

        Args:
            sample (torch.Tensor): shape (n_features, frame_len)

        Returns:
            log probabilities (torch.Tensor):
                shape (n_features, output_len)
        """
        _input = sample.reshape(1, sample.shape[0], sample.shape[1])
        log_prob = self.forward(_input)
        return log_prob
