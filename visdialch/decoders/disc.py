import torch
from torch import nn
from visdialch.utils import DynamicRNN


class DiscriminativeDecoder(nn.Module):
    def __init__(self, __C, vocabulary):
        super().__init__()
        self.__C = __C

        self.word_embed = nn.Embedding(
            len(vocabulary),
            __C["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.option_rnn = nn.LSTM(
            __C["word_embedding_size"],
            __C["hidden_size"],
            __C["lstm_num_layers"],
            batch_first=True,
        )

        self.option_rnn = DynamicRNN(self.option_rnn)   

    def forward(self, encoder_output, batch):

        options = batch["opt"]
        batch_size, num_rounds, num_options, max_sequence_length = (
            options.size()
        )
        options = options.view(
            batch_size * num_rounds * num_options, max_sequence_length
        )
        options_length = batch["opt_len"]
        options_length = options_length.view(
            batch_size * num_rounds * num_options
        )

        nonzero_options_length_indices = options_length.nonzero().squeeze()
        nonzero_options_length = options_length[nonzero_options_length_indices]
        nonzero_options = options[nonzero_options_length_indices]

        nonzero_options_embed = self.word_embed(nonzero_options)

        _, (nonzero_options_embed, _) = self.option_rnn(
            nonzero_options_embed, nonzero_options_length
        )

        options_embed = torch.zeros(
            batch_size * num_rounds * num_options,
            nonzero_options_embed.size(-1),
            device=nonzero_options_embed.device,
        )
        options_embed[nonzero_options_length_indices] = nonzero_options_embed

        encoder_output = encoder_output.unsqueeze(2).repeat(
            1, 1, num_options, 1
        )

        encoder_output = encoder_output.view(
            batch_size * num_rounds * num_options,
            self.__C["hidden_size"],
        )
        scores = torch.sum(options_embed * encoder_output, 1)
        scores = scores.view(batch_size, num_rounds, num_options)
        return scores
