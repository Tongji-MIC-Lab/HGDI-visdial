from torch import nn

class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        encoder_output, num_usage, sim_paths, sim_label = self.encoder(batch)
        decoder_output = self.decoder(encoder_output, batch)
        return encoder_output, decoder_output, num_usage, sim_paths, sim_label
