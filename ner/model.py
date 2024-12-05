import warnings
import torch
import torch.nn as nn
from TorchCRF import CRF

class LSTM_CRF_Model(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 lstm_hidden_sizes: list[int], 
                 output_dim, 
                 pad_idx, 
                 lstm_dropout_rates: list[float] = None, 
                 lstm_batchnorms: list[nn.Module] = None,
                 embedding_dropout=0.5):
        super(LSTM_CRF_Model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        num_layers = len(lstm_hidden_sizes)        
        if lstm_dropout_rates is not None and len(lstm_dropout_rates) != num_layers:
            warnings.warn("Length of lstm_dropout_rates does not match lstm_hidden_sizes. Padding with 0.0.")
            lstm_dropout_rates = lstm_dropout_rates + [0.0] * (num_layers - len(lstm_dropout_rates))

        if lstm_batchnorms is not None and len(lstm_batchnorms) != num_layers:
            warnings.warn("Length of lstm_batchnorms does not match lstm_hidden_sizes. Padding with None.")

        self.lstm_layers = nn.ModuleList()
        input_size = embedding_dim
        self.dropouts = nn.ModuleList() if lstm_dropout_rates else None
        self.batchnorms = nn.ModuleList() if lstm_batchnorms else None
        for i, hidden_size in enumerate(lstm_hidden_sizes):
            self.lstm_layers.append(nn.LSTM(input_size=input_size, 
                                            hidden_size=hidden_size, 
                                            batch_first=True, 
                                            bidirectional=True))
            input_size = hidden_size * 2 # *2 because of bidirectionality

            if lstm_dropout_rates and len(lstm_dropout_rates) > i:  
                self.dropouts.append(nn.Dropout(lstm_dropout_rates[i]))
            if lstm_batchnorms and len(lstm_batchnorms) > i:
                if lstm_batchnorms[i] is not None:
                    self.batchnorms.append(lstm_batchnorms[i](hidden_size * 2))
                else:
                    self.batchnorms.append(None)

        self.fc = nn.Linear(input_size, output_dim)
        self.crf = CRF(output_dim, batch_first=True)

    def forward(self, x, lengths):
        embeddings = self.embedding(x)
        embeddings = self.embedding_dropout(embeddings)
        lstm_out = embeddings
        for i, lstm in enumerate(self.lstm_layers):
            packed_embeds = nn.utils.rnn.pack_padded_sequence(lstm_out, lengths, batch_first=True, enforce_sorted=False)
            packed_lstm_out, _ = lstm(packed_embeds)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)

            if self.dropouts:
                lstm_out = self.dropouts[i](lstm_out)
            if (self.batchnorms) and (len(self.batchnorms) > i) and (self.batchnorms[i] is not None):
                lstm_out = self.batchnorms[i](lstm_out.transpose(1, 2)).transpose(1, 2)

        emissions = self.fc(lstm_out)
        return emissions

    def forward_crf(self, x, lengths, tags=None):
        emissions = self.forward(x, lengths)
        mask = (x != 0)
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask)
            return loss
        else:
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions
