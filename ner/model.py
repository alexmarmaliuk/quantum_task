import torch
import torch.nn as nn
from TorchCRF import CRF 

class LSTM_CRF_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, dropout=0.5):
        super(LSTM_CRF_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            batch_first=True, 
                            bidirectional=True)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 because of bidirectionality
        self.crf = CRF(output_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        embeddings = self.embedding(x)  
        embeddings = self.dropout(embeddings)
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
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
