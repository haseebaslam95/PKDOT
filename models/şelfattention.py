import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.modules import get_activation

from models.modules.fusion import get_fusion_layer


class PositionalEncoding(nn.Module):
    """
    Module that performs fixed sine-cosine position encoding
    """
    def __init__(self, d_model, max_len=5000, batch_first=True):
        super(PositionalEncoding, self).__init__()
        
        self.batch_first = batch_first
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)    # add another dimension
        if not batch_first:
            pe = pe.transpose(0, 1) # make the sequence dimension the 0 dim.
            
        self.register_buffer("pe", pe)  # not a model parameter, but is saved

    def forward(self, x):
        """
        Forward pass of positional Encoding
        :param x: Sequence passed to PE [sequence_length, batch_size, embedding_dim] or [batch_size, sequence_length embedding_dim]
        :return: output [sequence_length, batch_size, embedding_dim] or [batch_size, sequence_length embedding_dim]
        """
        if not self.batch_first:
            return x + self.pe[:x.size(0), :]
        
        else:
            return x + self.pe[:, :x.size(1), :]


class SelfAttentionModel(nn.Module):
    """
    Combines a linear encoder, and a self-attention stack
    """

    def __init__(self, n_inputs, n_heads, d_embedding, d_feedforward, n_layers, dropout=0.3, activation="gelu", batch_first=True, mask=None):
        """
        Args:
            n_inputs Dimension of the input features 
            n_heads Number of attention heads
            n_hidden Dimension of the embeddings in the attention and feedforward blocks
            n_layers Number of attention blocks
            dropout Dropout rate applied inside the attention stack
            batch_first Whether input has the batch as first dimension. Default True
            mask Tensor [seq_len, seq_len] masks timesteps in the sequence when computing attention. Defaults to None
        """
        
        super(SelfAttentionModel, self).__init__()

        self.model_type = "Transformer"
        self.batch_first = batch_first
        self.d_embedding = d_embedding
        self.d_ff = d_feedforward
        self.num_features = self.d_embedding
        
        # register mask into buffer
        self.register_buffer("src_mask", mask)
        
        self.linear = nn.Linear(n_inputs, d_embedding)
        if isinstance(activation, str):
            self.act = get_activation(activation)
        else:
            self.act = activation
        
        self.pos_encoder = PositionalEncoding(d_embedding, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        enc_layers = nn.TransformerEncoderLayer(d_model=d_embedding, nhead=n_heads, dim_feedforward=d_feedforward, dropout=dropout, batch_first=batch_first, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=enc_layers, num_layers=n_layers)

        self.init_weights() # does nothing atm

    def init_weights(self):
        pass

    def forward(self, src:torch.Tensor):
        """
        Inputs: [BS, T, N] or [T, BS, N] depending on self.batch_first
        """

        # bring input to dimension of attention model
        x = self.act(self.linear(src))

        #x = x * math.sqrt(self.n_hidden)
        
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        embedding = self.transformer_encoder(x, self.src_mask)
        #output = self.decoder(embedding)

        return embedding
    
    @staticmethod
    def _gen_square_subsequent_mask(seq_len) -> torch.Tensor:
        """
        Creates a mask that hides future time steps. Can be passed to the Self Attention Module as argument.
        :param seq_len: Length of the sequence
        :return: A tensor [seq_len, seq_len] with a triangle structure that contains 0.0 where entries are not to be masked and a large negative number where they are
        """

        mask = torch.tril(torch.ones(size=(seq_len, seq_len), device=torch.cuda.current_device()))
        mask = mask.float().masked_fill(mask == 0, float(-1.0e-9)).masked_fill(mask == 1, float(0.0))
        return mask
