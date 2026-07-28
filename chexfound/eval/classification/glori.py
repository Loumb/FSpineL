from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.modules.transformer import _get_activation_fn


def create_mldecoder_input(x_tokens_list, use_n_blocks):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([patch_token for patch_token, _ in intermediate_output], dim=-1)  # patch only
    # output = torch.cat(
    #     [torch.cat([class_.unsqueeze(1), patch_], dim=1) for patch_, class_ in intermediate_output],
    #     dim=-1
    # )  # input both class and patch tokens to mldecoder

    cls = torch.cat([cls_tok for _, cls_tok in intermediate_output], dim=-1)
    return output.float(), cls.float()


class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                return_attention=False) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        # tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt2, attn = self.multihead_attn(tgt, memory, memory, average_attn_weights=False)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_attention:
            return tgt, attn
        else:
            return tgt


class TransformerDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                return_attention=False) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i, mod in enumerate(self.layers):
            if i < len(self.layers) - 1:
                output = mod(output, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)
            else:
                output, attn = mod(output, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             return_attention=True)
                if return_attention:
                    return attn

        if self.norm is not None:
            output = self.norm(output)

        return output


@torch.jit.script
class GroupFC(object):

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):  # query dim
            h_i = h[:, i, :]  # features [bs, decoder_embed]
            w_i = duplicate_pooling[i, :, :]  # weights [decoder_embed, 1]
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class GLoRI(nn.Module):
    """Global and local Representations Integration"""
    def __init__(self, num_classes,# <—— 针对不同任务设置输出维度
                 decoder_embedding=768, initial_num_features=2048,
                 use_n_blocks=4, multiview=False, cat_cls=False):
        super(GLoRI, self).__init__()
        # number of queries
        num_queries = num_classes

        # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding  # dimension in cross-attention
        embed_standart = nn.Linear(initial_num_features, decoder_embedding)  # map to dimension in cross-attention

        # non-learnable queries
        query_embed = nn.Embedding(num_queries, decoder_embedding)  # random init for queries
        query_embed.requires_grad_(False)

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        layer_decode = TransformerDecoderLayerOptimal(d_model=decoder_embedding,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        # self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)
        self.decoder = TransformerDecoder(layer_decode, num_layers=num_layers_decoder)
        self.decoder.embed_standart = embed_standart
        self.decoder.query_embed = query_embed

        self.decoder.num_classes = num_classes
        if cat_cls:
            out_embed = decoder_embedding + initial_num_features
        else:
            out_embed = decoder_embedding
        self.decoder.duplicate_pooling = torch.nn.Parameter(
            torch.Tensor(num_queries, out_embed, 1))  # (queries, 786, 1)
        self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))

        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        self.decoder.group_fc = GroupFC()
        self.use_n_blocks = use_n_blocks  # use last n layer outputs
        self.multiview = multiview  # use multiview learning
        self.cat_cls = cat_cls  # add new query tokens

    def forward(self, x, return_attention=False):
        x, cls = create_mldecoder_input(x[0], use_n_blocks=self.use_n_blocks)

        embedding_spatial = x
        embedding_spatial_786 = self.decoder.embed_standart(embedding_spatial)  # (bs, seq_len, 768)
        embedding_spatial_786 = torch.nn.functional.relu(embedding_spatial_786, inplace=True)

        bs = embedding_spatial_786.shape[0]

        query_embed = self.decoder.query_embed.weight
        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand, add bs dimension
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1), return_attention=return_attention)  # [queries, bs, 786]

        # get attention weights
        if return_attention:
            return h

        h = h.transpose(0, 1)  # [bs, queries, 786]

        if self.cat_cls:
            # h = torch.cat([cls.unsqueeze(1), h], dim=-1)
            h = torch.cat([cls.unsqueeze(1).repeat(1, h.shape[1], 1), h], dim=-1)

        out_extrap = torch.zeros(h.shape[0], h.shape[1], 1, device=h.device, dtype=h.dtype)
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)
        h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]  # [bs, num_classes]
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out  # [bs, num_classes]

        return logits.squeeze(-1)
