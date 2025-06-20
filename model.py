import torch
import torch.nn as nn
import math

class InputEbeddings(nn.Module):

    def __init__(self,d_model: int,vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncodings(nn.Module):

    def __init__(self,d_model: int,seq_len: int, dropout : float)-> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #create a matrix of shape (seq_len,d_model)
        pe = torch.zeros(seq_len,d_model)
        #Create a vector of shape (seq_len)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        #Apply the sin to even position
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)
    
    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self,eps: float=10**-6) -> None:
        super().__init__()  
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)
        return self.alpha * (x-mean)/(std+self.eps) + self.bias

class FeedForwardblock(nn.Module):
    
    def __init__(self,d_model:int,d_ff:int,dropout:float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        # (Batch,seq_len,d_model) --> (Batch,seq_len,d_ff) --> (Batch,seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,d_model : int,h:int,dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        assert d_model%h==0,"d_model is not divisible by h"
        
        self.d_k = d_model // h 
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout) 
    
    @staticmethod
    def attention(query,key,value,mask,dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # (Batch,h,seq_len,d_k) --> (Batch,h,seq_len)
        attention_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1) # (Batch,h,seq_len,seq_len)
        if dropout is not None:
            attention_score = dropout(attention_score)
        return (attention_score @ value), attention_score
    

    def forward(self,q,k,v,mask):
        query = self.wq(q)  #(Batch,seq_len,d_model) --> (Batch,seq_len,d_model)
        key = self.w_k(k)   #(Batch,seq_len,d_model) --> (Batch,seq_len,d_model)
        value = self.w_v(v) #(Batch,seq_len,d_model) --> (Batch,seq_len,d_model)
        
        # (Batch,Seq_len,d_model) --> (Batch,Seq_len,h,d_k) --> (batch,k,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        
        # (Batch,h,seq_len,d_k) --> (Batch,Seq_len,h,d_k) --> (Batch,seq_len,d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)
        #(Batch,seq_len,d_model) --> (Batch,seq_len,d_model)
        return self.w_o(x)

class ResidualConnectionBlock(nn.Module):

    def __init__(self,dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __inti__(self,self_attention_block : MultiHeadAttentionBlock,feed_forward_block:FeedForwardblock,dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnectionBlock(dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x = self.residual_connections[0](x,lambda x : self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x,self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormaclass DecoderBlock(nn.Module):
    
    def __init__(self, self_attention_block:MultiHeadAttentionBlock,cross_attetnion_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardblock,dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attetnion_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module(ResidualConnectionBlock(dropout) for _ in range(3))


    def forward(self,x,encoder_out,src_mask,tgt_mask):

        x = self.residual_connections[0](x,lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x,lambda x: self.cross_attention_block(x,encoder_out,encoder_out,src_mask))
        x = self.residual_connections[2](x,self.feed_forward_block)
        return xlization()

    def forward(self,x,mask):
        
        for layers in self.layers:
            x = layers(x,mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self,layers : nn.ModuleList) -> None :
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,encoder_out,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_out,src_mask,tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        #(Batch,seqlen,dmodel) --> (Batch,seqlen,vocabsize)
        return torch.log_softmax(self.proj(x),dim = -1)

class Transformer(nn.Module):

    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEbeddings,tgt_embed:InputEbeddings,src_pos:PositionalEncodings,tgt_pos:PositionalEncodings,projection:ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = ProjectionLayer

    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self, encoder_out,src_mask,tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_out,src_mask,tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int=512,N:int=6,h:int=8,dropout:float=0.1,d_ff:int=2048) -> Transformer:
    #Create the embedding
    src_embed = InputEbeddings(d_model,src_vocab_size)
    tgt_embed = InputEbeddings(d_model,src_vocab_size)

    #Create the positional encoding layers
    src_pos = PositionalEncodings(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncodings(d_model,tgt_seq_len,dropout)

    #Create the encoder Blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardblock(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    #Create the Decoder Block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward = FeedForwardblock(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention,decoder_cross_attention,feed_forward,dropout)
        decoder_blocks.append(decoder_block)

    #Create the Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #Projection layers
    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)
    
    #Create the Transforemr
    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)
    
    # Initialize the Parameters
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return transformer




