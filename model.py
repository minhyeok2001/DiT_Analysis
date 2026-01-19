import torch 
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.models.embeddings import get_2d_sincos_pos_embed
from transformers import CLIPTokenizer, CLIPTextModel

from module import TimeEmbedding, DiTBlock

class DiT(nn.Module):
    def __init__(self, mode, num_blocks, num_head, patch_size=4, embedding_dim=256, resolution=128):
        super().__init__()
        
        mode_list = ["adaLN-Zero","Cross-Attention","In-Context Conditioning"]
        
        assert mode in mode_list, "mode 선택 다시하기 !!"
        
        self.mode = mode
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        
        self.time_emb = TimeEmbedding(hidden_size=embedding_dim)
        
        grid_size = (resolution // 8) // patch_size
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=embedding_dim, 
            grid_size=grid_size,
            output_type='pt'
        )
        self.register_buffer("pos_embed", pos_embed)
        
        self.linear1 = nn.Linear(4*self.patch_size*self.patch_size,embedding_dim)
        
        self.blocks = nn.ModuleList([
            DiTBlock(mode=mode, num_head=num_head, embedding_dim=embedding_dim) 
            for _ in range(num_blocks)
        ])
        
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.linear2 = nn.Linear(embedding_dim,self.patch_size*self.patch_size*4)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        
        self.B = B
        self.C = C
        self.H = H
        self.W = W
        
        assert H%p == 0 and W%p == 0, "패치 크기로 나눠떨어져야해요~~"
        
        x = x.reshape(B, C, H//p, p, W//p, p)
        x = x.permute(0,2,4,1,3,5)
        x = x.flatten(1,2).flatten(2,4) ## flatten은 시작, 끝점 기준으로 하니까..
        
        
        ## 결국 차원은 [B, 패치개수, C * p * p]
        return x
    
    def reverse_patchify(self,x):
        ## DiT block 거친 뒤의 차원 : B, (h 패치개수 x w 패치개수), (채널, 패치, 패치)
        x = x.reshape(self.B, self.H//self.patch_size, self.W//self.patch_size, self.C, self.patch_size, self.patch_size)
        x = x.permute(0,3,1,4,2,5)
        x = x.reshape(self.B, self.C, self.H, self.W)
        ## B, C, H, W = x.shape
        return x
        

    def forward(self,x,label,timestep):
        ## 1. VAE 거치기 ( 근데 이건 구현상 loop에서 호출하는게 더 낫다고 판단.)
        x = self.patchify(x)
        
        x = self.linear1(x)
        x = x + self.pos_embed
        
        timestep = self.time_emb(timestep)
        
        #print("timestep shape : ",timestep.shape)
        #print("label shape : ",label.shape)

        ## 2. DIT
        for block in self.blocks:
            x = block(x,label,timestep)
        
        ## 3. LN & linear
        # 지금 차원은 B, 패치개수, hidden_dim
        
        x = self.layernorm(x)
        x = self.linear2(x)
        
        ## 4. 다시 reshape
        # 지금 차원은 B, 패치개수, c*p*p
        x = self.reverse_patchify(x)
        
        return x

        
        
        