import torch 
import torch.nn as nn
import torch.nn.functional as F

import math

## timestep t -> 삼각함수 주파수 임베딩 → mlp 투영 → hidden_size 차원의 벡터
# https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py 참고.

class TimeEmbedding(nn.Module):
    def __init__(self,hidden_size=256, frequency_embedding_size = 256):
        # dim은 output dim
        super().__init__()
        
        self.frequency_embedding_size = frequency_embedding_size
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size)
        )
    
    def timestep_Embedding(self, timestep, dim, max_period = 10000):
        ## timestep은 [N] 꼴로 입력을 받음 
        if dim %2 != 0 :
            raise RuntimeError("DIM 2로 안나눠짐 -> cos sin embedding 불균형")
        device = timestep.device
        half = dim //2 # 반으로 나눠서 cos / sin 사용 
        #  exp ( -log(10000) * 0/half, -log(10000) * 1/half, -log(10000) * 2/half , ... )
        freqs = torch.exp(-torch.log(torch.tensor(float(max_period), device=device, dtype=torch.float32))* (torch.arange(0, half, device=device, dtype=torch.float32) / float(half))
)     ## 그러니까 이게 timestep에 의존적인게 아니라, 세로로.. 즉 dim 축을 구성하기 위한 frequency
        args = timestep[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # [N, dim]
        
        ### 동작과정 !!
        # timestep에는 각 배치마다 쓸 차원이 들어옴. N은 배치사이즈와 동일해야함
        # 1. freqs는 half 차원만큼만 만들어서 줌 
        # 2. 들어온 timestep들에다가 이걸 세로축으로 곱하고, cos sin한걸 세로축으로 또 이어붙임
        # 3. 결국 무슨 숫자가 어떻게 들어오던, frequency만 동일하다면 똑같은 값 나옴 즉 [10,20,30] 이나 [10,30,80]이나 10은 동일한 embedding 출력
        
        return embedding
    
    def forward(self,timestep):
        ## timestep 넣으면 embedding으로 치환됨
        if timestep.ndim == 0:
            timestep= timestep.unsqueeze(-1)
        t_freq = self.timestep_Embedding(timestep, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb    

class DiTBlock(nn.Module):
    def __init__(self, mode, num_head, embedding_dim, mlp_ratio = 4):
        super().__init__()
        
        self.mode = mode
        
        self.linear = nn.Linear(256,embedding_dim)
        
        if self.mode == "adaLN-Zero":
                    
            self.norm1 = nn.LayerNorm(embedding_dim)
            self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_head, batch_first=True)
            
            self.norm2 = nn.LayerNorm(embedding_dim)

            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim,embedding_dim*mlp_ratio),
                nn.SiLU(),
                nn.Linear(embedding_dim*mlp_ratio,embedding_dim),
            )

            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embedding_dim, 6 * embedding_dim)
            )
            
            ### adaln -> ZERO !!!!
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)
            
        elif self.mode == "Cross-Attention":
            self.norm1 = nn.LayerNorm(embedding_dim)
            self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_head, batch_first=True)
            
            self.norm2 = nn.LayerNorm(embedding_dim)
            self.proj = nn.Linear(embedding_dim,embedding_dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_head, batch_first=True) ## 여기에다가 cond 씌워주기
            
            self.norm3 = nn.LayerNorm(embedding_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim,embedding_dim*mlp_ratio),
                nn.SiLU(),
                nn.Linear(embedding_dim*mlp_ratio,embedding_dim),
            )
            
        elif self.mode == "In-Context-Conditioning":
            self.flag = False ## 이거로 concat 했는지 안했는지
            self.norm1 = nn.LayerNorm(embedding_dim)
            self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_head, batch_first=True)
            
            self.norm2 = nn.LayerNorm(embedding_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim,embedding_dim*mlp_ratio),
                nn.SiLU(),
                nn.Linear(embedding_dim*mlp_ratio,embedding_dim),
            )


        else :
            raise RuntimeError("모드 확인 고고")

    def modulate(self,x,scale,shift):
        ## 공식 깃허브에서는, modulate라는 함수를 따로 빼가지고 여기다가 cond 부분을 모두 구현해두엇음. 이 방식을 모방해보자.
        return x * (1+ scale.unsqueeze(1)) + shift.unsqueeze(1)
        
    def forward(self,x,label,timestep):
        ## x : [B, 패치개수, hidden_dim]
        if self.mode == "adaLN-Zero":
            if label.shape[-1] != timestep.shape[-1]:
                label = self.linear(label)
            cond = label+timestep ## cond 차원은 B, 256
            scale_a1,shift_b1,scale_c1,scale_a2,shift_b2,scale_c2 = self.adaLN_modulation(cond).chunk(6,dim=1)
            
            temp = x
            x = self.norm1(x)
            x = self.modulate(x,scale_c1,shift_b1)
            x, _= self.attn(x,x,x)
            x = x * scale_a1.unsqueeze(1)
            x = x + temp
            
            temp = x
            x = self.norm2(x)
            x = self.modulate(x,scale_c2,shift_b2)
            x = self.mlp(x)
            x = x * scale_a2.unsqueeze(1)
            x = x+temp
            return x
            
        elif self.mode == "Cross-Attention":
            if label.shape[-1] != timestep.shape[-1]:
                label = self.linear(label)
            cond = label+timestep
            
            temp = x
            x = self.norm1(x)
            x, _ = self.attn(x,x,x)
            x = x + temp
            
            temp = x 
            x = self.norm2(x)   
            ## 여기서 지금 x의 차원은 B, 패치개수, dim
            ## cond의 차원은 B, dim .. 이걸 어쩐담? -> projection 해주기
            x,_ = self.cross_attn(x,cond[:,None,:],cond[:,None,:])   ## 이거 순서가 Q K V  라고 함
            x = x + temp
            
            temp = x
            x = self.norm3(x)
            x = self.mlp(x)
            x = x + temp
            return x
            
        elif self.mode == "In-Context-Conditioning":
            ## 만약 이거라면, shape 맞춰주고 cat 해서 들어와야되니까 그냥 밖에서 처리해주기....forward(self,x,label,timestep): 중에서 label 만 들어오게 하기 그냥
            assert (timestep is None) and (label is None), "CONCAT이니까 밖에서 처리하고 들어오소~"
            
            temp = x
            x = self.norm1(x)
            x,_ = self.attn(x,x,x)
            x = x+temp
            
            temp = x
            x = self.norm2(x)
            x = self.mlp(x)
            x = x + temp
            return x

        else :
            raise RuntimeError("모드 확인 고고")

