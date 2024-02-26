import torch
from torch import nn

class TemporalOFEmbedding(nn.Module):

    def __init__(self, input_size, embed_size, position_embedding_data, max_pos_len=100, dropout=0):
        super().__init__()
        
        # # way1 cnn
        # self.projection = nn.Conv2d(2, input_size, kernel_size=224, stride=224)
        # self.proj = nn.Linear(input_size, embed_size)
        # # self.fc = self.proj
        
        # way2 downsampling
        # self.projection = nn.Conv2d(2, 1, kernel_size=8, stride=8)
        # input_size = (224 // 8) ** 2

        self.projection = nn.Conv2d(2, 1, kernel_size=16, stride=16)
        input_size = (224 // 16) ** 2
        
        # # way3 pathc
        # self.projection = nn.Conv2d(2, 16, kernel_size=32, stride=32)
        # input_size = (224 // 32) ** 2 * 16
        
        self.fc = nn.Linear(input_size, embed_size)
        # self.proj = nn.Linear(input_size, embed_size)
        self.bos = nn.Parameter(torch.empty(embed_size))
        self.eos = nn.Parameter(torch.empty(embed_size))

        # Frame positional embedding
        self.register_buffer("position_ids", torch.arange(max_pos_len+2).expand(1, -1))
        self.frame_pos_embed = nn.Embedding(max_pos_len+2, embed_size, _weight=position_embedding_data)
        self.ln = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.projection.apply(init_weights)
        self.fc.apply(init_weights)
        # self.proj.apply(objectives.init_weights)
        nn.init.trunc_normal_(self.bos, mean=0, std=0.02)
        nn.init.trunc_normal_(self.eos, mean=0, std=0.02)
        self.ln.apply(init_weights)

    def forward(self, batch):
        B, L, C, H, W = batch['video'].size()
        video_embed = self.projection(batch['video'].view(-1, C, H, W)) # -> B*L, 1024, 1, 1 / B*L, 1, 28, 28
        video_embed = video_embed.flatten(1).view(B, L, -1) # -> B, L, 747
        video_embed = self.fc(video_embed) # 747 -> 768
        # video_embed = self.proj(video_embed)
        B, S, D = video_embed.size()
        video_embed = torch.cat([self.bos.expand(B, 1, -1), video_embed,
                                  torch.zeros(B, 1, D, device=video_embed.device)], dim=1)
        ends = batch["video_mask"].sum(dim=1) - 1
        video_embed[torch.arange(B), ends] = self.eos

        pos_ids = self.position_ids[:, :video_embed.size(1)]
        video_embed += self.frame_pos_embed(pos_ids)
        video_embed = self.ln(video_embed)
        video_embed = self.dropout(video_embed)

        return video_embed


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()