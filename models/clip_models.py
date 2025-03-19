from pydantic_core import to_jsonable_python

from .clip import clip
from PIL import Image
import torch.nn as nn
import torch
from kan import KAN

CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768
}


class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1, top_layer='fc'):
        super(CLIPModel, self).__init__()
    
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
        self.model, self.preprocess = clip.load(name, device=device)
        # self.preprecess will not be used during training, which is handled in Dataset class
        self.top_layer = top_layer

        # print(top_layer, self.top_layer)

        if self.top_layer == 'fc':
            self.fc = nn.Linear(CHANNELS[name], num_classes, device=device)
    
        elif top_layer == 'transformer':
            # ---------------------------第一次修改---------------------------------------
            # self.fc = nn.Linear(CHANNELS[name], num_classes, device=device)
            # self.transformer_layer = nn.TransformerEncoderLayer(
            #     d_model=768,    # 输入维度与ViT输出对齐
            #     nhead=8,        # 注意力头数
            #     dim_feedforward=1024,
            #     dropout=0.1,
            #     activation='gelu'
            # )
            # ---------------------------------------------------------------------------
    
            # ---------------------------第二次修改---------------------------------------
            # self.transformer_layer = nn.Sequential(
            #     nn.Linear(768, 256),   # 降维
            #     nn.GELU(),
            #     nn.Dropout(0.3),       # 增强Dropout
            #     nn.Linear(256, 1)      # 直接输出
            # )
            # -------------------------------------------------------------------------
    
            # ---------------------------第三次修改---------------------------------------
            self.transformer_layer = nn.Sequential(
                nn.Linear(768, 384),
                nn.GELU(),
                nn.Dropout(0.4),
                # 添加轻量自注意力（单头+低维）
                nn.MultiheadAttention(embed_dim=384, num_heads=1, dropout=0.2),
                nn.LayerNorm(384),
                nn.Linear(384, 1)
            )
            # -------------------------------------------------------------------------

        elif top_layer == 'kan':
            # ---------------------------第一次修改---------------------------------------
            # 参数量：310293
            self.kan = KAN(width=[CHANNELS[name], 28, 1], ckpt_path='./kan_related/model')
            # -------------------------------------------------------------------------

            # ---------------------------第二次修改---------------------------------------
            # 参数量：19205
            # self.kan = KAN(width=[CHANNELS[name], 1], ckpt_path='./kan_related/model')
            # -------------------------------------------------------------------------

            # ---------------------------第三次修改---------------------------------------
            # 参数量：108549
            # self.kan = KAN(width=[CHANNELS[name], 8, 1], grid=5, k=3, ckpt_path='./kan_related/model', device=device)
            # -------------------------------------------------------------------------

            # ---------------------------第四次修改---------------------------------------
            # 参数量：22277
            # self.kan = KAN(width=[CHANNELS[name], 1], grid=5, ckpt_path='./kan_related/model', device=device)
            # -------------------------------------------------------------------------

        elif top_layer == 'kan_fc':
            self.kan = KAN(width=[CHANNELS[name], 16], grid=5, k=3, ckpt_path='./kan_related/model', device=device)
            self.fc = nn.Linear(16, num_classes, device=device)

        elif top_layer == 'fc_kan':
            self.fc = nn.Linear(CHANNELS[name], 16, device=device)
            self.kan = KAN(width=[16, 1], grid=5, k=3, ckpt_path='./kan_related/model', device=device)


    def forward(self, x, return_feature=False, kan_prune=False):
        features = self.model.encode_image(x).to(torch.float32)
        if return_feature:
            return features
        if self.top_layer == 'transformer':
            # ---------------------------第一次修改---------------------------------------
            # transformer_input = features.unsqueeze(0)
            # transformer_output = self.transformer_layer(transformer_input)
            # transformer_output = transformer_output.squeeze(0)
            # return self.fc(transformer_output)
            # -------------------------------------------------------------------------

            # ---------------------------第二次修改------------------------------------
            # return self.transformer_layer(features)
            # -------------------------------------------------------------------------

            # ---------------------------第三次修改-------------------------------------
            x = self.transformer_layer[0](features)  # Linear(768->384)
            x = self.transformer_layer[1](x)         # GELU
            x = self.transformer_layer[2](x)         # Dropout
            
            # 调整维度适配MultiheadAttention输入要求 (seq_len, batch_size, embed_dim)
            x_attn = x.unsqueeze(0)                 # [1, batch_size, 384]
            
            # 自注意力计算 (需要处理key_padding_mask时可添加参数)
            attn_output, _ = self.transformer_layer[3](x_attn, x_attn, x_attn)
            
            # 残差连接 + 层归一化
            x = x + attn_output.squeeze(0)           # [batch_size, 384]
            x = self.transformer_layer[4](x)         # LayerNorm
            
            # 最终分类层
            # return torch.sigmoid(self.transformer_layer[5](x))  # Linear(384->1)
            return self.transformer_layer[5](x)  # Linear(384->1)
            # -------------------------------------------------------------------------
            
        elif self.top_layer == 'fc':
            return self.fc(features)

        elif self.top_layer == 'kan':
            # ---------------------------第三次修改---------------------------------------
            # if kan_prune:
            #     self.kan.prune()
            #     print('Pruned.')
            # -------------------------------------------------------------------------
            return self.kan(features)

        elif self.top_layer == 'kan_fc':
            x = self.kan(features)
            x = self.fc(x)
            return x

        elif self.top_layer == 'fc_kan':
            x = self.fc(features)
            x = self.kan(x)
            return x



if __name__ == "__main__":
    clip = CLIPModel("ViT-L/14")
