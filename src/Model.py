import torch
import torchvision.models as models

class Feature_ViT(torch.nn.Module):
    def __init__(self, num_views=12):
        super(Feature_ViT, self).__init__()
        base_model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        self.num_views = num_views
        self.conv_proj = base_model.conv_proj
        self.encoder = base_model.encoder
        self.embed_dim = 1024
        
        if hasattr(base_model, 'class_token'):
            self.class_token = base_model.class_token
        else:
            self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.embed_dim))
            
        if hasattr(base_model, 'pos_embedding'):
            self.pos_embedding = base_model.pos_embedding
        else:
            self.pos_embedding = torch.nn.Parameter(torch.randn(1, 197, self.embed_dim))
    
    def forward(self, x):
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding
        return self.encoder(x)
    
class MultiView_Classifier(torch.nn.Module):
    def __init__(self, num_views=12, num_classes=40, embed_dim=1024, num_heads=4, num_layers=2):
        super(MultiView_Classifier, self).__init__()
        self.num_views = num_views
        self.embed_dim = embed_dim
        
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
        self.view_pos_embedding = torch.nn.Parameter(torch.randn(1, num_views+1, embed_dim))

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.1)
        self.view_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, view_features):
        B = view_features.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        view_features = torch.cat((cls_tokens, view_features), dim=1)
        
        view_features = view_features + self.view_pos_embedding
        
        x = self.view_transformer(view_features)
        x = x[:, 0]
        x = self.dropout(x)
        x = self.classifier(x)
        return x