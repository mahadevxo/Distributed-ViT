import torch
import torchvision.models as models
import torch.nn.functional as F

class Feature_ViT(torch.nn.Module):
    def __init__(self, num_views=12):
        super(Feature_ViT, self).__init__()
        base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.num_views = num_views
        self.conv_proj = base_model.conv_proj
        self.encoder = base_model.encoder
        self.embed_dim = 768
        
        # Add layer normalization after encoder
        self.layer_norm = torch.nn.LayerNorm(self.embed_dim)
        
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
        x = self.encoder(x)
        x = self.layer_norm(x)
        return x
    
class MultiView_Classifier(torch.nn.Module):
    def __init__(self, num_views=12, num_classes=40, embed_dim=768, num_heads=4, num_layers=2):
        super(MultiView_Classifier, self).__init__()
        self.num_views = num_views
        self.embed_dim = embed_dim
        
        # Enhanced architecture
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
        self.view_pos_embedding = torch.nn.Parameter(torch.randn(1, num_views+1, embed_dim))

        # Deeper transformer with more heads
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,  # Larger FFN
            batch_first=True, 
            dropout=0.1,
            activation='gelu'
        )
        self.view_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling for better aggregation
        self.attention_pool = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.pool_query = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Enhanced classifier head
        self.dropout1 = torch.nn.Dropout(0.3)
        self.classifier_hidden = torch.nn.Linear(embed_dim, embed_dim // 2)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(embed_dim // 2, num_classes)
        
        # Layer normalization
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, view_features):
        B = view_features.shape[0]
        
        # Add cls token and positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        view_features = torch.cat((cls_tokens, view_features), dim=1)
        view_features = view_features + self.view_pos_embedding
        
        # Multi-view transformer
        x = self.view_transformer(view_features)
        x = self.layer_norm(x)
        
        # Attention pooling
        query = self.pool_query.expand(B, -1, -1)
        pooled_features, _ = self.attention_pool(query, x, x)
        pooled_features = pooled_features.squeeze(1)
        
        # Classification head
        x = self.dropout1(pooled_features)
        x = F.gelu(self.classifier_hidden(x))
        x = self.dropout2(x)
        x = self.classifier(x)
        return x