import torch
import torchvision.models as models

class Feature_ViT(torch.nn.Module):
    def __init__(self, num_views=12):
        super(Feature_ViT, self).__init__()
        base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.num_views = num_views
        self.conv_proj = base_model.conv_proj
        self.encoder = base_model.encoder
        self.embed_dim = 768
        
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
    def __init__(self, num_views=12, num_classes=40, embed_dim=768, num_heads=4, num_layers=2):
        super(MultiView_Classifier, self).__init__()
        self.num_views = num_views
        self.embed_dim = embed_dim
        
        # Simpler view embedding - reduce dimension
        self.view_embedding = torch.nn.Embedding(num_views, embed_dim // 4)
        self.view_proj = torch.nn.Linear(embed_dim + embed_dim // 4, embed_dim)
        
        # Simpler CLS token
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Much simpler transformer - single layer, fewer heads, no dropout initially
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=4,  # Reduce heads
            dim_feedforward=embed_dim,  # Smaller feedforward 
            dropout=0.0,  # No dropout initially
            activation='relu',  # Simpler activation
            batch_first=True,
            norm_first=False
        )
        self.view_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)  # Single layer

        # Very simple classifier - direct mapping
        self.classifier = torch.nn.Linear(embed_dim, num_classes)
        
        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        # Initialize CLS token to zeros
        torch.nn.init.zeros_(self.cls_token)
        
        # Initialize view embedding with small values
        torch.nn.init.normal_(self.view_embedding.weight, std=0.02)
        
        # Initialize classifier with small weights
        torch.nn.init.normal_(self.classifier.weight, std=0.02)
        torch.nn.init.zeros_(self.classifier.bias)
        
        # Initialize projection layer
        torch.nn.init.xavier_uniform_(self.view_proj.weight)
        torch.nn.init.zeros_(self.view_proj.bias)

    def forward(self, view_features):
        B = view_features.shape[0]
        
        # Add view-specific embeddings (smaller dimension)
        view_ids = torch.arange(self.num_views, device=view_features.device).unsqueeze(0).expand(B, -1)
        view_emb = self.view_embedding(view_ids)  # [B, num_views, embed_dim//4]
        
        # Concatenate and project
        view_features_expanded = torch.cat([view_features, view_emb], dim=-1)  # [B, num_views, embed_dim + embed_dim//4]
        view_features = self.view_proj(view_features_expanded)  # [B, num_views, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, view_features], dim=1)  # [B, 1 + num_views, embed_dim]
        
        # Single transformer layer
        x = self.view_transformer(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]  # [B, embed_dim]
        return self.classifier(cls_output)