from ImgDataset import MultiviewImgDataset
from Model import Feature_ViT, MultiView_Classifier
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import gc

class Trainer:
    def __init__(self, num_views=12, num_classes=40, embed_dim=768, num_heads=4, num_layers=2, 
                 freeze_feat_vit=False, freeze_class_model=False, use_amp=True):
        self.num_views = num_views
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.feature_vit = Feature_ViT(num_views=num_views).to(self.device)
        self.multi_view_model = MultiView_Classifier(num_views=num_views, num_classes=num_classes, 
                                                     embed_dim=embed_dim, num_heads=num_heads,
                                                     num_layers=num_layers).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(list(self.feature_vit.parameters()) + list(self.multi_view_model.parameters()), 
                                    lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        
        # Mixed precision training
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        self.train_loader, self.test_loader = None, None
        if freeze_feat_vit:
            for param in self.feature_vit.parameters():
                param.requires_grad = False
        if freeze_class_model:
            for param in self.multi_view_model.parameters():
                param.requires_grad = False
        
        self.freeze_feat_vit = freeze_feat_vit
        self.freeze_class_model = freeze_class_model
        
    def get_train_loader(self, train_dir, batch_size=12, shuffle=True, num_workers=8):
        train_set = MultiviewImgDataset(train_dir, num_views=self.num_views)
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, 
                                     num_workers=num_workers, pin_memory=True, persistent_workers=True)
        return self.train_loader

    def get_test_loader(self, test_dir, batch_size=12, shuffle=False, num_workers=8):
        test_set = MultiviewImgDataset(test_dir, num_views=self.num_views, test_mode=True)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, 
                                    num_workers=num_workers, pin_memory=True, persistent_workers=True)
        return self.test_loader
    
    def get_test_accuracy(self):
        self.feature_vit.eval()
        self.multi_view_model.eval()
        correct_class = torch.zeros(self.num_classes).to(self.device)
        total_class = torch.zeros(self.num_classes).to(self.device)
        correct = 0
        total = 0
        with torch.no_grad():
            for label, data, _ in tqdm(self.test_loader, desc="Testing", leave=False):
                label = label.to(self.device, non_blocking=True)
                data = data.to(self.device, non_blocking=True)
                B, V, C, H, W = data.shape
                data = data.view(B * V, C, H, W)
                
                with autocast(enabled=self.use_amp):
                    features = self.feature_vit(data)
                    cls_tokens = features[:, 0, :]
                    cls_tokens = cls_tokens.view(B, V, -1)
                    outputs = self.multi_view_model(cls_tokens)
                
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                for i in range(label.size(0)):
                    total_class[label[i]] += 1
                    if predicted[i] == label[i]:
                        correct_class[label[i]] += 1
                        
        average_class_accuracy = (correct_class / total_class).mean().item()
        return correct / total, average_class_accuracy
    
    def train(self, num_epochs=10):
        if self.train_loader is None or self.test_loader is None:
            raise ValueError("Train and test loaders must be set before training.")

        best_accuracy = 0.0
        for epoch in range(num_epochs):
            self.feature_vit.eval() if self.freeze_feat_vit else self.feature_vit.train()
            self.multi_view_model.eval() if self.freeze_class_model else self.multi_view_model.train()
            running_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (label, data, _) in enumerate(progress_bar):
                label = label.to(self.device, non_blocking=True)
                data = data.to(self.device, non_blocking=True)
                B, V, C, H, W = data.shape
                data = data.view(B * V, C, H, W)

                self.optimizer.zero_grad()

                with autocast(enabled=self.use_amp, device_type=self.device.type):
                    features = self.feature_vit(data)
                    cls_tokens = features[:, 0, :]
                    cls_tokens = cls_tokens.view(B, V, -1)
                    outputs = self.multi_view_model(cls_tokens)
                    loss = self.criterion(outputs, label)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.feature_vit.parameters()) + list(self.multi_view_model.parameters()), 
                        max_norm=1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.feature_vit.parameters()) + list(self.multi_view_model.parameters()), 
                        max_norm=1.0
                    )
                    self.optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])
                
                # Memory cleanup every 10 batches
                if batch_idx % 10 == 0:
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    elif self.device.type == 'mps':
                        torch.mps.empty_cache()
                    gc.collect()

            self.scheduler.step()
            avg_loss = running_loss / len(self.train_loader)
            test_accuracy, class_accuracy = self.get_test_accuracy()
            
            # Save best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                self.save_model("best_feature_vit.pth", "best_multi_view_model.pth")
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
                  f"Class Accuracy: {class_accuracy:.4f}, Best: {best_accuracy:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        print("Training complete.")
    
    def save_model(self, feat_vit_path="feature_vit.pth", class_model_path="multi_view_model.pth"):
        torch.save(self.feature_vit.state_dict(), feat_vit_path)
        torch.save(self.multi_view_model.state_dict(), class_model_path)
        print("Models saved successfully.")