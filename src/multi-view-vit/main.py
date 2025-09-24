from Trainer import Trainer
import torch

def main():
    trainer = Trainer(num_views=12, num_classes=40, embed_dim=1024, 
                      num_heads=16, num_layers=4, freeze_feat_vit=False,
                      freeze_class_model=False, use_amp=True)
    
    trainer.get_train_loader("../data/ModelNet40-12-split/train", batch_size=16, shuffle=True, num_workers=8)
    trainer.get_test_loader("../data/ModelNet40-12-split/test", batch_size=32, shuffle=False, num_workers=8)
    
    try:
        trainer.train(num_epochs=50)
        test_accuracy, class_accuracy = trainer.get_test_accuracy()
        print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%, Class Accuracy: {class_accuracy * 100:.2f}%")
        torch.save(trainer.feature_vit.state_dict(), "feature_vit.pth")
        torch.save(trainer.multi_view_model.state_dict(), "multi_view_model.pth")
        print("Models saved successfully.")
        
    except  Exception as e:
        print(f"Training interrupted. Error: {e}")
        torch.save(trainer.feature_vit.state_dict(), "feature_vit-interrupted.pth")
        torch.save(trainer.multi_view_model.state_dict(), "multi_view-model-interrupted.pth")
        print("Models saved successfully.")
    
    trainer.save_model()

if __name__ == "__main__":
    print("Starting training process...")
    main()