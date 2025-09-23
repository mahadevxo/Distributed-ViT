from Trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(num_views=12, num_classes=40, embed_dim=768, 
                      num_heads=8, num_layers=3, freeze_feat_vit=False,
                      freeze_class_model=False, use_amp=True)
    
    train_loader = trainer.get_train_loader("data/ModelNet40-12-split/train", batch_size=16, shuffle=True, num_workers=4)
    test_loader = trainer.get_test_loader("data/ModelNet40-12-split/test", batch_size=32, shuffle=False, num_workers=4)
    
    trainer.train(num_epochs=50)
    
    test_accuracy, class_accuracy = trainer.get_test_accuracy()
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%, Class Accuracy: {class_accuracy * 100:.2f}%")
    
    trainer.save_model()