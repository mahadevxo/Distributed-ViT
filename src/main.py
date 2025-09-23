from Trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(num_views=12, num_classes=40, embed_dim=768, 
                      num_heads=4, num_layers=2, freeze_feat_vit=False,
                      freeze_class_model=False)
    
    train_loader = trainer.get_train_loader("data/ModelNet40-12-split/train", batch_size=32, shuffle=True)
    test_loader = trainer.get_test_loader("data/ModelNet40-12-split/test", batch_size=32, shuffle=False)
    
    trainer.train(num_epochs=10)
    
    test_accuracy = trainer.get_test_accuracy()
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    trainer.save_model()