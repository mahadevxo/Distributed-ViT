from Trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(num_views=12, num_classes=40, embed_dim=1024, 
                      num_heads=16, num_layers=4, freeze_feat_vit=False,
                      freeze_class_model=False, use_amp=True)
    
    trainer.get_train_loader("data/ModelNet40-12-split/train", batch_size=16, shuffle=True, num_workers=8)
    trainer.get_test_loader("data/ModelNet40-12-split/test", batch_size=32, shuffle=False, num_workers=8)
    trainer.train(num_epochs=50)
    
    test_accuracy, class_accuracy = trainer.get_test_accuracy()
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%, Class Accuracy: {class_accuracy * 100:.2f}%")
    
    trainer.save_model()