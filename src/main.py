from Trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(num_views=12, num_classes=40, embed_dim=768, 
                      num_heads=8, num_layers=3, freeze_feat_vit=False,
                      freeze_class_model=False, use_amp=True)
    
    trainer.train(num_epochs=50)
    
    test_accuracy, class_accuracy = trainer.get_test_accuracy()
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%, Class Accuracy: {class_accuracy * 100:.2f}%")
    
    trainer.save_model()