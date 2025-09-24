from Trainer import Trainer

def main():
    train_data_dir = "../data/ModelNet40-12-split/train"
    test_data_dir = "../data/ModelNet40-12-split/test"

    trainer = Trainer(num_views=12, num_classes=40, embed_dim=768, 
                     num_heads=12, num_layers=6,  # Increased depth
                     freeze_feat_vit=False,
                     use_amp=True)
    

    trainer.get_train_loader(train_data_dir, batch_size=8, shuffle=True, num_workers=4)  # Smaller batch for gradient accumulation
    trainer.get_test_loader(test_data_dir, batch_size=32, shuffle=False, num_workers=4)

    try:
        trainer.train(num_epochs=200)  # Increased epochs significantly
        test_accuracy, class_accuracy = trainer.get_test_accuracy()
        print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%, Class Accuracy: {class_accuracy * 100:.2f}%")
        
    except (Exception, KeyboardInterrupt) as e:
        print(f"Training interrupted. Error: {e}")
        trainer.save_model("feature_vit-interrupted.pth", "multi_view_model-interrupted.pth")

if __name__ == "__main__":
    print("Starting end-to-end training...")
    main()