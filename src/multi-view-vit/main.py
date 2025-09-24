from Trainer import Trainer

def main():
    train_data_dir = "../data/ModelNet40-12-split/train"
    test_data_dir = "../data/ModelNet40-12-split/test"

    # --- Stage 1: Train the classifier head ---
    print("--- Starting Stage 1: Training Classifier Head ---")
    trainer_stage1 = Trainer(num_views=12, num_classes=40, embed_dim=768, 
                             num_heads=4, num_layers=2, 
                             freeze_feat_vit=True,
                             use_amp=True)
    
    trainer_stage1.optimizer.param_groups[0]['lr'] = 0 
    trainer_stage1.optimizer.param_groups[1]['lr'] = 1e-3

    trainer_stage1.get_train_loader(train_data_dir, batch_size=32, shuffle=True, num_workers=8)
    trainer_stage1.get_test_loader(test_data_dir, batch_size=32, shuffle=False, num_workers=8)
    
    try:
        trainer_stage1.train(num_epochs=15)
    except (Exception, KeyboardInterrupt) as e:
        print(f"Stage 1 training interrupted. Error: {e}")
        trainer_stage1.save_model("feature_vit-s1-interrupted.pth", "multi_view-model-s1-interrupted.pth")
        return

    print("\n--- Stage 1 Complete. Starting Stage 2: Fine-tuning ---")

    # --- Stage 2 ---
    trainer_stage2 = Trainer(num_views=12, num_classes=40, embed_dim=768, 
                             num_heads=4, num_layers=2, 
                             freeze_feat_vit=False,
                             use_amp=True)
    
    trainer_stage2.feature_vit.load_state_dict(trainer_stage1.feature_vit.state_dict())
    trainer_stage2.multi_view_model.load_state_dict(trainer_stage1.multi_view_model.state_dict())

    trainer_stage2.optimizer.param_groups[0]['lr'] = 1e-5
    trainer_stage2.optimizer.param_groups[1]['lr'] = 1e-4

    trainer_stage2.get_train_loader(train_data_dir, batch_size=32, shuffle=True, num_workers=8)
    trainer_stage2.get_test_loader(test_data_dir, batch_size=32, shuffle=False, num_workers=8)

    num_epochs_s2 = int(input("Enter number of epochs for Stage 2 (fine-tuning): ") or "35")

    try:
        trainer_stage2.train(num_epochs=num_epochs_s2)
        test_accuracy, class_accuracy = trainer_stage2.get_test_accuracy()
        print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%, Class Accuracy: {class_accuracy * 100:.2f}%")
        trainer_stage2.save_model("feature_vit-final.pth", "multi_view_model-final.pth")
        
    except (Exception, KeyboardInterrupt) as e:
        print(f"Stage 2 training interrupted. Error: {e}")
        trainer_stage2.save_model("feature_vit-s2-interrupted.pth", "multi_view-model-s2-interrupted.pth")

if __name__ == "__main__":
    print("Starting training process...")
    main()