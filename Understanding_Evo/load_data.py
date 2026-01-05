if __name__ == "__main__":
    import sys
    import os
    # Add project root to Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    from Evo_1.dataset.lerobot_dataset_pretrain_mp import LeRobotDataset
    import yaml
    
    batch_size=16
    image_size=448     
    dataset_config_path="../Evo_1/dataset/config.yaml"
    max_samples=None
    horizon=50
    binarize_gripper=False
    use_augmentation=False
    num_workers=1
    batch_size=8
    
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
        
    dataset = LeRobotDataset(
                    config=dataset_config,
                    image_size=image_size,
                    max_samples_per_file=max_samples,
                    action_horizon=horizon,
                    binarize_gripper=binarize_gripper,
                    use_augmentation=use_augmentation
                )
    
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    
    # for batch in tqdm(dataloader, desc="Training", disable=not accelerator.is_main_process):
    #         if step >= max_steps:
    #             break
    #         prompts = batch["prompts"]
    #         images_batch = batch["images"]
    #         image_masks = batch["image_masks"]
    #         states = batch["states"].to(dtype=torch.bfloat16)
    #         actions_gt = batch["actions"].to(dtype=torch.bfloat16)
    #         action_mask = batch["action_mask"]
    #         state_mask = batch["state_mask"]
    #         embodiment_ids = batch["embodiment_ids"]
    #         fused_tokens_list = []