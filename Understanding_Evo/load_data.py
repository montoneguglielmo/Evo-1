def custom_collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    images = [item["images"] for item in batch]
    states = torch.stack([item["state"] for item in batch], dim=0)
    actions = torch.stack([item["action"] for item in batch], dim=0)
    action_mask = torch.stack([item["action_mask"] for item in batch], dim=0)
    image_masks = torch.stack([item["image_mask"] for item in batch], dim=0)
    state_mask = torch.stack([item["state_mask"] for item in batch], dim=0)
    embodiment_ids = torch.stack([item["embodiment_id"] for item in batch], dim=0)

    return {
        "prompts": prompts,
        "images": images,
        "states": states,
        "actions": actions,
        "action_mask": action_mask,
        "state_mask": state_mask,
        "image_masks": image_masks,
        "embodiment_ids": embodiment_ids
    }



if __name__ == "__main__":
    import sys
    import os
    # Add project root to Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    from Evo_1.dataset.lerobot_dataset_pretrain_mp import LeRobotDataset
    import yaml
    from torch.utils.data import DataLoader
    import torch


    
    image_size=256    
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
    
    # Inspect a single batch
    batch = next(iter(dataloader))
    
    print("=" * 50)
    print("BATCH INSPECTION")
    print("=" * 50)
    print(f"Batch keys: {batch.keys()}")
    print(f"\nPrompts (type: {type(batch['prompts'])}, length: {len(batch['prompts'])})")
    if len(batch['prompts']) > 0:
        print(f"  First prompt: {batch['prompts'][0][:100]}..." if len(batch['prompts'][0]) > 100 else f"  First prompt: {batch['prompts'][0]}")
        print(f"  Last prompt: {batch['prompts'][-1][:100]}..." if len(batch['prompts'][-1]) > 100 else f"  Last prompt: {batch['prompts'][-1]}")
    
    # Images are always reshaped to 448x448
    # There are always 3 views, but the last one is padded with zeros in the case of the Libero Spatial Lerobot dataset
    print(f"\nImages (type: {type(batch['images'])}, length: {len(batch['images'])})")
    if len(batch['images']) > 0:
        print(f"  First image shape: {batch['images'][0].shape if hasattr(batch['images'][0], 'shape') else 'N/A'}")
    
    print(f"\nStates shape: {batch['states'].shape}, dtype: {batch['states'].dtype}")
    # The action shape is (batch_size, horizon, action_dim)
    print(f"Actions shape: {batch['actions'].shape}, dtype: {batch['actions'].dtype}")
    # The action on Frank robot is actully a 7D vector. The action mask will be used to mask the rest of the action vector.
    print(f"Action mask shape: {batch['action_mask'].shape}, dtype: {batch['action_mask'].dtype}")
    
    # The state mask will mask the state vector leaving only the first 8 elements visible. The LIBERO Franka robot has 8 stae dimensions.
    print(f"State mask shape: {batch['state_mask'].shape}, dtype: {batch['state_mask'].dtype}")
    
    # The 3rd image mask is always False. This is because the LIBERO Spatial Lerobot dataset has 2 views.
    print(f"Image masks shape: {batch['image_masks'].shape}, dtype: {batch['image_masks'].dtype}")
    print(f"Embodiment IDs shape: {batch['embodiment_ids'].shape}, dtype: {batch['embodiment_ids'].dtype}")
    print(f"Embodiment IDs values: {batch['embodiment_ids']}")
    print("=" * 50)