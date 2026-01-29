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
    from Evo_1.scripts.Evo1 import EVO1
    import torch.nn as nn
    import time

    import argparse
    parser = argparse.ArgumentParser(description="Train Evo-1")
    parser.add_argument("--state_dim", type=int, default=7)


    
    image_size=256    
    config_path="../Evo_1/dataset/config.yaml"
    max_samples=None
    horizon=50
    binarize_gripper=False
    use_augmentation=False
    num_workers=1
    batch_size=2
    
    with open(config_path, 'r') as f:
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
    
    prompts = batch["prompts"]
    images_batch = batch["images"]
    image_masks = batch["image_masks"]
    states = batch["states"].to(dtype=torch.bfloat16)
    actions_gt = batch["actions"].to(dtype=torch.bfloat16)
    action_mask = batch["action_mask"]
    state_mask = batch["state_mask"]
    embodiment_ids = batch["embodiment_ids"]
    fused_tokens_list = []
    
    
    dataset_config["state_dim"] = 24
    dataset_config["per_action_dim"] = 24
    dataset_config["horizon"] = 50
    model = EVO1(dataset_config)
    model.train()
    device = next(model.parameters()).device  # Get the device the model is on
    model = model.to(device)
    
    loss_fn = nn.MSELoss() 
    
    for prompt, images, image_mask in zip(prompts, images_batch, image_masks):
        fused = model.get_vl_embeddings(images=images, image_mask=image_mask, prompt=prompt, return_cls_only=False)
        fused_tokens_list.append(fused.to(dtype=torch.bfloat16))
            
    fused_tokens = torch.cat(fused_tokens_list, dim=0)
    # fused_tokens shape is [8, 1024, 896] where 8 is the batch size
    
    states = states.to(device=device, dtype=torch.bfloat16)
    # States shape is [8, 24]
    actions_gt = actions_gt.to(device=device, dtype=torch.bfloat16)
    # Actions shape is [8, 50, 24]. It is a sequence of 50 actions, each with 24 dimensions.
    action_mask = action_mask.to(device=device)
    # Action mask shape is [8, 50, 24]. For each of the 50 actions, only the first 7 dimensions will be predicted.
    embodiment_ids = embodiment_ids.to(device=device) if embodiment_ids is not None else None
    fused_tokens = fused_tokens.to(device=device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_velocity, noise = model(fused_tokens, state=states, actions_gt=actions_gt, action_mask=action_mask)
    # pred_velocity has shape [8, 1200].  It correspond to the 50 actions each with 24 dimensions.
    # noise has shape [8, 50, 24]
                
    target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)
            
    assert pred_velocity.shape == target_velocity.shape

    # if action_mask.sum() == 0:
    #     raise ValueError(f"[Step {step}] action_mask.sum() is 0! All actions are masked. "
    #                 f"This indicates a problem with the data or mask generation. "
    #                 f"action_mask shape: {action_mask.shape}, "
    #                 f"action_mask: {action_mask}")
            
    action_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity.dtype)
    pred_velocity_mask = pred_velocity * action_mask
    loss = loss_fn(pred_velocity_mask, target_velocity)
    # scale_factor = action_mask.numel() / (action_mask.sum() + 1e-8)
    # loss = loss * scale_factor

