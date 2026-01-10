# model/internvl3/internvl3_embedder.py
import torch
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers import GenerationConfig
from torchvision.transforms.functional import to_pil_image
from typing import Union, List
from torch import nn
import logging

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# === Image Transformations ===
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

# === Aspect Ratio Handling ===
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

class InternVL3Embedder(nn.Module):
    def __init__(self, model_name="OpenGVLab/InternVL3-1B", image_size=448, device="cuda"):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.max_text_length = 1024  # InternVL3 supports up to 1024 tokens
        self.transform = build_transform(image_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=True,
            low_cpu_mem_usage=True,
            _fast_init=False,
        ).to(self.device) 
        
        if hasattr(self.model.language_model, 'model'):
            layers = self.model.language_model.model.layers

        else:
            layers = self.model.language_model.layers
        layers = layers[:14]

        if hasattr(self.model.language_model, 'model'):
            self.model.language_model.model.layers = torch.nn.ModuleList(layers)
        else:
            self.model.language_model.layers = torch.nn.ModuleList(layers)
        self.model.language_model.lm_head = torch.nn.Identity()

        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
            self.model.vision_model.encoder.gradient_checkpointing = False
        

    def _preprocess_images(
        self,
        image_tensors: List[Union[Image.Image, torch.Tensor]]
    ) -> (torch.Tensor, List[int]):

        #Image tensors shape: 3x3x448x448
        # Steps:
        # 1. Convert the images tensor to PIL images
        # 2. Split the images into tiles (list of PIL images)
        # 3. Normalize the tiles and convert them to tensors
        # 4. Concatenate the tiles into a single tensor recreating the original tensor 

        pixel_values_list = []
        for i, image in enumerate(image_tensors):
            if isinstance(image, torch.Tensor):
                image = to_pil_image(image)
            tiles = dynamic_preprocess(image, image_size=self.image_size)
            # Normalize the images
            tile_tensors = torch.stack([self.transform(t) for t in tiles])  # (T_i, 3, 448, 448)
            pixel_values_list.append(tile_tensors)

        pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype=torch.bfloat16, device=self.device)
        num_tiles_list = [pv.shape[0] for pv in pixel_values_list]
        # pixel_values image with shape: 3x3x448x448 and normalized
        # num_tiles_list: [1, 1, 1]. Number of tiles for each image in the tensor pixel_values
        return pixel_values, num_tiles_list

    def _build_multimodal_prompt(
        self,
        num_tiles_list: List[int],
        text_prompt: str
    ) -> str:

        prompt = ''
        for i in range(len(num_tiles_list)):
            prompt += f"Image-{i+1}: <image>\n"
        prompt += text_prompt.strip()

        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"

        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        # self.img_context_token_id is an integer token id: 151667
        # Adding to the prompt a number of <IMG_CONTEXT> tokens equal to the number of tiles for each image
        # multiplied by the number of convolutional filters in the vision model.
        # Example:
        # num_tiles_list = [1,1,1]
        # self.model.num_image_token = 256
        # image_tokens = "<img><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT>"
        for tile_count in num_tiles_list:
            token_count = self.model.num_image_token * tile_count
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
            prompt = prompt.replace("<image>", image_tokens, 1)

        return prompt
    
    def _prepare_and_fuse_embeddings(
        self,
        prompt: str,
        vit_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        num_tiles_list: List[int]
    ) -> (torch.Tensor, torch.Tensor):
   
        untruncated_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        true_sequence_length = untruncated_ids.shape[1]

        if true_sequence_length > self.max_text_length:
            print("\n" + "="*80)
            print(f" WARNING: Input prompt was TRUNCATED!")
            print(f"   - Max Length Allowed    : {self.max_text_length}")
            print(f"   - Actual Length      : {true_sequence_length}")
            print(f"   - Truncated Prompt (first 100 chars): '{prompt[:100]}...'")
            print("="*80 + "\n")

        model_inputs = self.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_text_length).to(self.device)
        input_ids = model_inputs["input_ids"]
        logger.debug("Input ids shape: %s", input_ids.shape)
        attention_mask = model_inputs["attention_mask"]
       
        img_token_mask = (input_ids == self.img_context_token_id)
     
        img_token_locations = torch.where(img_token_mask)[1]


        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)

        selected = (input_ids == self.img_context_token_id)

            
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

 
        tokens_per_tile = self.model.num_image_token 
 
        torch.set_printoptions(profile="full", threshold=float('inf'))
   
        torch.set_printoptions(profile="default")
        current_token_idx = 0
        for i in range(len(image_mask)):
           
            num_tiles_for_this_image = num_tiles_list[i]
            num_tokens_for_this_image = num_tiles_for_this_image * tokens_per_tile
       
            if not image_mask[i]:
                
                start_idx = img_token_locations[current_token_idx]
                end_idx = start_idx + num_tokens_for_this_image
               
                attention_mask[0, start_idx:end_idx] = 0
    
            current_token_idx += num_tokens_for_this_image

        input_embeds = input_embeds.reshape(B, N, C)
    
        torch.set_printoptions(profile="full", threshold=float('inf'))
     
        torch.set_printoptions(profile="default")
        return input_embeds, attention_mask


    def get_fused_image_text_embedding_from_tensor_images(
        self,
        image_tensors: list[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        text_prompt: str,
        return_cls_only: bool = True,
    ):

   
        pixel_values, num_tiles_list = self._preprocess_images(image_tensors)
        # pixel_values image with shape: 3x3x448x448 and normalized
        # num_tiles_list = [1,1,1]  This is the number of tiles for each image in the tensor pixel_values
        if pixel_values.shape[0] == 0:           
            print("Warning: No valid images to process after masking.")
        
        # pixel_values image with shape: 3x3x448x448 and normalized
        vit_embeds = self.model.extract_feature(pixel_values)
        # Shape of vit_embeds: [3, 256, 896]. This is the embedding of the image tiles.
        
        fused_embeds = vit_embeds
        # num_tiles_list = [1,1,1]  
        # text_prompt = 'pick up the black bowl from table center...' 
        prompt = self._build_multimodal_prompt(num_tiles_list, text_prompt)
        # prompt: 'Image-1: <img><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT>pick up the black bowl from table center...'
        inputs_embeds, attention_mask = self._prepare_and_fuse_embeddings(prompt, fused_embeds, image_mask, num_tiles_list)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        fused_hidden = outputs.hidden_states[-1].to(torch.float32)

        return fused_hidden[:, 0, :] if return_cls_only else fused_hidden

if __name__ == "__main__":
    import sys
    import os
    # Add project root to Python path (parent of dataset directory)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    
    from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset
    # Add project root to Python path   
    import yaml
    from torch.utils.data import DataLoader


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


    image_size=448    
    dataset_config_path="../../dataset/config.yaml"
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
    
    prompts = batch["prompts"]
    images_batch = batch["images"]
    image_masks = batch["image_masks"]
            
    embedder = InternVL3Embedder(image_size=image_size)        

    for idx, (prompt, images, image_mask) in enumerate(zip(prompts, images_batch, image_masks)):
        logger.info("Processing sample %d", idx)
        logger.debug("Images shape: %s", images.shape) # 3x3x448x448
        logger.debug("Prompt: %s", prompt) # "pick up the red ball"
        logger.debug("Image mask shape: %s | values: %s", image_mask.shape, image_mask) # (3,) | [True, True, False]

        fused = embedder.get_fused_image_text_embedding_from_tensor_images(
            images, image_mask, prompt
        )
        