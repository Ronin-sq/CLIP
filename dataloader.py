import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import datasets
import PIL.Image as Image


class FlickrDataset(Dataset):
    
    def __init__(self,hf_dataset, tokenizer, image_processor, max_length=128):
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dataset = hf_dataset
        self.max_length = max_length
        
    def __len__(self):
        
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        item = self.dataset[idx]
        
        # image = Image.open(item["image"]).convert('RGB')
        image = item["image"].convert("RGB")
        # caption = item['caption'][0] if isinstance(item['caption'], list) else item['caption']
        # print(item)
        import random
        caption = random.choice(item['caption']) if isinstance(item['caption'], list) else item['caption']
        if isinstance(caption, dict): # 有时候 flickr 数据集里 sentence 是个 dict 包含 raw 字段
            caption = caption['raw']
        
        image_inputs = self.image_processor(image, return_tensors="pt")
        
        text_inputs = self.tokenizer(
            text=caption,
            # images=image,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # print(f"text_inputs:{text_inputs}")
        return {
            "pixel_values": image_inputs['pixel_values'].squeeze(0),  # 去掉多余的batch维度
            "input_ids": text_inputs['input_ids'].squeeze(0),
            "attention_mask": text_inputs['attention_mask'].squeeze(0),
            "caption": item["caption"]
        }
            
            
if __name__=="__main__":
    from datasets import load_dataset
    from transformers import CLIPProcessor
    # import os

    # 1. 解决网络问题（国内加速）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # 2. 加载在线数据集
    print("正在从 Hugging Face 加载数据集...")
    # 这是一个更现代、不需要脚本的版本
    raw_ds = load_dataset("lmms-lab/flickr30k", split="test")

    # 3. 切分训练集和验证集
    split_ds = raw_ds.train_test_split(test_size=0.1, seed=42)
    train_raw = split_ds["train"]
    val_raw = split_ds["test"]

    # 4. 初始化 Processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 5. 封装成 PyTorch Dataset
    train_dataset = FlickrDataset(train_raw, processor)
    val_dataset = FlickrDataset(val_raw, processor)

    # 6. 放入 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    print(f"数据准备完毕！训练 Batch 数量: {len(train_loader)}")