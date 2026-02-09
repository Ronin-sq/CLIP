from model.clip import CLIP
# from dataloader import FlickrDataset
import torch 
import PIL.Image as Image
from transformers import AutoImageProcessor, AutoTokenizer

def run_inference():
    v_id = "google/vit-base-patch16-224-in21k"
    t_id = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(t_id)
    image_processor = AutoImageProcessor.from_pretrained(v_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载模型以及权重
    model = CLIP(
        vision_id=v_id,
        text_model_id=t_id,
        hidden_dim=512
    )
    model.load_state_dict(torch.load("clip_best.pth",map_location=device))
    
    model.to(device)
    model.eval()
    
    image = Image.open("test_1.png").convert("RGB")
    texts = [
        "a dog running on the grass",
        "a person riding a bicycle",
        "a sunset over the ocean",
        "a group of people in a meeting"
    ]
    
    image_inputs = image_processor(image, return_tensors="pt")
    
    text_inputs = tokenizer(
        text=texts,
        # images=image,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt")
    pixel_values = image_inputs['pixel_values'].to(device)
    input_ids = text_inputs['input_ids'].to(device)
    attention_mask = text_inputs['attention_mask'].to(device)    
    
    with torch.no_grad():
        # 调用你写的 forward 逻辑
        # 注意：这里我们只取 logits_per_image
        _, logits_per_image, _ = model(pixel_values, input_ids, attention_mask,return_loss=False)
        
        # 计算概率分布
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
    print("\n--- 推理结果 ---")
    # 找出概率最大的索引
    best_idx = probs.argmax()
    
    for i, (text, prob) in enumerate(zip(texts, probs)):
        mark = " ★" if i == best_idx else ""
        print(f"描述: {text:30} | 概率: {prob:.4f}{mark}")

    print(f"\n模型认为最匹配的描述是: {texts[best_idx]}")


if __name__ == "__main__":
    run_inference()