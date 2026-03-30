import torch
from torch import optim
from datasets import load_dataset
from model.clip import SigLip
from dataloader import FlickrDataset
from transformers import AutoImageProcessor, AutoTokenizer
from torch.utils.data import DataLoader
import os
from tqdm import tqdm 
import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses, save_path='loss_curve_siglip.png'):
    """
    绘制训练和验证损失对比图并保存
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r--', label='Validation Loss')
    plt.title('Training and Validation Loss (SigLIP)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存至: {save_path}")


def dataloader(v_id,t_id):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("正在从 Hugging Face 加载数据集...")
    raw_ds = load_dataset("lmms-lab/flickr30k", split="test")

    split_ds = raw_ds.train_test_split(test_size=0.1, seed=42)
    train_raw = split_ds["train"]
    val_raw = split_ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(t_id)
    image_processor = AutoImageProcessor.from_pretrained(v_id)
    
    train_dataset = FlickrDataset(train_raw, tokenizer, image_processor)
    val_dataset = FlickrDataset(val_raw, tokenizer, image_processor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset,batch_size=32, shuffle=False, num_workers=4)
    
    return train_loader, val_loader


def train(model, train_loader, optimizer, device, epoch):
    model.train()
    running_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        
        # SigLip的forward当前只返回了loss，如果修改过返回了3个值，请按需调整
        out = model(pixel_values, input_ids, attention_masks, return_loss=True)
        if isinstance(out, tuple):
            loss = out[0]
        else:
            loss = out

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
        
        log_t_val = model.log_t.item()
        log_b_val = model.log_b.item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{running_loss/(i+1):.4f}",
            'log_t': f"{log_t_val:.2f}",
            'log_b': f"{log_b_val:.2f}"
        })
    return running_loss / len(train_loader)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 由于验证集主要需要准确率，我们需要获取 logits
            # 这里我们手动调用 model.get_logits 或模拟前向获取 logits
            
            # 提取特征
            v_raw = model.vision_encoder(pixel_values).last_hidden_state[:,0,:]
            output = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            t_raw = output.pooler_output
            
            v_proj = model.vision_proj(v_raw)
            t_proj = model.text_proj(t_raw)
            
            v = v_proj / v_proj.norm(dim=-1, keepdim=True)
            t = t_proj / t_proj.norm(dim=-1, keepdim=True)
            
            logits = model.get_logits(v, t)
            
            # 计算 loss
            n = logits.shape[0]
            labels_matrix = 2 * torch.eye(n, device=logits.device) - 1
            loss = -torch.nn.functional.logsigmoid(labels_matrix * logits).sum() / n
            total_loss += loss.item()
            
            # 计算准确率 (Image-to-Text)
            batch_size = pixel_values.size(0)
            labels = torch.arange(batch_size).to(device)
            
            # 预测值：每一行相似度最高的索引
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += batch_size
            
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, 100 * accuracy    
    

def main():
    EPOCHES = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    v_id = "google/vit-base-patch16-224-in21k"
    t_id = "bert-base-uncased"
    train_loader, val_loader = dataloader(v_id=v_id,t_id=t_id)
    h_dim = 512
    lr = 1e-4
    
    print("正在初始化 SigLip 模型...")
    model = SigLip(vision_id=v_id, text_model_id=t_id, hidden_dim=h_dim)
    model.to(device)
    
    # 针对 log_t 和 log_b 可以设置不同的学习率
    optimizer = optim.AdamW([
        {'params': model.vision_encoder.parameters()},
        {'params': model.text_encoder.parameters()},
        {'params': model.vision_proj.parameters()},
        {'params': model.text_proj.parameters()},
        {'params': [model.log_t, model.log_b], 'lr': lr * 10} # 温度参数通常使用更大的学习率
    ], lr=lr)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHES)
    train_loss_epoch = []
    val_loss_epoch = []
    best_acc = 0
    
    for epoch in range(1, EPOCHES + 1):
        train_loss = train(model=model, train_loader=train_loader, optimizer=optimizer, device=device, epoch=epoch)
        val_loss, val_acc = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        print(f"Epoch [{epoch:03d}/{EPOCHES}] "
              f"| Train_Loss: {train_loss:.4f} "
              f"| Val_Loss: {val_loss:.4f} "
              f"| Acc: {val_acc:.3f}% "
              f"| LR: {current_lr:.6f}")
              
        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)
        
        if val_loss > train_loss * 1.5:
            print("Warning: 可能出现明显的过拟合")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "siglip_best.pth")
            print(f"--> Best model saved with Acc: {best_acc:.2f}%")
            
    print("训练结束")
    plot_losses(train_loss_epoch,val_loss_epoch)
        
        
if __name__=='__main__':
    main()
