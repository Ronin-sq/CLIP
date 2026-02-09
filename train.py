import torch
from torch import optim
from datasets import load_dataset
from model.clip import CLIP
from dataloader import FlickrDataset
# from transformers import CLIPProcessor
from transformers import AutoImageProcessor, AutoTokenizer
from torch.utils.data import DataLoader
import os
from tqdm import tqdm 
import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses, save_path='loss_curve.png'):
    """
    绘制训练和验证损失对比图并保存
    :param train_losses: 训练集 loss 列表
    :param val_losses: 验证集 loss 列表
    :param save_path: 图片保存路径
    """
    
    # 确保两个列表长度一致（通常以 epoch 为单位）
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # 2. 绘制训练 Loss (通常用蓝色实线)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    
    # 3. 绘制验证 Loss (通常用红色虚线，便于区分)
    plt.plot(epochs, val_losses, 'r--', label='Validation Loss')
    
    # 4. 添加图例、标题和坐标轴
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()  # 关键：显示左上角的图例，不然分不清哪条线是哪个
    plt.grid(True) # 加网格方便看数值
    
    # 5. 保存图片而不是 show
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 关闭画布，释放内存（在循环中画图时这步很重要）
    plt.close()
    print(f"图表已保存至: {save_path}")


def dataloader(v_id,t_id):
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
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained(t_id)
    image_processor = AutoImageProcessor.from_pretrained(v_id)
    
    
    # 5. 封装成 PyTorch Dataset
    train_dataset = FlickrDataset(train_raw, tokenizer, image_processor)
    val_dataset = FlickrDataset(val_raw, tokenizer, image_processor)

    # 6. 放入 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataset = DataLoader(val_dataset,batch_size=32, shuffle=True, num_workers=4)
    
    return train_loader, val_dataset


def train(model, train_loader, optimizer,device,epoch):
    model.train()
    running_loss = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", leave=True)
    for i, (batch) in pbar:
    # for batch in train_loader:
        optimizer.zero_grad()
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        loss, loss_v, loss_t = model(pixel_values,input_ids,attention_masks,return_loss=True)

        loss.backward()
        # print(f"model.logits_scale.grad:{model.logits_scale.grad}")s
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
        # print(f"Logits Max: {loss_v.max().item():.2f}") # 正常应在 10~30 之间
        # print(f"Logits Min: {loss_v.min().item():.2f}")
        # print(f"Logit Scale: {model.logit_scale.exp().item():.2f}")
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{running_loss/(i+1):.4f}",
            'Logits Max': f"{loss_v.max().item():.2f}",
            'Logits Min': f"{loss_v.min().item():.2f}",
            "model.logits_scale.grad":f"{model.logits_scale.grad}"
        })
    return running_loss/len(train_loader)

def evaluate(model, val_loader,device):
    model.eval()
    # correct = 0
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            # 1. 数据搬运
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 2. 前向传播
            loss, logits_per_image, _ = model(pixel_values, input_ids, attention_mask, return_loss=True)
            
            # 3. 统计 Loss
            total_loss += loss.item()
            
            # 4. 计算准确率 (Image-to-Text)
            # 每一行的最大值索引应该对应 labels [0, 1, 2, ..., n]
            batch_size = pixel_values.size(0)
            labels = torch.arange(batch_size).to(device)
            
            # 预测值：每一行相似度最高的索引
            preds = torch.argmax(logits_per_image, dim=-1)
            correct += (preds == labels).sum().item()
            total += batch_size
            
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, 100 * accuracy    
    
    

def main():
    
    EPOCHES = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # batch_size = 32
    v_id = "google/vit-base-patch16-224-in21k"
    t_id = "bert-base-uncased"
    train_loader, val_loader = dataloader(v_id=v_id,t_id=t_id)
    h_dim = 512
    lr = 1e-4
    model = CLIP(vision_id=v_id, text_model_id=t_id, hidden_dim=h_dim)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHES)
    train_loss_epoch = []
    val_loss_epoch = []
    best_acc = 0
    for epoch in range(EPOCHES):
        train_loss = train(model=model, train_loader=train_loader, optimizer=optimizer,device=device,epoch= epoch)

        val_loss, val_acc = evaluate(model, val_loader,device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        print(f"Epoch [{epoch+1:03d}/{EPOCHES}] "
              f"| Train_Loss: {train_loss:.4f} "
              f"| Val_Loss: {val_loss:.4f}"
              f"| Acc: {val_acc:.3f}% "
              f"| LR: {current_lr:.6f}")
        # writer.add_scalar("Loss/train",train_loss,epoch)
        # writer.add_scalar("Loss/val",val_loss,epoch)
        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)
        
        if val_loss > train_loss * 1.5:
            print("Warning: 可能出现明显的过拟合")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "clip_best.pth")
            print(f"--> Best model saved with Acc: {best_acc:.2f}%")
    print("训练结束")
    plot_losses(train_loss_epoch,val_loss_epoch)
        
        
if __name__=='__main__':
    main()

