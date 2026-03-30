from transformers import ViTModel, AutoModel
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CLIP(nn.Module):
    
    def  __init__(self, vision_id, text_model_id, hidden_dim):
        super(CLIP, self).__init__()
        self.hidden_dim = hidden_dim
        self.vision_encoder = ViTModel.from_pretrained(vision_id)
        self.text_encoder = AutoModel.from_pretrained(text_model_id)
        # 投影层
        
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size,self.hidden_dim,bias=True)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size,self.hidden_dim,bias=True)
        
        # 温度系数 初始值 ： ln(1/0.07)
        # self.logits_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.logits_scale = nn.Parameter(torch.ones([]) * 2.6593)
        nn.init.normal_(self.vision_proj.weight, std=self.vision_encoder.config.hidden_size**-0.5)
        nn.init.normal_(self.text_proj.weight, std=self.text_encoder.config.hidden_size**-0.5)
                
        
    def forward(self, pixel_values, inputs_id, attention_masked, return_loss):
        
        # .last_hidden_state返回的是[batch, seq_len, hidden_dim],我们取第一个token(cls)的输出作为整体的表示
        v_raw = self.vision_encoder(pixel_values).last_hidden_state[:,0,:]
        output = self.text_encoder(input_ids=inputs_id, attention_mask=attention_masked)
        t_raw = output.pooler_output
        
        
        # 投影到相同的维度
        v_proj = self.vision_proj(v_raw)
        t_proj = self.text_proj(t_raw)
        # L2归一化
        # p 代表范数的类型，dim代表归一化作用的维度
        # 假设 v_proj 的形状是 [batch_size, hidden_dim]
        # 那么 F.normalize(v_proj, p=2, dim=-1) 会对 hidden_dim 维度进行 L2 归一化
        # 使得每个样本的向量长度为1
        # v = F.normalize(v_proj, p=2, dim=-1)
        # t = F.normalize(t_proj, p=2, dim=-1)
        # 必须做这一步，否则 loss 基本不降
        v = v_proj / v_proj.norm(dim=-1, keepdim=True)
        t = t_proj / t_proj.norm(dim=-1, keepdim=True)
        # 计算相似度得分
        # 矩阵乘法 torch.matmul,矩阵点积运算
        # 图找文 文找图
        # 双向对称交叉熵损失
        logits_per_text = torch.matmul(t,v.t()) * self.logits_scale.exp()
        logits_per_image = logits_per_text.t()
        if not return_loss:
            # 推理模式：直接返回 logits，跳过 loss 计算
            return None, logits_per_image, logits_per_text
        # 创建标签,对角线上的元素
        labels = torch.arange(v_proj.size(0)).to(logits_per_text.device)
        
        loss_v = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_v + loss_t ) / 2
        
        return loss, logits_per_image, logits_per_text     


class SigLip(nn.Module):
    def  __init__(self, vision_id, text_model_id, hidden_dim):
        super(SigLip, self).__init__()
        self.hidden_dim = hidden_dim
        self.vision_encoder = ViTModel.from_pretrained(vision_id)
        self.text_encoder = AutoModel.from_pretrained(text_model_id)
        # 投影层
        
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size,self.hidden_dim,bias=True)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size,self.hidden_dim,bias=True)
        
        # 温度系数 初始值 ： ln(1/0.07)
        # self.logits_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.logits_scale = nn.Parameter(torch.ones([]) * 2.6593)
        self.log_t = nn.Parameter(torch.tensor(np.log(10.0)))
        self.log_b = nn.Parameter(torch.tensor(-10.0))

        nn.init.normal_(self.vision_proj.weight, std=self.vision_encoder.config.hidden_size**-0.5)
        nn.init.normal_(self.text_proj.weight, std=self.text_encoder.config.hidden_size**-0.5)


    def get_logits(self, image_embeds, text_embeds):
        t = torch.exp(self.log_t)
        b = self.log_b
        # 计算点积相似度矩阵 [Batch, Batch]
        logits = torch.matmul(image_embeds, text_embeds.t()) * t + b
        return logits      
        
    def forward(self, pixel_values, inputs_id, attention_masked, return_loss):
        
        # .last_hidden_state返回的是[batch, seq_len, hidden_dim],我们取第一个token(cls)的输出作为整体的表示
        v_raw = self.vision_encoder(pixel_values).last_hidden_state[:,0,:]
        output = self.text_encoder(input_ids=inputs_id, attention_mask=attention_masked)
        t_raw = output.pooler_output
        
        
        # 投影到相同的维度
        v_proj = self.vision_proj(v_raw)
        t_proj = self.text_proj(t_raw)
        # L2归一化
        # p 代表范数的类型，dim代表归一化作用的维度
        # 假设 v_proj 的形状是 [batch_size, hidden_dim]
        # 那么 F.normalize(v_proj, p=2, dim=-1) 会对 hidden_dim 维度进行 L2 归一化
        # 使得每个样本的向量长度为1
        # v = F.normalize(v_proj, p=2, dim=-1)
        # t = F.normalize(t_proj, p=2, dim=-1)
        # 必须做这一步，否则 loss 基本不降
        v = v_proj / v_proj.norm(dim=-1, keepdim=True)
        t = t_proj / t_proj.norm(dim=-1, keepdim=True)
        # 计算相似度得分
        # 矩阵乘法 torch.matmul,矩阵点积运算
        # 图找文 文找图
        # # 双向对称交叉熵损失
        # logits_per_text = torch.matmul(t,v.t()) * self.logits_scale.exp()
        # logits_per_image = logits_per_text.t()
        # if not return_loss:
        #     # 推理模式：直接返回 logits，跳过 loss 计算
        #     return None, logits_per_image, logits_per_text
        # # 创建标签,对角线上的元素
        # labels = torch.arange(v_proj.size(0)).to(logits_per_text.device)
        
        # loss_v = F.cross_entropy(logits_per_image, labels)
        # loss_t = F.cross_entropy(logits_per_text, labels)
        # loss = (loss_v + loss_t ) / 2

        logits = self.get_logits(v,t)

        n = logits.shape[0]
        # 构建标签矩阵：对角线为1，其余为-1
        labels = 2 * torch.eye(n, device=logits.device) - 1
        
        # Sigmoid 损失：-log(sigmoid(y * logits))
        # 这等价于正样本做靠近 1 的 BCE，负样本做靠近 0 的 BCE
        loss = -torch.nn.functional.logsigmoid(labels * logits).sum() / n
        
        return loss
        
                
        
        
        
if __name__ == '__main__':
    # 1. 初始化配置
    # 我们使用 base 级别的模型 ID 进行测试
    v_id = "google/vit-base-patch16-224-in21k"
    t_id = "roberta-base"
    h_dim = 512
    batch_size = 4  # 模拟 4 个图文对

    # 2. 实例化模型
    print("正在初始化模型...")
    model = CLIP(vision_id=v_id, text_model_id=t_id, hidden_dim=h_dim)
    
    # 3. 构造伪造输入数据 (模拟 CLIPProcessor 的输出)
    # 图像：[Batch, Channels, Height, Width] -> [4, 3, 224, 224]
    dummy_pixel_values = torch.randn(batch_size, 3, 224, 224)
    
    # 文本：[Batch, Seq_Len] -> [4, 32] (假设序列长度为 32)
    dummy_input_ids = torch.randint(0, 1000, (batch_size, 32))
    dummy_attention_mask = torch.ones((batch_size, 32))

    # 4. 执行前向传播
    print("开始前向传播测试...")
    try:
        loss, logits_v, logits_t = model(
            pixel_values=dummy_pixel_values,
            inputs_id=dummy_input_ids,
            attention_masked=dummy_attention_mask
        )
        
        # 5. 验证输出结果
        print("-" * 30)
        print(f"测试成功！")
        print(f"Loss 结果: {loss.item():.4f}")
        print(f"图像相似度矩阵维度: {logits_v.shape}")  # 应该是 [4, 4]
        print(f"文本相似度矩阵维度: {logits_t.shape}")  # 应该是 [4, 4]
        
        # 验证对角线逻辑
        # 初始状态下，随机输入的对角线不一定最大，但矩阵形状必须是对称的
        assert logits_v.shape == (batch_size, batch_size)
        assert not torch.isnan(loss), "错误：Loss 出现了 NaN 值！"
        
        print("维度校验完毕：符合预期。")
        
    except Exception as e:
        print(f"测试失败，错误信息: {e}")