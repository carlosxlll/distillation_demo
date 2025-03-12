import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from datetime import datetime

# 假设模型代码保存在 model.py 中
from model import Encoder, Decoder
from dataset import FaceDataset

# 训练配置
config = {
    "data_root": "./data",    # 数据集路径
    "batch_size": 1,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "save_dir": "./checkpoints",
    "save_input_path": "./checkpoints/input",
    "save_output_path": "./checkpoints/output",
    "save_encoder_path": "./checkpoints/encoder",
    "save_decoder_path": "./checkpoints/decoder",
    "save_optimizer_path": "./checkpoints/optimizer",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
def create_dir(config):
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["save_input_path"], exist_ok=True)
    os.makedirs(config["save_output_path"], exist_ok=True)
    os.makedirs(config["save_encoder_path"], exist_ok=True)
    os.makedirs(config["save_decoder_path"], exist_ok=True)
    os.makedirs(config["save_optimizer_path"], exist_ok=True)

def main():
    create_dir(config)
    # 准备数据集
    train_dataset = FaceDataset(config["data_root"])
    train_loader = DataLoader(train_dataset, 
                              batch_size=config["batch_size"],
                              shuffle=True,
                              num_workers=1)
    
    # 分别初始化 encoder 和 decoder，并移动到对应设备
    encoder = Encoder().to(config["device"])
    decoder = Decoder().to(config["device"])
    
    # 损失函数和优化器（优化器同时更新 encoder 和 decoder 的参数）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config["learning_rate"])
    
    # 创建保存目录
    os.makedirs(config["save_dir"], exist_ok=True)
    
    best_loss = float('inf')
    print(f"Start training on {config['device']}...")
    
    for epoch in range(config["num_epochs"]):
        encoder.train()
        decoder.train()
        running_loss = 0.0
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(config["device"])
            
            # 前向传播：先通过 encoder，再通过 decoder
            latent = encoder(images)
            outputs = decoder(latent)
            
            # 计算损失
            loss = criterion(outputs, images)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 每100个 batch 打印状态
            if batch_idx % 100 == 99:
                avg_loss = running_loss / 100
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Epoch [{epoch+1}/{config['num_epochs']}], "
                      f"Batch {batch_idx+1}, Loss: {avg_loss:.4f}")
                running_loss = 0.0
            
            # 保存可视化结果（保存当前 batch 的第一张图像及其重建结果）
            if batch_idx == 0:
                input_image = images[0].cpu().detach()
                output_image = outputs[0].cpu().detach()
                # 转换为 PIL Image
                input_pil = transforms.ToPILImage()(input_image)
                output_pil = transforms.ToPILImage()(output_image)
                input_pil.save(os.path.join(config["save_input_path"], f"input_{epoch}.png"))
                output_pil.save(os.path.join(config["save_output_path"], f"output_{epoch}.png"))
        
        # 这里直接使用最后一个 batch 的 loss 作为 epoch_loss（也可以计算全 epoch 平均 loss）
        # 保存检查点
        epoch_loss = loss.item()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(encoder.state_dict(), os.path.join(config["save_encoder_path"], f"best_encoder.pth"))
            torch.save(decoder.state_dict(), os.path.join(config["save_decoder_path"], f"best_decoder.pth"))
            torch.save(optimizer.state_dict(), os.path.join(config["save_optimizer_path"], f"best_optimizer.pth"))

        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] completed, Loss: {epoch_loss:.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
