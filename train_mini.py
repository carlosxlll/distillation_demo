import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from datetime import datetime

# 导入原模型和mini模型
from model import Encoder, Decoder
from model_mini import Encoder_mini
from dataset import FaceDataset

# 训练配置
config = {
    "data_root": "./data",    # 数据集路径
    "batch_size": 1,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "teacher_encoder_pth": "./checkpoints/encoder/best_encoder.pth",
    "decoder_pth": "./checkpoints/decoder/best_decoder.pth",
    "teacher_r" : 0.2,
    "save_dir": "./checkpoints_mini",
    "save_input_path": "./checkpoints_mini/input",
    "save_output_path": "./checkpoints_mini/output",
    "save_encoder_path": "./checkpoints_mini/encoder",
    "save_decoder_path": "./checkpoints_mini/decoder",
    "save_optimizer_path": "./checkpoints_mini/optimizer",
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
    
    # 初始化模型并移动到对应设备
    encoder = Encoder().to(config["device"])
    decoder = Decoder().to(config["device"])
    encoder.load_state_dict(torch.load(config["teacher_encoder_pth"]))
    decoder.load_state_dict(torch.load(config["decoder_pth"]))
    mini_encoder = Encoder_mini().to(config["device"])
    
    # 冻结原decoder和原encoder
    for param in decoder.parameters():
        param.requires_grad = False
    for param in encoder.parameters():
        param.requires_grad = False
    
    # 只优化mini_encoder的参数
    optimizer = optim.Adam(mini_encoder.parameters(), lr=config["learning_rate"])
    
    # 损失函数
    mse_loss = nn.MSELoss()
    
    # 创建保存目录
    os.makedirs(config["save_dir"], exist_ok=True)
    
    best_loss = float('inf')
    print(f"Start training on {config['device']}...")
    
    for epoch in range(config["num_epochs"]):
        mini_encoder.train()
        decoder.eval()
        encoder.eval()
        
        running_loss = 0.0
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(config["device"])
            
            # 原encoder和mini encoder的输出
            with torch.no_grad():
                original_latent = encoder(images)
                
            # mini encoder的输出
            mini_latent = mini_encoder(images)
            
            # 通过原decoder重建
            outputs = decoder(mini_latent)
            
            # 计算损失：重建损失 + 特征空间损失
            reconstruction_loss = mse_loss(outputs, images)
            feature_loss = mse_loss(mini_latent, original_latent)
            
            # 总损失
            loss = config["teacher_r"] * feature_loss +(1 - config["teacher_r"]) * reconstruction_loss
            
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
            
            # 保存可视化结果
            if batch_idx == 0:
                input_image = images[0].cpu().detach()
                output_image = outputs[0].cpu().detach()
                # 转换为 PIL Image
                input_pil = transforms.ToPILImage()(input_image)
                output_pil = transforms.ToPILImage()(output_image)
                input_pil.save(os.path.join(config["save_input_path"], f"input_{epoch}.png"))
                output_pil.save(os.path.join(config["save_output_path"], f"output_{epoch}.png"))
        
        # 保存检查点
        epoch_loss = loss.item()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(encoder.state_dict(), os.path.join(config["save_encoder_path"], f"best_encoder.pth"))
            torch.save(decoder.state_dict(), os.path.join(config["save_decoder_path"], f"best_decoder.pth"))
            torch.save(optimizer.state_dict(), os.path.join(config["save_optimizer_path"], f"best_optimizer.pth"))

        print(f"Epoch [{epoch+1}/{config['num_epochs']}] completed, Loss: {epoch_loss:.4f}")
    
    print("蒸馏训练完成!")

if __name__ == "__main__":
    main()
