import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.models as models

from tqdm import tqdm
from PIL import Image
import numpy as np

from models.base import BaseModel
from utils.post_processing import enhance_color, enhance_contrast

# 添加PSNR感知损失
class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val
        
    def forward(self, x, y):
        mse = F.mse_loss(x, y)
        psnr = 10 * torch.log10((self.max_val ** 2) / mse)
        # 转换为损失函数（值越小越好）
        loss = 1.0 / psnr
        return loss

# 添加可学习的后处理模块，用于自适应增强
class AdaptiveEnhancementModule(nn.Module):
    def __init__(self):
        super(AdaptiveEnhancementModule, self).__init__()
        # 输入通道为6：3通道RGB原始输入 + 3通道网络输出
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # 预测增强参数（对比度和饱和度）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 2)  # 2个参数：对比度和饱和度
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, original, enhanced):
        # 特征提取
        x = torch.cat([original, enhanced], dim=1)
        x = self.relu(self.conv1(x))
        features = self.relu(self.conv2(x))
        residual = self.conv3(features)
        
        # 参数预测
        params = self.global_pool(features).squeeze(-1).squeeze(-1)
        params = self.relu(self.fc1(params))
        params = self.sigmoid(self.fc2(params))
        
        # 参数缩放到合适的范围
        contrast_factor = 0.5 + params[:, 0:1] * 1.5  # 范围：0.5-2.0
        saturation_factor = 0.5 + params[:, 1:2] * 1.5  # 范围：0.5-2.0
        
        # 应用自适应增强
        enhanced = enhanced + residual
        
        # 应用对比度增强
        mean = enhanced.mean(dim=[2, 3], keepdim=True)
        enhanced = (enhanced - mean) * contrast_factor.view(-1, 1, 1, 1) + mean
        
        # 应用饱和度增强
        luminance = 0.299 * enhanced[:, 0:1] + 0.587 * enhanced[:, 1:2] + 0.114 * enhanced[:, 2:3]
        enhanced = luminance + saturation_factor.view(-1, 1, 1, 1) * (enhanced - luminance)
        
        # 确保输出在有效范围内
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        
        return enhanced


class Model(BaseModel):
    def __init__(self, network, **kwargs):
        """Must to init BaseModel with kwargs."""
        super(Model, self).__init__(**kwargs)

        self.network = network.to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        # 添加增强的损失函数组件
        self.l1_loss = nn.L1Loss()
        self.psnr_loss = PSNRLoss()
        
        # 添加自适应后处理模块
        self.adaptive_enhancement = AdaptiveEnhancementModule().to(self.device)
        # 将后处理模块参数添加到优化器
        self.optimizer.add_param_group({'params': self.adaptive_enhancement.parameters()})
        
        # 注：现在使用BaseModel中的metrics_path，不需要创建metrics_dir
        # self.metrics_dir = os.path.join(self.model_path, 'metrics')
        # os.makedirs(self.metrics_dir, exist_ok=True)

    def enhanced_loss(self, outputs, targets):
        """更强大的复合损失函数，针对PSNR优化"""
        # 基础MSE损失（与PSNR直接相关）
        mse_loss = self.criterion(outputs, targets)
        
        # L1损失（减少噪声，提高细节）
        l1_loss = self.l1_loss(outputs, targets) * 0.5
        
        # PSNR感知损失（直接优化PSNR）
        psnr_loss_val = self.psnr_loss(outputs, targets) * 10.0
        
        # 感知损失（保持高级特征一致性）
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:20].to(self.device)
        for param in vgg.parameters():
            param.requires_grad = False
        perceptual_loss = F.mse_loss(vgg(outputs), vgg(targets)) * 0.1
        
        # 梯度损失（保持边缘锐利度）
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(self.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(self.device)
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        # 计算梯度
        padded_outputs = F.pad(outputs, (1, 1, 1, 1), 'replicate')
        padded_targets = F.pad(targets, (1, 1, 1, 1), 'replicate')
        
        outputs_grad_x = F.conv2d(padded_outputs, sobel_x, groups=3)
        outputs_grad_y = F.conv2d(padded_outputs, sobel_y, groups=3)
        targets_grad_x = F.conv2d(padded_targets, sobel_x, groups=3)
        targets_grad_y = F.conv2d(padded_targets, sobel_y, groups=3)
        
        gradient_loss = (F.mse_loss(outputs_grad_x, targets_grad_x) + 
                         F.mse_loss(outputs_grad_y, targets_grad_y)) * 0.5
        
        # 组合所有损失
        total_loss = mse_loss + l1_loss + psnr_loss_val + perceptual_loss + gradient_loss
        
        return total_loss

    def composite_loss(self, outputs, targets):
        """For backward compatibility, maintains the old composite loss function"""
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:20].to(self.device)
        perceptual_loss_weight = 0.25
        loss = self.criterion(outputs, targets)
        perceptual_loss = perceptual_loss_weight * F.mse_loss(vgg(outputs), vgg(targets))

        return loss + perceptual_loss
    
    def apply_adaptive_enhancement(self, inputs, outputs):
        """应用自适应后处理增强"""
        return self.adaptive_enhancement(inputs, outputs)
    
    def generate_output_images(self, outputs, save_dir):
        """Generates and saves output images to the specified directory."""
        os.makedirs(save_dir, exist_ok=True)
        for i, output_image in enumerate(outputs):
            output_image = output_image.detach().cpu().permute(1, 2, 0).numpy()
            output_image = (output_image * 255).astype(np.uint8)
            output_image = Image.fromarray(output_image)

            output_path = os.path.join(save_dir, f'output_{i + 1}.png')

            output_image.save(output_path)
        print(f'{len(outputs)} output images generated and saved to {save_dir}')


    def train_step(self):
        """Trains the model."""
        train_losses = np.zeros(self.epoch)
        best_loss = float('inf')
        self.network.to(self.device)
        
        # 创建训练指标记录
        train_metrics = {
            'epoch': [],
            'loss': []
        }

        for epoch in range(self.epoch):
            train_loss = 0.0
            dataloader_iter = tqdm(
                self.dataloader, desc=f'Training... Epoch: {epoch + 1}/{self.epoch}', total=len(self.dataloader))
            for inputs, targets in dataloader_iter:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.network(inputs)
                
                # 应用自适应后处理模块
                enhanced_outputs = self.apply_adaptive_enhancement(inputs, outputs)
                
                # 使用增强版损失函数
                loss = self.enhanced_loss(enhanced_outputs, targets)
                train_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                dataloader_iter.set_postfix({'loss': loss.item()})

            train_loss = train_loss / len(self.dataloader)

            if train_loss < best_loss:
                best_loss = train_loss
                self.save_model(self.network)
                # 同时保存后处理模块
                torch.save(self.adaptive_enhancement.state_dict(), 
                          os.path.join(self.model_path, 'adaptive_enhancement.pt'))

            train_losses[epoch] = train_loss
            
            # 记录当前轮次的训练指标
            train_metrics['epoch'].append(epoch + 1)
            train_metrics['loss'].append(float(train_loss))
            
            # 每轮保存一次训练指标，使用metrics_path
            with open(os.path.join(self.metrics_path, 'train_metrics.json'), 'w') as f:
                json.dump(train_metrics, f, indent=4)

            print(f"Epoch [{epoch + 1}/{self.epoch}] Train Loss: {train_loss:.4f}")


    def test_step(self):
        """Test the model."""
        path = os.path.join(self.model_path, self.model_name)
        self.network.load_state_dict(torch.load(path))
        self.network.eval()
        
        # 加载后处理模块（如果存在）
        adaptive_enhancement_path = os.path.join(self.model_path, 'adaptive_enhancement.pt')
        if os.path.exists(adaptive_enhancement_path):
            self.adaptive_enhancement.load_state_dict(torch.load(adaptive_enhancement_path))
            self.adaptive_enhancement.eval()

        psnr = PeakSignalNoiseRatio().to(self.device)
        ssim = StructuralSimilarityIndexMeasure().to(self.device)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)
        
        # 创建测试指标字典
        test_metrics = {
            'batch': [],
            'loss': [],
            'psnr': [],
            'ssim': [],
            'lpips': []
        }

        with torch.no_grad():
            test_loss = 0.0
            test_psnr = 0.0
            test_ssim = 0.0
            test_lpips = 0.0
            batch_count = 0
            self.network.eval()
            self.optimizer.zero_grad()
            
            if self.is_dataset_paired:
                for batch_idx, (inputs, targets) in enumerate(tqdm(self.dataloader, desc='Testing...')):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.network(inputs)
                    
                    # 应用自适应后处理
                    if os.path.exists(adaptive_enhancement_path):
                        outputs = self.apply_adaptive_enhancement(inputs, outputs)
                    elif self.apply_post_processing:
                        # 如果没有自适应后处理模块，则使用传统后处理
                        outputs = enhance_contrast(outputs, contrast_factor=1.12)
                        outputs = enhance_color(outputs, saturation_factor=1.35)
                        
                    loss = self.criterion(outputs, targets)
                    current_psnr = psnr(outputs, targets)
                    current_ssim = ssim(outputs, targets)
                    current_lpips = lpips(outputs, targets)
                    
                    test_loss += loss.item()
                    test_psnr += current_psnr
                    test_ssim += current_ssim
                    test_lpips += current_lpips
                    batch_count += 1
                    
                    # 记录每个批次的指标
                    test_metrics['batch'].append(batch_idx)
                    test_metrics['loss'].append(float(loss.item()))
                    test_metrics['psnr'].append(float(current_psnr.item()))
                    test_metrics['ssim'].append(float(current_ssim.item()))
                    test_metrics['lpips'].append(float(current_lpips.item()))
            else:
                for batch_idx, inputs in enumerate(tqdm(self.dataloader, desc='Testing...')):
                    inputs = inputs.to(self.device)
                    outputs = self.network(inputs)
                    
                    # 应用自适应后处理（如果有）
                    if os.path.exists(adaptive_enhancement_path):
                        outputs = self.apply_adaptive_enhancement(inputs, outputs)
                    elif self.apply_post_processing:
                        outputs = enhance_contrast(outputs, contrast_factor=1.12)
                        outputs = enhance_color(outputs, saturation_factor=1.35)
                    
                    batch_count += 1
                    test_metrics['batch'].append(batch_idx)

            if batch_count > 0:
                test_loss = test_loss / batch_count
                test_psnr = test_psnr / batch_count
                test_ssim = test_ssim / batch_count
                test_lpips = test_lpips / batch_count
                
                # 添加平均指标
                test_metrics['average'] = {
                    'loss': float(test_loss),
                    'psnr': float(test_psnr.item()) if hasattr(test_psnr, 'item') else float(test_psnr),
                    'ssim': float(test_ssim.item()) if hasattr(test_ssim, 'item') else float(test_ssim),
                    'lpips': float(test_lpips.item()) if hasattr(test_lpips, 'item') else float(test_lpips)
                }

            # 保存测试指标到JSON文件，使用metrics_path
            model_name = os.path.splitext(self.model_name)[0]
            metrics_filename = f'test_metrics_{model_name}{"_enhanced" if self.apply_post_processing else ""}.json'
            with open(os.path.join(self.metrics_path, metrics_filename), 'w') as f:
                json.dump(test_metrics, f, indent=4)

            if self.is_dataset_paired:
                print(
                    f'Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.4f}, Test SSIM: {test_ssim:.4f}, Test LIPIS: {test_lpips:.4f}')
                print(f'Test metrics saved to {os.path.join(self.metrics_path, metrics_filename)}')

            self.generate_output_images(outputs, self.output_images_path)