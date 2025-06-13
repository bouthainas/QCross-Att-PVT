import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from timm.models import create_model
from pytorchcv.model_provider import get_model
import torchmetrics.functional as metrics
import numpy as np
import random
import math


#Weighted Loss
class WeightedL1Loss(nn.Module):
    def __init__(self, score_levels, total_samples, label_counts):
        """
        Args:
            score_levels (int): Number of discrete score levels, e.g., 17 for RALO, 101 for CT.
            total_samples (int): Total number of training samples.
            label_counts (dict): Dictionary mapping each label value to its count in the dataset.
        """
        super(WeightedL1Loss, self).__init__()
        self.score_levels = score_levels
        self.total_samples = total_samples

        # Compute class weights: w_l = N / (c_l * k)
        self.weights = {}
        for label, count in label_counts.items():
            if count == 0:
                self.weights[label] = 0.0  # Avoid division by zero
            else:
                self.weights[label] = total_samples / (count * score_levels)

    def forward(self, preds, targets):
        """
        Args:
            preds: Tensor of shape (B,) — model predictions.
            targets: Tensor of shape (B,) — ground truth labels.
        Returns:
            Weighted L1 loss scalar.
        """
        weights = torch.tensor(
            [self.weights[float(t.item())] for t in targets],
            device=preds.device,
            dtype=preds.dtype
        )
        loss = weights * torch.abs(preds - targets)
        return loss.mean()

# Convolution
class ConvTransform(nn.Module):
    def __init__(self):
        super(ConvTransform, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


# Attention Block
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


   

# Main Model
class Model(pl.LightningModule):
    def __init__(self, backbone='pvt_v2_b0', label_list=None, trainingsize, dataset='RALO'):
        super().__init__()
        self.save_hyperparameters()
	
	if dataset == 'RALO':
            self.score_levels = 17  # 0–8 by 0.5
            total_samples = len(label_list)
            label_counts = Counter(label_list)
        elif dataset_name.lower() == 'percovid':
            self.score_levels = 101  # 0–100
	    total_samples = len(label_list)
            label_counts = Counter(label_list)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Define loss
        self.loss_func = WeightedL1Loss(score_levels=self.score_levels,
                                        total_samples=self.total_samples,
                                        label_counts=self.label_counts)
        self.lr = 1e-5
        self.lr_patience = 5
        self.lr_min = 1e-7

        self.encoder1 = timm.create_model('pvt_v2_b0', pretrained=True, num_classes=1)
        self.encoder1.head = nn.Identity()
        self.encoder2 = timm.create_model('pvt_v2_b0', pretrained=True, num_classes=1)
        self.encoder2.head = nn.Identity()
        self.encoder3 = timm.create_model('pvt_v2_b0', pretrained=True, num_classes=1)
        self.encoder3.head = nn.Identity()
        self.encoder4 = timm.create_model('pvt_v2_b0', pretrained=True, num_classes=1)
        self.encoder4.head = nn.Identity()

        self.aggregator = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=1)
        self.aggregator.norm = nn.Identity()
        self.aggregator.pre_legits = nn.Identity()
        self.aggregator.head = nn.Sequential(nn.Linear(192, 128), nn.Linear(128, 1))

        self.att1 = Attention_block(F_g=768, F_l=256, F_int=128)
        self.att2 = Attention_block(F_g=768, F_l=256, F_int=128)
        self.att3 = Attention_block(F_g=768, F_l=256, F_int=128)
        self.att4 = Attention_block(F_g=768, F_l=256, F_int=128)

        self.conv = ConvTransform()

        self.tr_loss, self.tr_mae, self.vl_loss, self.vl_mae, self.ts_loss, self.ts_mae = [], [], [], [], [], []

    def split(self, x):
        B, C, H, W = x.shape
        return x[:, :, :H//2, :W//2], x[:, :, :H//2, W//2:], x[:, :, H//2:, :W//2], x[:, :, H//2:, W//2:]

    def rand_bbox(self, size, lam):
    	W = size[2]
    	H = size[3]
    	cut_rat = np.sqrt(1. - lam)
    	cut_w = int(W * cut_rat)
    	cut_h = int(H * cut_rat)

    	cx = np.random.randint(W)
    	cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
    	bby1 = np.clip(cy - cut_h // 2, 0, H)
    	bbx2 = np.clip(cx + cut_w // 2, 0, W)
    	bby2 = np.clip(cy + cut_h // 2, 0, H)

    	return bbx1, bby1, bbx2, bby2

     def cutmix_data(self, images, labels, alpha=1.0):
        assert images.size(0) > 1, "CutMix requires a batch of at least 2 images"

        # Sample lambda from beta distribution
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(images.size(0)).to(images.device)

        y1 = labels
        y2 = labels[rand_index]

        bbx1, bbx2, bby1, bby2 = self.rand_bbox(images.size(), lam)
        images_mixed = images.clone()
        images_mixed[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda based on the actual area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))

        # Return the mixed image, original labels, and the box coordinates
        return images_mixed, y1, y2, lam, (bbx1, bbx2, bby1, bby2)

    def transmix_label(self, image, label, ratio=0.5):
        # Hook into the final attention block
        self.model.model.blocks[-1].attn.forward = self.my_forward_wrapper(self.model.model.blocks[-1].attn)

        # Extract the attention maps
        attn = self.model.model.blocks[-1].attn.attn_map.mean(dim=1).squeeze(0).detach()  # (1, h*w)
        cls_attn = self.model.model.blocks[-1].attn.cls_attn_map.mean(dim=1).detach()     # (1, h*w)

        input_shape = image.shape

        # Get mixed image and labels from CutMix
        x, y1, y2, lam0, box = self.cutmix_data(image, label)  # <- label is needed to generate y1, y2

        # Create binary mask from CutMix box
        mask = torch.zeros((input_shape[2], input_shape[3])).cuda()
        mask[box[0]:box[1], box[2]:box[3]] = 1

        # Resize mask to match attention map size
        mask = nn.Upsample(size=int(math.sqrt(attn.shape[1])), mode='nearest')(mask.unsqueeze(0).unsqueeze(0)).int()
        mask = mask.view(1, -1).repeat(attn.shape[0], 1)  # (B, h*w)

        # Compute attention-weighted mix ratio
        w1 = torch.sum((1 - mask) * attn, dim=1)
        w2 = torch.sum(mask * attn, dim=1)
        lam1 = w1 / (w1 + w2 + 1e-6)
        lam2 = w2 / (w1 + w2 + 1e-6)

        # Weighted label (single value)
        target = y1 * lam1 + y2 * lam2

        # Apply threshold-based filtering (replace < and > with your preferred values)
        valid_mask = (target > self.lower_thresh) & (target < self.upper_thresh)

        if valid_mask.sum() == 0:
            # Avoid empty batch after filtering
            return x, target

        return x[valid_mask], target[valid_mask]



    def forward(self, images):
        batch_size = images.size(0)
        i1, i2, i3, i4 = self.split(images)

        s1 = self.encoder1.forward_features(i1).view(batch_size, 256, 7, 7)
        s2 = self.encoder2.forward_features(i2).view(batch_size, 256, 7, 7)
        s3 = self.encoder3.forward_features(i3).view(batch_size, 256, 7, 7)
        s4 = self.encoder4.forward_features(i4).view(batch_size, 256, 7, 7)

        g1 = torch.cat((s2, s3, s4), dim=1)
        g2 = torch.cat((s1, s3, s4), dim=1)
        g3 = torch.cat((s1, s2, s4), dim=1)
        g4 = torch.cat((s1, s2, s3), dim=1)

        Ax1 = self.att1(g1, s1)
        Ax2 = self.att2(g2, s2)
        Ax3 = self.att3(g3, s3)
        Ax4 = self.att4(g4, s4)

        top = torch.cat((Ax1, Ax2), dim=2)
        bottom = torch.cat((Ax3, Ax4), dim=2)
        combined = torch.cat((top, bottom), dim=3)

        out = self.conv(combined)
        y = self.aggregator(out)
        return y

    def training_step(self, batch, batch_idx):
        images, labels = batch
        c_images, c_labels = self.transmix_data(images, labels, lower,  upper)
        combined_images = torch.cat([images, c_images], dim=0)
        combined_labels = torch.cat([labels, c_labels], dim=0)
        output = self.forward(combined_images)
        loss = self.loss_func(output, combined_labels)
        mae = metrics.mean_absolute_error(output, combined_labels)
        pc = metrics.pearson_corrcoef(output, combined_labels)
        mae_sdv = metrics.mean_absolute_error(output, combined_labels)
        return {'loss': loss, 'mae': mae, "pc": pc}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self.forward(images)
        loss = self.loss_func(output, labels)
        mae = metrics.mean_absolute_error(output, labels)
        pc = metrics.pearson_corrcoef(output, labels)
        return {'loss': loss, 'mae': mae, "pc": pc}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        output = self.forward(images)
        loss = self.loss_func(output, labels)
        mae = metrics.mean_absolute_error(output, labels)
        pc = metrics.pearson_corrcoef(output, labels)
        return {"loss": loss, "mae": mae, "pc": pc}

    def training_epoch_end(self, outs):
        loss = torch.stack([x['loss'] for x in outs]).mean()
        mae = torch.stack([x['mae'] for x in outs]).mean()
        pc = torch.stack([x['pc'] for x in outs]).mean()
        self.tr_loss.append(loss)
        self.tr_mae.append(mae)
        self.log('Loss/Train', loss, prog_bar=True, on_epoch=True)
        self.log('MAE/Train', mae, prog_bar=True, on_epoch=True)
        self.log('PC/Train', pc, prog_bar=True, on_epoch=True)

    def validation_epoch_end(self, outs):
        loss = torch.stack([x['loss'] for x in outs]).mean()
        mae = torch.stack([x['mae'] for x in outs]).mean()
        pc = torch.stack([x['pc'] for x in outs]).mean()
        self.vl_loss.append(loss)
        self.vl_mae.append(mae)
        self.log('Loss/Val', loss, prog_bar=True, on_epoch=True)
        self.log('MAE/Val', mae, prog_bar=True, on_epoch=True)
        self.log('PC/Val', pc, prog_bar=True, on_epoch=True)

    def test_epoch_end(self, outs):
        loss = torch.stack([x['loss'] for x in outs]).mean()
        mae = torch.stack([x['mae'] for x in outs]).mean()
        pc = torch.stack([x['pc'] for x in outs]).mean()
        mae_sdv = torch.stack([x['mae'] for x in outs]).std()
        self.ts_loss.append(loss)
        self.ts_mae.append(mae)
        self.log('Loss/Test', loss, prog_bar=True, on_epoch=True)
        self.log('PC/Test', pc, prog_bar=True, on_epoch=True)
        self.log('MAE/Test', mae, prog_bar=True, on_epoch=True)
        self.log('MAE_SDV/Test', mae_sdv, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.5, 0.99))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=trainingsize, T_mult=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'Loss/Val', "interval": 'epoch'}
