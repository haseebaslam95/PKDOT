from dataset.video_dataset_mm import  VideoFrameDataset, ImglistToTensor
from comet_ml import Experiment
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
from biovid_physio_classification import PhysioResNet18

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from torchvision.models.video import r3d_18
from torchvision import models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

"""
Training settings

"""

def validate_vis_only_pkd(vis_model, val_dataloader, criterion, device):
    # Validation phase
    vis_model.eval() 
    val_correct = 0
    val_total = 0
    val_vis_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)

            _, val_vis_outputs = vis_model.model_out(val_inputs)

            val_vis_loss += criterion(val_vis_outputs, val_labels).item()

            val_outputs = val_vis_outputs

            _,val_predicted = torch.max(val_outputs.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_vis_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss

def validate_vis_only(vis_model, val_dataloader, criterion, device):
    # Validation phase
    vis_model.eval() 
    val_correct = 0
    val_total = 0
    val_vis_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)

            val_vis_outputs = vis_model(val_inputs)

            val_vis_loss += criterion(val_vis_outputs, val_labels).item()

            val_outputs = val_vis_outputs

            _,val_predicted = torch.max(val_outputs.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_vis_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss

def validate(vis_phy_mod, val_dataloader, criterion, device):
    # Validation phase
    vis_phy_mod.vis_model.eval() 
    vis_phy_mod.phy_model.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)

            val_out = vis_phy_mod.model_out(val_inputs,spec_2d)


            # val_physio_loss = criterion(val_out, val_labels)
            val_t_loss += criterion(val_out, val_labels).item()


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss


def validate_cam(vis_phy_mod,cam, val_dataloader, criterion, device):
    # Validation phase
    vis_phy_mod.vis_model.eval() 
    vis_phy_mod.phy_model.eval()
    cam.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)
            
            
            visfeats, phyfeats = vis_phy_mod.model_out_feats(val_inputs,spec_2d)
            visual_feats=visfeats.unsqueeze(1)
            physio_feats=phyfeats.unsqueeze(1)
            physiovisual_outs = cam(visual_feats, physio_feats)
            val_out=physiovisual_outs.squeeze(1)


            # val_physio_loss = criterion(val_out, val_labels)
            val_t_loss += criterion(val_out, val_labels).item()


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss


def validate_attn(vis_phy_mod,attention_m, val_dataloader, criterion, device):
    # Validation phase
    vis_phy_mod.vis_model.eval() 
    vis_phy_mod.phy_model.eval()
    attention_m.eval()
    # classifier_m.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)

            vis_feats, phy_feats = vis_phy_mod.model_out_feats(val_inputs,spec_2d)
            concatenated_features = torch.cat((vis_feats, phy_feats), dim=1)
            val_out = attention_m(concatenated_features)
            # val_out = classifier_m(attended_features)
            # val_physio_loss = criterion(val_out, val_labels)
            val_t_loss += criterion(val_out, val_labels).item()


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss


def validate_mmtransformer(vis_phy_mod,mm_transformer, val_dataloader, criterion, device):
    # Validation phase
    vis_phy_mod.vis_model.eval() 
    vis_phy_mod.phy_model.eval()
    mm_transformer.eval()
    # classifier_m.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            
            # spec_2d = spec_2d.to(device= device, dtype=torch.float)
            imputed_spec_2d = torch.zeros_like(spec_2d).to(device= device, dtype=torch.float)

            vis_feats, phy_feats = vis_phy_mod.model_out_feats(val_inputs,imputed_spec_2d)
            vis_feats = vis_feats.unsqueeze(1)
            phy_feats = phy_feats.unsqueeze(1)

            imputed_phy_feats = torch.zeros_like(phy_feats).to(device= device, dtype=torch.float)
            val_out,_ = mm_transformer(vis_feats, imputed_phy_feats)
            # val_out = classifier_m(attended_features)
            # val_physio_loss = criterion(val_out, val_labels)
            
            
            val_t_loss += criterion(val_out, val_labels).item()  #normal


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss


def validate_mmtransformer_dmwl(vis_phy_mod,mm_transformer, val_dataloader, criterion, device):
    # Validation phase
    vis_phy_mod.vis_model.eval() 
    vis_phy_mod.phy_model.eval()
    mm_transformer.eval()
    # classifier_m.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            
            # spec_2d = spec_2d.to(device= device, dtype=torch.float)
            imputed_spec_2d = torch.zeros_like(spec_2d).to(device= device, dtype=torch.float)

            vis_feats, phy_feats = vis_phy_mod.model_out_feats(val_inputs,imputed_spec_2d)
            vis_feats = vis_feats.unsqueeze(1)
            phy_feats = phy_feats.unsqueeze(1)

            imputed_phy_feats = torch.zeros_like(phy_feats).to(device= device, dtype=torch.float)
            val_out,_ = mm_transformer(vis_feats, imputed_phy_feats)
            # val_out = classifier_m(attended_features)
            # val_physio_loss = criterion(val_out, val_labels)
            
            
            val_t_loss += criterion(val_out, val_labels).item()  #normal


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss


def validate_mmtransformer_dmwl_wtn(visual_model,mm_transformer,transform_net,val_dataloader, criterion, device):
    # Validation phase
    visual_model.vis_model.eval() 
    mm_transformer.eval()
    # classifier_m.eval()
    transform_net.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            
            # spec_2d = spec_2d.to(device= device, dtype=torch.float)
            # imputed_spec_2d = torch.zeros_like(spec_2d).to(device= device, dtype=torch.float)


            vis_feats, _ = visual_model.model_out(val_inputs)
            # vis_feats, _ = visual_model.model_out_feats(val_inputs,spec_2d)

            recon_phy_feats = transform_net(vis_feats.detach())
            vis_feats = vis_feats.unsqueeze(1)
            recon_phy_feats = recon_phy_feats.unsqueeze(1)



            # imputed_phy_feats = torch.zeros_like(phy_feats).to(device= device, dtype=torch.float)
            val_out,_ = mm_transformer(vis_feats, recon_phy_feats)
            # val_out = classifier_m(attended_features)
            # val_physio_loss = criterion(val_out, val_labels)
            
            
            val_t_loss += criterion(val_out, val_labels).item()  #normal


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    # print(f'Validation accuracy: {val_accuracy}%')
    # print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss


def validate_mmtransformer_dmwl_kf(visual_model,mm_transformer,transform_net,val_dataloader, criterion, device):
    # Validation phase
    visual_model.vis_model.eval() 
    mm_transformer.eval()
    # classifier_m.eval()
    transform_net.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            
            # spec_2d = spec_2d.to(device= device, dtype=torch.float)
            # imputed_spec_2d = torch.zeros_like(spec_2d).to(device= device, dtype=torch.float)


            vis_feats, outs = visual_model.model_out(val_inputs)
            # vis_feats, _ = visual_model.model_out_feats(val_inputs,spec_2d)

            recon_phy_feats = transform_net(vis_feats.detach())
            vis_feats = vis_feats.unsqueeze(1)
            recon_phy_feats = recon_phy_feats.unsqueeze(1)



            # imputed_phy_feats = torch.zeros_like(phy_feats).to(device= device, dtype=torch.float)
            # val_out,_ = mm_transformer(vis_feats, recon_phy_feats)
            # val_out = classifier_m(attended_features)
            # val_physio_loss = criterion(val_out, val_labels)
            val_out = outs  #normal
            
            val_t_loss += criterion(val_out, val_labels).item()  #normal


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    # print(f'Validation accuracy: {val_accuracy}%')
    # print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss


def validate_mmtransformer_dmwl_loso(visual_model,mm_transformer,transform_net,val_dataloader, criterion, device):
    # Validation phase
    visual_model.vis_model.eval() 
    mm_transformer.eval()
    # classifier_m.eval()
    transform_net.eval()
    val_correct = 0
    val_total = 0
    val_t_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            
            # spec_2d = spec_2d.to(device= device, dtype=torch.float)
            # imputed_spec_2d = torch.zeros_like(spec_2d).to(device= device, dtype=torch.float)


            vis_feats, outs = visual_model.model_out(val_inputs)
            # vis_feats, _ = visual_model.model_out_feats(val_inputs,spec_2d)

            recon_phy_feats = transform_net(vis_feats.detach())
            vis_feats = vis_feats.unsqueeze(1)
            recon_phy_feats = recon_phy_feats.unsqueeze(1)



            # imputed_phy_feats = torch.zeros_like(phy_feats).to(device= device, dtype=torch.float)
            # val_out,_ = mm_transformer(vis_feats, recon_phy_feats)
            # val_out = classifier_m(attended_features)
            # val_physio_loss = criterion(val_out, val_labels)
            val_out = outs  #normal
            
            val_t_loss += criterion(val_out, val_labels).item()  #normal


            _, val_predicted = torch.max(val_out.data, 1)
            
            val_total += len(val_labels)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_t_loss)/2) / len(val_dataloader)
    # print(f'Validation accuracy: {val_accuracy}%')
    # print(f'Validation loss: {avg_val_loss}')
    return val_accuracy, avg_val_loss