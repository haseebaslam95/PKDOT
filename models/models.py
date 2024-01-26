# from video_dataset_mm import  VideoFrameDataset, ImglistToTensor
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # Number of classes



class VIS_MODEL(nn.Module):
    def __init__(self,fold_num):
        super(VIS_MODEL,self).__init__()
        
        visual_model = r3d_18(pretrained=True, progress=True)
        visual_model.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        # visual_model.fc = nn.Linear(512, 256)
        visual_model.fc= None

        visonly_model_path=f'/home/livia/work/Biovid/PartB/biovid_vis_only/model_best_2_class_visonly_fold_{fold_num}.pth'


        # v_m=torch.load('../model_saved_biovid/model_best_2_class_72.pth')
        v_m=torch.load(visonly_model_path)
        del v_m['fc.weight']
        del v_m['fc.bias']

        visual_model.load_state_dict(v_m)
        

        visual_model.fc = nn.Linear(512, 512)
        
        self.out_layer3=nn.Linear(512,num_classes)      

        self.vis_model=visual_model

        visual_model = visual_model.to(device)
        # self.vis_optim = vis_optimizer
        # self.phy_optim = physio_optimizer

        for params in self.vis_model.parameters():
            params.requires_grad = False

        self.vis_model.eval()

    def model_out(self,video_batch):
        output=[]
        # self.vis_model.zero_grad()
        # self.vis_optim.zero_grad()
        # self.phy_optim.zero_grad()
        video_batch = video_batch.to(device)
        vis_feats=self.vis_model(video_batch)

        # output = self.out_layer2(output)
        output = self.out_layer3(vis_feats)
        return vis_feats, output


class VIS_MODEL_LOSO(nn.Module):
    def __init__(self,fold_num):
        super(VIS_MODEL_LOSO,self).__init__()

        fold_num_int = int(fold_num)
        if fold_num_int > 5:
            fold_num = round(fold_num_int%5)
        
        visual_model = r3d_18(pretrained=True, progress=True)
        visual_model.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        # visual_model.fc = nn.Linear(512, 256)
        visual_model.fc= None

        # visonly_model_path=f'/home/livia/work/Biovid/PartB/biovid_vis_only/model_best_2_class_visonly_fold_{fold_num}.pth'


        v_m=torch.load('../model_saved_biovid/model_best_2_class_72.pth')
        # v_m=torch.load(visonly_model_path)
        del v_m['fc.weight']
        del v_m['fc.bias']

        visual_model.load_state_dict(v_m)
        

        visual_model.fc = nn.Linear(512, 512)
        
        self.out_layer3=nn.Linear(512,num_classes)      

        self.vis_model=visual_model

        visual_model = visual_model.to(device)
        # self.vis_optim = vis_optimizer
        # self.phy_optim = physio_optimizer

        # for params in self.vis_model.parameters():
        #     params.requires_grad = False

        # self.vis_model.eval()

    def model_out(self,video_batch):
        output=[]
        # self.vis_model.zero_grad()
        # self.vis_optim.zero_grad()
        # self.phy_optim.zero_grad()
        video_batch = video_batch.to(device)
        vis_feats=self.vis_model(video_batch)

        # output = self.out_layer2(output)
        output = self.out_layer3(vis_feats)
        return vis_feats, output

    



class VIS_PHY_MODEL(nn.Module):
    '''
    CNN-LSTM model.
    
    '''
    def __init__(self):
        super(VIS_PHY_MODEL,self).__init__()
        
        visual_model = r3d_18(pretrained=False, progress=True)
        visual_model.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        # visual_model.fc = nn.Linear(512, 256)
        visual_model.fc= None
        

        v_m=torch.load('model_best_2_class_72.pth')
        del v_m['fc.weight']
        del v_m['fc.bias']

        visual_model.load_state_dict(v_m)
        visual_model.fc = nn.Linear(512, 256)
        visual_model.requires_grad_(False)
        visual_model.eval()

        physio_model=PhysioResNet18(num_classes=2)
        physio_model.load_state_dict(torch.load('model_best_phy_63.pth'))
        physio_model.requires_grad_(False)
        physio_model.eval()
        physio_model.fc = nn.Linear(512, 256)


        # physio_model = models.resnet18(pretrained=True)
        # physio_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # physio_model.fc = nn.Linear(512, 256)



        self.out_layer1=nn.Linear(512,64)
        #add dropout layer after out_layer1

        # self.out_layer2=nn.Linear(256,64)  
        self.out_layer3=nn.Linear(64,num_classes)      
        # vis_optimizer = optim.SGD(visual_model.parameters(), lr=0.0005, momentum=0.9)
        # physio_optimizer = optim.SGD(physio_model.parameters(), lr=0.0005, momentum=0.9)
        


        self.vis_model=visual_model
        self.phy_model=physio_model
        
        physio_model = physio_model.to(device)
        visual_model = visual_model.to(device)
        
        # self.vis_optim = vis_optimizer
        # self.phy_optim = physio_optimizer



    def model_out(self,video_batch,specs_2d):
        output=[]
        # self.vis_model.zero_grad()
        # self.vis_optim.zero_grad()
        # self.phy_optim.zero_grad()
        video_batch = video_batch.to(device)
        specs_2d = specs_2d.to(device= device, dtype=torch.float)

        vis_out=self.vis_model(video_batch)
        phy_out=self.phy_model(specs_2d)
        #contatenate vis_out and phy_out

        combined_out=torch.cat((vis_out,phy_out),1)
        output = self.out_layer1(combined_out)
        # output = self.out_layer2(output)
        output = self.out_layer3(output)
        return output
    
    def model_out_feats(self,video_batch,specs_2d):
        output=[]
        # self.vis_model.zero_grad()
        # self.vis_optim.zero_grad()
        # self.phy_optim.zero_grad()
        video_batch = video_batch.to(device)
        specs_2d = specs_2d.to(device= device, dtype=torch.float)

        vis_out=self.vis_model(video_batch)
        phy_out=self.phy_model(specs_2d)
        #contatenate vis_out and phy_out

        return vis_out,phy_out



class VIS_PHY_MODEL_CAM(nn.Module):
    '''
    CNN-LSTM model.
    
    '''
    def __init__(self):
        super(VIS_PHY_MODEL_CAM,self).__init__()
        
        visual_model = r3d_18(pretrained=False, progress=True)
        visual_model.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        # visual_model.fc = nn.Linear(512, 256)
        visual_model.fc= None
        

        v_m=torch.load('../model_saved_biovid/model_best_2_class_72.pth')
        del v_m['fc.weight']
        del v_m['fc.bias']

        visual_model.load_state_dict(v_m)
        
        visual_model.requires_grad_(False)
        visual_model.eval()
        visual_model.fc = nn.Linear(512, 512)

        physio_model=PhysioResNet18(num_classes=2)
        physio_model.load_state_dict(torch.load('../model_saved_biovid/model_best_phy_63.pth'))
        physio_model.requires_grad_(False)
        physio_model.eval()
        physio_model.fc = nn.Linear(512, 512)



        self.out_layer1=nn.Linear(1024,512)
        #add dropout layer after out_layer1

        # self.out_layer2=nn.Linear(256,64)  
        self.out_layer3=nn.Linear(512,num_classes)      
        # vis_optimizer = optim.SGD(visual_model.parameters(), lr=0.0005, momentum=0.9)
        # physio_optimizer = optim.SGD(physio_model.parameters(), lr=0.0005, momentum=0.9)
        


        self.vis_model=visual_model
        self.phy_model=physio_model
        
        physio_model = physio_model.to(device)
        visual_model = visual_model.to(device)
        
        # self.vis_optim = vis_optimizer
        # self.phy_optim = physio_optimizer



    def model_out(self,video_batch,specs_2d):
        output=[]
        # self.vis_model.zero_grad()
        # self.vis_optim.zero_grad()
        # self.phy_optim.zero_grad()
        video_batch = video_batch.to(device)
        specs_2d = specs_2d.to(device= device, dtype=torch.float)

        vis_out=self.vis_model(video_batch)
        phy_out=self.phy_model(specs_2d)
        #contatenate vis_out and phy_out

        combined_out=torch.cat((vis_out,phy_out),1)
        output = self.out_layer1(combined_out)
        # output = self.out_layer2(output)
        output = self.out_layer3(output)
        return output
    
    def model_out_feats(self,video_batch,specs_2d):
        output=[]
        # self.vis_model.zero_grad()
        # self.vis_optim.zero_grad()
        # self.phy_optim.zero_grad()
        video_batch = video_batch.to(device)
        specs_2d = specs_2d.to(device= device, dtype=torch.float)

        vis_out=self.vis_model(video_batch)
        phy_out=self.phy_model(specs_2d)
        #contatenate vis_out and phy_out

        return vis_out,phy_out




    
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.V = nn.Linear(input_dim, input_dim, bias=False)
        self.tanh = nn.Tanh()
        self.fc=nn.Linear(input_dim,2)

    def forward(self, x):
        q = self.W(x)
        attn_weights = torch.softmax(self.V(self.tanh(q)), dim=1)
        attended_x = attn_weights * x
        outs=self.fc(attended_x)
        return outs
    


class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)
        
        # Apply feed forward network
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x
    
class MultimodalTransformer(nn.Module):
    def __init__(self, visual_dim, physiological_dim, num_heads, hidden_dim, num_layers, num_classes):
        super(MultimodalTransformer, self).__init__()
        self.visual_encoder = TransformerEncoderBlock(visual_dim, num_heads, hidden_dim, num_layers)
        self.physiological_encoder = TransformerEncoderBlock(physiological_dim, num_heads, hidden_dim, num_layers)
        self.cross_attention_v = nn.MultiheadAttention(visual_dim, num_heads)
        self.cross_attention_p = nn.MultiheadAttention(physiological_dim, num_heads)
        self.gated_attention = nn.Linear(visual_dim + physiological_dim, 1)
        self.fc = nn.Linear(visual_dim , num_classes)
    
    def forward(self, visual_features, physiological_features):
        visual_encoded = self.visual_encoder(visual_features)
        physiological_encoded = self.physiological_encoder(physiological_features)

        cross_attention_output_v, _ = self.cross_attention_v(physiological_encoded.permute(1, 0, 2), visual_encoded.permute(1, 0, 2), visual_encoded.permute(1, 0, 2))
        
        cross_attention_output_v = cross_attention_output_v.permute(1, 0, 2)
        
        cross_attention_output_p, _ = self.cross_attention_p(visual_encoded.permute(1, 0, 2), physiological_encoded.permute(1, 0, 2), physiological_encoded.permute(1, 0, 2))
        cross_attention_output_p = cross_attention_output_p.permute(1, 0, 2)

        #concatenate
        # cross_attention_output=torch.cat((cross_attention_output_v,cross_attention_output_p),dim=2)
        # cross_attention_output = cross_attention_output_v * cross_attention_output_p
        gating_input = torch.cat((cross_attention_output_v, cross_attention_output_p), dim=2)
        gating_coefficients = torch.sigmoid(self.gated_attention(gating_input))
        combined_attention = (gating_coefficients * cross_attention_output_v) + ((1 - gating_coefficients) * cross_attention_output_p)
        combined_attention = combined_attention.squeeze(1)


        output = self.fc(combined_attention)  # Use only the final timestep for classification
        return output,combined_attention
    

# class MultimodalTransformer(nn.Module):
#     def __init__(self, visual_dim, physiological_dim, num_heads, hidden_dim, num_layers, num_classes):
#         super(MultimodalTransformer, self).__init__()
#         self.visual_encoder = TransformerEncoderBlock(visual_dim, num_heads, hidden_dim, num_layers)
#         self.physiological_encoder = TransformerEncoderBlock(physiological_dim, num_heads, hidden_dim, num_layers)
#         self.visual_attention = nn.MultiheadAttention(visual_dim, num_heads)
#         self.physiological_attention = nn.MultiheadAttention(physiological_dim, num_heads)
#         self.fc = nn.Linear(2*visual_dim, num_classes)
    
#     def forward(self, visual_features, physiological_features):
#         #encoders
#         visual_encoded = self.visual_encoder(visual_features)
#         physiological_encoded = self.physiological_encoder(physiological_features)
        
#         #cross attentions
#         physiological_attention_output, _ = self.physiological_attention(visual_encoded.permute(1, 0, 2), physiological_encoded.permute(1, 0, 2), physiological_encoded.permute(1, 0, 2))   
#         visual_attention_output = self.visual_attention(physiological_encoded.permute(1, 0, 2), visual_encoded.permute(1, 0, 2), visual_encoded.permute(1, 0, 2))
        
#         #permuting and concatenating
#         visual_attention_output = visual_attention_output.permute(1, 0, 2)
#         physiological_attention_output = physiological_attention_output.permute(1, 0, 2)
#         combined_attention_output = torch.cat((visual_attention_output, physiological_attention_output), dim=2)
        
#         output = self.fc(combined_attention_output[:, -1, :])  # Use only the final timestep for classification
#         return output