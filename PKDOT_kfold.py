from dataset.video_dataset_mm import  VideoFrameDataset, ImglistToTensor
from comet_ml import Experiment
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from biovid_physio_classification import PhysioResNet18
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from torchvision import models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from validate import validate_mmtransformer_dmwl_wtn, validate_mmtransformer_dmwl_kf
from models.models import  VIS_PHY_MODEL_CAM, MultimodalTransformer, VIS_MODEL
from models.transformation_network import TransformNet 
import ot
from geomloss import SamplesLoss
from pkdot_utils import *




"""
Training settings

"""
num_epochs = 20
best_epoch = 0
check_every = 1
b_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_val_acc=0

lr_vis_phy = 0.0001  
lr_mmtransformer = 0.0001 



# comet_ml settings
experiment = Experiment(
  api_key="",
  project_name="",
  workspace="",
  disabled=True
)

parameters = {'batch_size': b_size,
              'learning_rate bb': lr_vis_phy,
              'learning_rate mmtransformer': lr_mmtransformer,
              'epochs':num_epochs            
              }
experiment.log_parameters(parameters)




num_frames = 5  # Number of frames in each video clip
num_channels = 3  # Number of channels (e.g., RGB)
video_length = 112  # Length of the video in each dimension
num_classes = 2  # Number of classes



videos_root = '/home/livia/work/Biovid/PartB/biovid_classes'


preprocess_train = transforms.Compose([
    ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    transforms.Resize(112),  # image batch, resize smaller edge to 299
    # transforms.CenterCrop(112),  # image batch, center crop to square 299x299
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(p=0.2),
    # transforms.RandomRotation(degrees=10),
    # transforms.RandomCrop(112),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    # transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
    # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10, resample=False, fillcolor=0),
])


preprocess_test = transforms.Compose([
    ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    transforms.Resize(112),  # image batch, resize smaller edge to 299
    # transforms.CenterCrop(112),  # image batch, center crop to square 299x299
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])





def train_student(train_file_path, val_file_path,fold_num):
    train_annotation_file = os.path.join(videos_root, train_file_path)
    val_annotation_file = os.path.join(videos_root, val_file_path)
    model_save_path_bb = os.path.join(os.getcwd(), 'model_best_dmwl_bb_ot_visonly_'+str(fold_num)+'.pth')
    model_save_path_mmtransformer = os.path.join(os.getcwd(), 'model_best_dmwl_mmtrans_ot_visonly_'+str(fold_num)+'.pth')

    best_val_acc = 0
    best_epoch = 0

    sinkhorn_loss_func = SamplesLoss("sinkhorn", p=2, blur=0.1)


    criterion = nn.CrossEntropyLoss()


    vis_model_student=VIS_MODEL(fold_num).to(device=device)
    vis_model_student.eval()
    mm_transformer_student = MultimodalTransformer(visual_dim=512, physiological_dim=512, num_heads=2, hidden_dim=512, num_layers=2, num_classes=2)

    mm_transformer_student = mm_transformer_student.to(device=device)

    transform_net = TransformNet().to(device=device)

    #freeze the weights of the transformation network
    for param in transform_net.parameters():
        param.requires_grad = False

    vis_optimizer = optim.Adam(vis_model_student.parameters(), lr=lr_vis_phy)
    mmtransformer_optimizer = optim.Adam(mm_transformer_student.parameters(), lr=lr_mmtransformer)

    mmtransformer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(mmtransformer_optimizer, mode='max', factor=0.01, patience=5,verbose=True)


    vis_phy_model_teacher = VIS_PHY_MODEL_CAM().to(device=device)
    mm_transformer_teacher = MultimodalTransformer(visual_dim=512, physiological_dim=512, num_heads=2, hidden_dim=512, num_layers=2, num_classes=2).to(device=device)

    vis_phy_model_teacher.load_state_dict(torch.load(''))  #path to pretrained teacher model
    mm_transformer_teacher.load_state_dict(torch.load('')) #path to pretrained teacher model
    


    #freeze the weights of the teacher
    for param in vis_phy_model_teacher.parameters():
        param.requires_grad = False
    
    for param in mm_transformer_teacher.parameters():
        param.requires_grad = False

    
    vis_phy_model_teacher.eval()
    mm_transformer_teacher.eval()
    transform_net.eval()

    train_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=train_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess_train,
        test_mode=False)

    val_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=val_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess_test,
        test_mode=True)


    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=b_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)


    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)


    with experiment.train():
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            
            vis_model_student.vis_model.eval()
            mm_transformer_student.train()
            
            running_loss = 0.0
            correct = 0
            total = 0
            for i,(spec_2d,video_batch, labels) in enumerate(train_dataloader,0):
    
                mmtransformer_optimizer.zero_grad()
                vis_optimizer.zero_grad()

                #prepare data
                video_batch=video_batch.permute(0, 2, 1, 3, 4)
                video_batch = video_batch.to(device)
                labels = labels.to(device)

                
                #get teacher embeddings
                with torch.no_grad():
                    vis_feats_teacher, phy_feats_teacher = vis_phy_model_teacher.model_out_feats(video_batch,spec_2d)
                    vis_feats_teacher = vis_feats_teacher.unsqueeze(1)
                    phy_feats_teacher = phy_feats_teacher.unsqueeze(1)
                    _, intermediate_outs_teacher = mm_transformer_teacher(vis_feats_teacher, phy_feats_teacher)

                    #sort intermediate_outs_teacher based on class labels where all the samples of the same class are together
                    # intermediate_outs_teacher = intermediate_outs_teacher[np.argsort(labels.detach().cpu().numpy())]


                # get student embeddings
                vis_feats, _ = vis_model_student.model_out(video_batch)

                #get the hallucinated features from T-Net
                recon_phy_feats = transform_net(vis_feats.detach())
                vis_feats = vis_feats.unsqueeze(1)
                recon_phy_feats = recon_phy_feats.unsqueeze(1)

                outs, intermediate_outs_student = mm_transformer_student(vis_feats, recon_phy_feats)



                cosine_similarity_matrix_teacher = cosine_similarity_matrix_transpose_torch(intermediate_outs_teacher)


 
                cosine_similarity_matrix_student = cosine_similarity_matrix_transpose_torch(intermediate_outs_student)

                # ensure that the similarity matrices are non negative
                cosine_similarity_matrix_teacher_nonneg = cosine_similarity_matrix_teacher - cosine_similarity_matrix_teacher.min()
                cosine_similarity_matrix_student_nonneg = cosine_similarity_matrix_student - cosine_similarity_matrix_student.min()

            
                
        



                #select topk values from the similiarity matrix where the similarity is maximum from the teacher
                # topk = 10
                # topk_indices = np.argpartition(cosine_similarity_matrix_teacher.detach().cpu().nump(), -topk, axis=1)[:, -topk:]
                # cosine_similarity_matrix_teacher = cosine_similarity_matrix_teacher[np.arange(cosine_similarity_matrix_teacher.shape[0])[:, None], topk_indices]
                # cosine_similarity_matrix_student = cosine_similarity_matrix_student[np.arange(cosine_similarity_matrix_student.shape[0])[:, None], topk_indices]


                #select topk values from the similiarity matrix where the similarity is minimum from the teacher
                topk = 30
                topk_indices = np.argpartition(cosine_similarity_matrix_teacher.detach().cpu().numpy(), topk, axis=1)[:, :topk]
                cosine_similarity_matrix_teacher_tk = cosine_similarity_matrix_teacher_nonneg[np.arange(cosine_similarity_matrix_teacher_nonneg.shape[0])[:, None], topk_indices]
                cosine_similarity_matrix_student_tk = cosine_similarity_matrix_student_nonneg[np.arange(cosine_similarity_matrix_student_nonneg.shape[0])[:, None], topk_indices]
                
                
                '''
                OT between two feature vectors
                '''    
                # intermediate_outs_teacher_flat = intermediate_outs_teacher.view(intermediate_outs_teacher.shape[0], -1)
                # i ntermediate_outs_student_flat = intermediate_outs_student.view(intermediate_outs_student.shape[0], -1)

                # a = torch.softmax(intermediate_outs_teacher_flat, dim=1)
                # b = torch.softmax(intermediate_outs_student_flat, dim=1)
                # C = torch.cdist(a, b)
                
                # sinkhorn_loss = sinkhorn_loss_func(intermediate_outs_teacher,intermediate_outs_student )

                #normalize this sinkhorn loss so that it is between 0 and 1
                # sinkhorn_loss = sinkhorn_loss/sinkhorn_loss.max()

                
                
                '''
                OT between two simalrity matrices (euclidean distance matrices)
                '''  
                
                # calculate distance between the euclidean distance matrices 
            
                # sinkhorn_loss = sinkhorn_loss_func(euclidean_distance_matrix_teacher_n, euclidean_distance_matrix_student_n)
                

                '''
                OT between two simalrity matrices (cosine sim matrices)
                ''' 
   
                sinkhorn_loss = sinkhorn_loss_func(cosine_similarity_matrix_teacher_tk, cosine_similarity_matrix_student_tk)

                
                #task loss
                gt_loss = criterion(outs, labels)



                lambdaa = 0.6 # lambdaa is the weight for the sinkhorn loss
                total_loss= gt_loss + lambdaa * sinkhorn_loss

                
                total_loss.backward()

                mmtransformer_optimizer.step()
                vis_optimizer.step()

                running_loss += total_loss.item()
    
                _, predicted = torch.max(outs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0
            


                # plot similarity matrices 
                # if epoch == 0:
                #     if i == tmp_i:
                #         print("saved at batch: ", tmp_i)
                #         plot_simmat(cosine_similarity_matrix_teacher[original_order,:].detach().to('cpu').numpy(), cosine_similarity_matrix_student[original_order,:].detach().to('cpu').numpy(),epoch,fold_num) 
                
                # if epoch == 1 or epoch == 5 or epoch==10 or epoch==15 or epoch == num_epochs-1:
                #     if i == tmp_i:
                #         print("second saved at batch: ", i)
                #         plot_simmat(cosine_similarity_matrix_teacher.detach().to('cpu').numpy(), cosine_similarity_matrix_student.detach().to('cpu').numpy(),epoch,fold_num) 

                
                
            
            
            train_accuracy= 100 * correct / total
            print("*********************************************\n")
            print(f"Accuracy after epoch {epoch + 1}: {train_accuracy}%")
            train_loss= running_loss / 100
            experiment.log_metric('Loss', train_loss,epoch= epoch)
            experiment.log_metric('Accuracy', train_accuracy ,epoch= epoch)
            print("Sinkhorn loss: ", sinkhorn_loss.item())
            print("total loss: ", total_loss.item())
            print("Best validation accuracy: ", best_val_acc, "at epoch: ", best_epoch)






            if epoch % check_every == 0:
                val_acc, val_loss = validate_mmtransformer_dmwl_wtn(vis_model_student,mm_transformer_student,transform_net, val_dataloader, criterion, device)
                print( "Validation accuracy: ", val_acc)
                # experiment.log_metric('Val Accuracy', val_acc,epoch= epoch)
                # experiment.log_metric('Val Loss', val_loss,epoch= epoch)
                mmtransformer_scheduler.step(val_acc)   
                current_lr = mmtransformer_optimizer.param_groups[0]['lr']
                experiment.log_metric('Learning Rate', current_lr,epoch= epoch)
                # print('Current learning rate: ', current_lr)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                    torch.save(vis_model_student.state_dict(), model_save_path_bb)
                    torch.save(mm_transformer_student.state_dict(), model_save_path_mmtransformer)  
                    print('Validation_accuracy: ', val_acc)
                    print('Best model saved at epoch: ', epoch+1)
                    best_epoch = epoch+1


    print("Finished Training")

    train_accuracy = 100 * correct / total
    avg_train_loss = running_loss / len(train_dataloader)
    print(f'Training accuracy: {train_accuracy}%')
    print(f'Training loss: {avg_train_loss}')

    print("Best model saved at epoch: ", best_epoch)
    print("Best validation accuracy: ", best_val_acc)
    return best_val_acc




def train_k_fold():
    k=5
    #create list of size 
    best_acc_list = []


    for i in range(k):
        print("Fold: ", i+1)
        train_file_path = 'fold_'+ str(i+1) +'_train.txt'
        val_file_path = 'fold_'+ str(i+1) +'_test.txt'
        fold_num = str(i+1)
        #call only for fold 3 and 4
        best_fold_acc=train_student(train_file_path, val_file_path,fold_num)
        best_acc_list.append(best_fold_acc)

    print("Best validation accuracy for each fold: ", best_acc_list)
    print("Average best validation accuracy: ", sum(best_acc_list)/k)



def test_k_fold():
    saved_path_root= '' #root path to saved models
    k=5
    #create list of size 
    best_acc_list = [] 

    for i in range(k):
        print("Fold: ", i+1)
        train_file_path = 'fold_'+ str(i+1) +'_train.txt'
        val_file_path = 'fold_'+ str(i+1) +'_test.txt'
        fold_num = str(i+1)

        transform_net = TransformNet().to(device=device)
        val_annotation_file = os.path.join(videos_root, val_file_path)

        #freeze the weights of the transformation network
        for param in transform_net.parameters():
            param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        val_dataset = VideoFrameDataset(
            root_path=videos_root,
            annotationfile_path=val_annotation_file,
            num_segments=10,
            frames_per_segment=1,
            imagefile_template='img_{:05d}.jpg',
            transform=preprocess_test,
            test_mode=True)

        val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)
        
            
        #load vis_model and mmtransformer

        vis_model_student=VIS_MODEL(fold_num).to(device=device)

        mm_transformer_student = MultimodalTransformer(visual_dim=512, physiological_dim=512, num_heads=2, hidden_dim=512, num_layers=2, num_classes=2).to(device=device)
        vis_model_student_load_path= '' # path to saved student model for the current fold
        mm_transformer_student_load_path = '' #path to saved student model for the current fold
        
        vis_model_student.load_state_dict(torch.load(vis_model_student_load_path))
        mm_transformer_student.load_state_dict(torch.load(mm_transformer_student_load_path))
        best_fold_acc, val_loss = validate_mmtransformer_dmwl_wtn(vis_model_student,mm_transformer_student,transform_net, val_dataloader, criterion, device)
        best_acc_list.append(best_fold_acc)


        print("Best validation accuracy for each fold: ", best_acc_list)
        print("Average best validation accuracy: ", sum(best_acc_list)/k)
        mean_accuracy = np.mean(best_acc_list)
        std_error = np.std(best_acc_list) / np.sqrt(k)

        print("Mean accuracy: ", mean_accuracy)
        print("Standard error: ", std_error)






if __name__ == '__main__':
    train_k_fold()
    # test_k_fold()

