from video_dataset_mm import  VideoFrameDataset, ImglistToTensor
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
num_epochs = 50
best_epoch = 0
check_every = 1
b_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_acc=0

l_rate = 0.0005

experiment = Experiment(
    api_key="U0t7sSZhwEHvDLko0tJ4kbPH0",
    project_name="biovid",
    
)

parameters = {'batch_size': b_size,
              'learning_rate': l_rate,
              'epochs':num_epochs            
              }
experiment.log_parameters(parameters)




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



# Define a custom collate function
def custom_collate_fn(batch):
    # Assuming batch is a list of (image, label) tuples
    images, labels = zip(*batch)
    # Convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])
    images = [transform(image) for image in images]
    return torch.stack(images), torch.tensor(labels)








# batch_size = 2  # Adjust as needed
num_frames = 5  # Number of frames in each video clip
num_channels = 3  # Number of channels (e.g., RGB)
video_length = 112  # Length of the video in each dimension
num_classes = 2  # Number of classes

# dummy_data = torch.randn(batch_size, num_frames, num_channels, video_length, video_length)  # Example shape


"""
Model definition 
Visual model: R3D-18
Physiological model: Resnet 18 layer MLP

"""










criterion = nn.CrossEntropyLoss()

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




vis_phy_model=VIS_PHY_MODEL().to(device=device)

vis_phy_optimizer = optim.Adam(vis_phy_model.parameters(), lr=l_rate,)
# scheduler = optim.lr_scheduler.StepLR(vis_phy_optimizer, step_size=5, gamma=0.01)
scheduler = ReduceLROnPlateau(vis_phy_optimizer, mode='max', factor=0.01, patience=3, verbose=True)



if __name__ == '__main__':

    videos_root = '/home/livia/work/Biovid/PartB/biovid_classes'
    # videos_root = '/home/livia/work/Biovid/PartB/Video-Dataset-Loading-Pytorch-main/demo_dataset'
    train_annotation_file = os.path.join(videos_root, 'annotations_filtered_peak_2_train.txt')
    val_annotation_file = os.path.join(videos_root, 'annotations_filtered_peak_2_val.txt')




    """ DEMO 3 WITH TRANSFORMS """
    # As of torchvision 0.8.0, torchvision transforms support batches of images
    # of size (BATCH x CHANNELS x HEIGHT x WIDTH) and apply deterministic or random
    # transformations on the batch identically on all images of the batch. Any torchvision
    # transform for image augmentation can thus also be used  for video augmentation.
    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(112),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(112),  # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=train_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=False)
    
    val_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=val_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=True)


    def denormalize(video_tensor):
        """
        Undoes mean/standard deviation normalization, zero to one scaling,
        and channel rearrangement for a batch of images.
        args:
            video_tensor: a (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        """
        inverse_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        return (inverse_normalize(video_tensor) * 255.).type(torch.uint8).permute(0, 2, 3, 1).numpy()


    # frame_tensor = denormalize(frame_tensor)
    # plot_video(rows=1, cols=5, frame_list=frame_tensor, plot_width=15., plot_height=3.,
    #            title='Evenly Sampled Frames, + Video Transform')



    """ DEMO 3 CONTINUED: DATALOADER """
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=b_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)
    





    with experiment.train():
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            
            vis_phy_model.vis_model.train()
            vis_phy_model.phy_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for i,(spec_2d,video_batch, labels) in enumerate(train_dataloader,0):
                vis_phy_optimizer.zero_grad()
                video_batch=video_batch.permute(0, 2, 1, 3, 4)
                out = vis_phy_model.model_out(video_batch,spec_2d)
                labels = labels.to(device)
                t_loss = criterion(out, labels)

                t_loss.backward()
                # vis_phy_model.vis_optim.step()
                # vis_phy_model.phy_optim.step()
                vis_phy_optimizer.step()

                running_loss += t_loss.item()
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0
            
            train_accuracy= 100 * correct / total
            print("*********************************************\n")
            print(f"Accuracy after epoch {epoch + 1}: {train_accuracy}%")
            train_loss= running_loss / 100
            experiment.log_metric('Loss', train_loss,epoch= epoch)
            experiment.log_metric('Accuracy', train_accuracy ,epoch= epoch)
            # last_lr=scheduler.get_last_lr()
            # experiment.log_metric('Learning Rate', last_lr,epoch= epoch)
            if epoch % check_every == 0:
                val_acc, val_loss = validate(vis_phy_model, val_dataloader, criterion, device)
                # print( "Validation accuracy: ", val_acc)
                experiment.log_metric('Val Accuracy', val_acc,epoch= epoch)
                experiment.log_metric('Val Loss', val_loss,epoch= epoch)
                scheduler.step(val_acc)
                current_lr = vis_phy_optimizer.param_groups[0]['lr']
                experiment.log_metric('Learning Rate', current_lr,epoch= epoch)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_save_path = os.path.join(os.getcwd(), 'model_best_feat_concat_fusion.pth')
                    torch.save(vis_phy_model.state_dict(), model_save_path)
                    print('Best model saved at epoch: ', epoch+1)
                    best_epoch = epoch+1


    print("Finished Training")

    train_accuracy = 100 * correct / total
    avg_train_loss = running_loss / len(train_dataloader)
    print(f'Training accuracy: {train_accuracy}%')
    print(f'Training loss: {avg_train_loss}')

    print("Best model saved at epoch: ", best_epoch)
    print("Best validation accuracy: ", best_val_acc)


    


