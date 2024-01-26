from dataset.video_dataset_mm import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from torchvision.models.video import r3d_18
from torchvision import models
from tqdm import tqdm

"""
Training settings

"""
num_epochs = 150
best_epoch = 0
check_every = 1
b_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_acc=0





def validate(vis_model,physio_model, val_dataloader, criterion, device):
    # Validation phase
    vis_model.eval() 
    physio_model.eval() 
    val_correct = 0
    val_total = 0
    val_vis_loss = 0.0
    val_physio_loss = 0.0

    with torch.no_grad():
        for val_data in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Validation'):
            spec_2d,val_inputs, val_labels = val_data
            # val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            # val_inputs = val_inputs.permute(0, 2, 1, 3, 4)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)

            val_physio_outputs = physio_model(spec_2d)
            # val_vis_outputs = vis_model(val_inputs)

            val_physio_loss += criterion(val_physio_outputs, val_labels)
            # val_vis_loss += criterion(val_vis_outputs, val_labels).item()

            # val_both_outputs = val_physio_outputs + val_vis_outputs

            _,val_predicted = torch.max(val_physio_outputs.data, 1)

            # _, val_both_predicted = torch.max(val_both_outputs.data, 1)
            
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_physio_loss)) / len(val_dataloader)
    print(f'Validation accuracy: {val_accuracy}%')
    print(f'Validation loss: {avg_val_loss}')
    return val_accuracy



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



visual_model = r3d_18(pretrained=True, progress=True)
visual_model.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
visual_model.fc = nn.Linear(512, num_classes)

visual_model = visual_model.to(device)


#load resnet18 model from torchvision
# create cnn model class with resnet18 as base and change the input channels to 1 and output to 2 classes


class PhysioResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(PhysioResNet18, self).__init__()
        # Load the pre-trained ResNet18 model
        resnet = models.resnet18(pretrained=True)
        
        # Modify the first layer to accept 1 input channel
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Copy the remaining layers from the pre-trained model
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Define the new classifier (fully connected layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)  # 512 is the number of features after the last convolutional layer
        # self.dropout = nn.Dropout(p=0.5)
        # self.fc2= nn.Linear(256, 64)
        # self.fc3 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x= self.dropout(x)
        # x= self.fc2(x)
        # out= self.fc3(x)

        return x








physio_model = PhysioResNet18(num_classes=2)


physio_model = physio_model.to(device)




criterion = nn.CrossEntropyLoss()
# vis_optimizer = optim.SGD(visual_model.parameters(), lr=0.0005, momentum=0.9)
physio_optimizer = optim.SGD(physio_model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(physio_optimizer, step_size=5, gamma=0.01)


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
    






    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        visual_model.train()
        physio_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i,(spec_2d,video_batch, labels) in enumerate(train_dataloader,0):
            
            physio_optimizer.zero_grad()
            
            # video_batch=video_batch.permute(0, 2, 1, 3, 4)
            
            # video_batch = video_batch.to(device)
            labels = labels.to(device)
            spec_2d = spec_2d.to(device= device, dtype=torch.float)


            # vis_outputs = visual_model(video_batch)
            # vis_loss = criterion(vis_outputs, labels)
            
            physio_outputs = physio_model(spec_2d)
            physio_loss = criterion(physio_outputs, labels)

            # vis_loss.backward()
            # vis_optimizer.step()

            physio_loss.backward()
            physio_optimizer.step()

            
            # running_loss += vis_loss.item()
            running_loss += physio_loss.item()
            
            # both_outputs = vis_outputs + physio_outputs
            # _, both_predicted = torch.max(both_outputs.data, 1)
            _, physio_predicted = torch.max(physio_outputs.data, 1)
            total += labels.size(0)
            correct += (physio_predicted == labels).sum().item()


            # _, predicted = torch.max(vis_outputs.data, 1)
            
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

            scheduler.step()    
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
        print("*********************************************\n")
        print(f"Accuracy after epoch {epoch + 1}: {100 * correct / total}%")
        if epoch % check_every == 0:
            val_acc = validate(visual_model,physio_model, val_dataloader, criterion, device)
            # print( "Validation accuracy: ", val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_save_path = os.path.join(os.getcwd(), 'model_best_phy.pth')
                torch.save(physio_model.state_dict(), model_save_path)
                print('Best model saved at epoch: ', epoch+1)
                best_epoch = epoch+1


    print("Finished Training")

    train_accuracy = 100 * correct / total
    avg_train_loss = running_loss / len(train_dataloader)
    print(f'Training accuracy: {train_accuracy}%')
    print(f'Training loss: {avg_train_loss}')

    print("Best model saved at epoch: ", best_epoch)
    print("Best validation accuracy: ", best_val_acc)


    


