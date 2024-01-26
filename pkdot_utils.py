import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns




class CustomLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(CustomLoss, self).__init__()
        self.weight = weight

    def forward(self, frob_loss):
        return torch.tensor(frob_loss * self.weight, requires_grad=True)




def cosine_similarity_matrix_rows(matrix):
    similarity_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))
    
    for i, row1 in enumerate(matrix):
        for j, row2 in enumerate(matrix):
            dot_product = np.dot(row1, row2)
            norm_row1 = np.linalg.norm(row1)
            norm_row2 = np.linalg.norm(row2)
            similarity_matrix[i, j] = dot_product / (norm_row1 * norm_row2)
    
    return similarity_matrix
# 
# 
def cosine_similarity_matrix_transpose(matrix):
    transpose_matrix = matrix.T
    similarity_matrix = np.dot(matrix, transpose_matrix)
    norm_matrix = np.linalg.norm(matrix, axis=1)[:, np.newaxis]
    norm_transpose = np.linalg.norm(transpose_matrix, axis=0)[np.newaxis, :]
    similarity_matrix /= np.dot(norm_matrix, norm_transpose)
    return similarity_matrix


def cosine_similarity_matrix_transpose_torch(matrix):
    transpose_matrix = matrix.t()
    similarity_matrix = torch.mm(matrix, transpose_matrix)
    norm_matrix = torch.norm(matrix, dim=1, keepdim=True)
    norm_transpose = torch.norm(transpose_matrix, dim=0, keepdim=True)
    similarity_matrix.div_(torch.mm(norm_matrix, norm_transpose))
    return similarity_matrix


def plot_simmat(sim_mat_teacher, sim_mat_student, epoch, fold_num):
    # plot the similarity matrices
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(sim_mat_teacher)
    axs[1].imshow(sim_mat_student)   
    # plt.show()

    #save the similarity matrices as images
    fig.savefig(f'sim_mat_{epoch}_{fold_num}.png')



def plot_mds(matrix, title):
    embedding = MDS(n_components=2)
    mds = embedding.fit_transform(matrix)
    plt.figure(figsize=(10, 8))
    plt.scatter(mds[:, 0], mds[:, 1])
    plt.title(title)
    plt.show()


def plot_tsne(data, labels, random_state=0):
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=random_state)
    tsne_results = tsne.fit_transform(data)

    # Create a DataFrame to make plotting easier
    df = pd.DataFrame(data = {
        'x': tsne_results[:,0],
        'y': tsne_results[:,1],
        'label': labels
    })

    # Set the figure size
    plt.figure(figsize=(10,10))

    # Use seaborn to create a scatterplot with different colors for each label
    sns.scatterplot(
        x="x", y="y",
        hue="label",
        palette=sns.color_palette("hsv", len(df['label'].unique())),
        data=df,
        legend="full",
        alpha=0.9
    )

    # Set the title of the plot
    plt.title('t-SNE visualization of the data')

    # Show the plot
    plt.show()

def denormalize(video_tensor):
    inverse_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inverse_normalize(video_tensor) * 255.).type(torch.uint8).permute(0, 2, 3, 1).numpy()