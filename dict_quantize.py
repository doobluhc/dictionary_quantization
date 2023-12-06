from sklearn.cluster import KMeans,MiniBatchKMeans,MeanShift,AgglomerativeClustering,DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
import torchvision.models as models
from collections import namedtuple


def kmeans_cosine(X, k, max_iters=100, tol=1e-4):
    centroids = [X[np.random.choice(len(X))]]
    for _ in range(1, k):
        distances = pairwise_distances(X, np.array(centroids), metric='cosine')
        min_distances = np.min(distances+1e-8, axis=1)
        new_centroid = X[np.argmax(min_distances)]
        centroids.append(new_centroid)
    for _ in range(max_iters):
        # print(centroids)
        distances = pairwise_distances(X, centroids,metric='cosine')
        # print(distances)
        labels = np.argmin(distances, axis=1)
        new_centroids = []
        for j in range(k):
            # Check if there are any points assigned to the cluster
            if np.any(labels == j):
                new_centroids.append(np.mean(X[labels == j], axis=0))
            else:
                # If no points are assigned, keep the centroid unchanged
                new_centroids.append(centroids[j])
        
        new_centroids = np.array(new_centroids)
        # print(new_centroids)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return labels, centroids

class KMeansQuantizer:
    def __init__(self, model, bitwidth=6, test=False):
        self.num_clusters = 2**bitwidth
        # torch.save(model,'original_model.pth')
        self.codebooks = self.quantize(model)
        torch.save(model,'quantized_model_w_outlier_64.pth')
        # torch.save(self.codebooks,'codebooks.pth')

    
    def kmeans_quantize(self,param,codebook=None,test=True):

        if codebook is None:
            flattened_data = param.view(-1, 1).detach().numpy()
            gm = GaussianMixture(n_components=1, random_state=0).fit(flattened_data)
            log_prob = gm.score_samples(flattened_data)
            
            indices_g = log_prob > -4
            g_values = flattened_data[indices_g]
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
            labels = kmeans.fit_predict(g_values)
            flattened_data[indices_g] = kmeans.cluster_centers_[labels]
                        
            # kmeans = SphericalKMeans(n_clusters=self.num_clusters, random_state=0)
            # labels = np.array(kmeans.cluster(g_values,assign_clusters = True))
            # centroids = np.squeeze(kmeans.means())
            
            # kmeans.fit(g_values)
            # print(labels)
            # print(mshift.cluster_centers_.size)
            
            # centroids = torch.tensor(kmeans.cluster_centers_,dtype=torch.float32).to(param.device)
            # codebook = Codebook(centroids, indices)
        
        if test == True:
            quantized_tensor = torch.tensor(flattened_data,dtype=torch.float32)
            param.data = (quantized_tensor.view_as(param))
        
        return None,None

    @torch.no_grad()
    def quantize(self,model: nn.Module):
        codebooks = dict()
        num_layers = len(list(model.named_parameters()))
        for i,(name, param) in enumerate(model.named_parameters()):
            if 'weight' in name and param.dim() > 1:
                shape = param.size()
                param.requires_grad = False
                indices,centroids = self.kmeans_quantize(param)
                print('progress:',i,'/',num_layers)
                # break
                # param.data = indices.view(shape)
                # # print(param.data)
                # codebooks[name+'_codebook'] = centroids
                # break
        
        return codebooks

class GMMQuantizer:
    def __init__(self, model, bitwidth=5, test=False):
        self.num_clusters = 2**bitwidth
        # torch.save(model,'original_model.pth')
        self.codebooks = self.quantize(model)
        torch.save(model,'gmm_quantized_unet_w_outlier_32.pth')
        # torch.save(self.codebooks,'codebooks.pth')

    
    def gmm_quantize(self,param,codebook=None,test=True):
        if codebook is None:
            flattened_data = param.view(-1, 1).detach().numpy()
            gm = GaussianMixture(n_components=1, random_state=0).fit(flattened_data)
            log_prob = gm.score_samples(flattened_data)
            
            indices_g = log_prob > -4
            g_values = flattened_data[indices_g]
            gmm = GaussianMixture(n_components=self.num_clusters,init_params='k-means++')
            labels = gmm.fit_predict(g_values)
            # print(gmm.means_[labels].size)
            flattened_data[indices_g] = gmm.means_[labels]
            
            # centroids = torch.tensor(kmeans.cluster_centers_,dtype=torch.float32).to(param.device)
            # codebook = Codebook(centroids, indices)
        
        if test == True:
            quantized_tensor = torch.tensor(flattened_data,dtype=torch.float32)
            param.data = (quantized_tensor.view_as(param))
        
        return None,None

    @torch.no_grad()
    def quantize(self,model: nn.Module):
        codebooks = dict()
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                shape = param.size()
                param.requires_grad = False
                indices,centroids = self.gmm_quantize(param)
                # print(1)
                # break
                # param.data = indices.view(shape)
                # # print(param.data)
                # codebooks[name+'_codebook'] = centroids
                # break
        
        return codebooks
        
def get_ssd():
    # original_model = torch.load('original_model.pth')
    original_model = models.resnet18(pretrained=True)
    quantized_model = torch.load('kmeans_quantized_res18_w_outlier_32.pth')
    # codebooks = torch.load('codebooks.pth')
    distances = []
    for (o_name,o_param),(name, param) in zip(original_model.named_parameters(),quantized_model.named_parameters()):
        if 'weight' in name and param.dim() > 1:
            # shape = param.size()
            # centroids = codebooks[name+'_codebook'].detach().numpy()
            # param.data = torch.tensor(centroids[param.data.detach().numpy()],dtype=torch.float32).view(shape)
            squared_distances = torch.sum((o_param.data - param.data) ** 2)
            # print(squared_distances)
            distances.append(squared_distances)

    print(sum(distances)/len(distances))
    return sum(distances)/len(distances)



if __name__ == "__main__":
    # print(2)
    model = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    # model = models.resnet18(pretrained=True)
    quantizer = KMeansQuantizer(model)
    # get_ssd()