from sklearn.cluster import KMeans,MiniBatchKMeans
import numpy as np
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
import torchvision.models as models
from collections import namedtuple



class KMeansQuantizer:
    def __init__(self, model, bitwidth=5, test=False):
        self.num_clusters = 2**bitwidth
        torch.save(model,'original_model.pth')
        self.codebooks = self.quantize(model)
        torch.save(model,'quantized_model.pth')
        torch.save(self.codebooks,'codebooks.pth')

    
    def kmeans_quantize(self,param,codebook=None,test=False):
        if codebook is None:
            # Codebook = namedtuple('Codebook', ['centroids', 'indices'])
            flattened_data = param.view(-1, 1).cpu().numpy()
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
            indices = torch.tensor(kmeans.fit_predict(flattened_data),dtype=torch.uint8).to(param.device)
            centroids = torch.tensor(kmeans.cluster_centers_,dtype=torch.float32).to(param.device)
            # codebook = Codebook(centroids, indices)
        
        if test == True:
            quantized_tensor = codebook.centroids[codebook.indices]
            param.set_(quantized_tensor.view_as(param))
        
        return indices,centroids

    @torch.no_grad()
    def quantize(self,model: nn.Module):
        codebooks = dict()
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                shape = param.size()
                param.requires_grad = False
                indices,centroids = self.kmeans_quantize(param)
                param.data = indices.view(shape)
                # print(param.data)
                codebooks[name+'_codebook'] = centroids
        
        return codebooks
        
def get_ssd():
    original_model = torch.load('original_model.pth')
    quantized_model = torch.load('quantized_model.pth')
    codebooks = torch.load('codebooks.pth')
    distances = []
    for (o_name,o_param),(name, param) in zip(original_model.named_parameters(),quantized_model.named_parameters()):
        if 'weight' in name and param.dim() > 1:
            shape = param.size()
            centroids = codebooks[name+'_codebook'].detach().numpy()
            param.data = torch.tensor(centroids[param.data.detach().numpy()],dtype=torch.float32).view(shape)
            squared_distances = torch.sum((o_param.data - param.data) ** 2)
            # print(squared_distances)
            distances.append(squared_distances)

    print(sum(distances)/len(distances))
    return sum(distances)/len(distances)



if __name__ == "__main__":
    model = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    quantizer = KMeansQuantizer(model)
    get_ssd()