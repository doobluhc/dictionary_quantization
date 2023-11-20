from diffusers import UNet2DConditionModel
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import torch
import graphviz
from torchview import draw_graph

def visulize_unet():
    # graphviz.set_jupyter_format('png')
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    # unet.eval()
    # model_scripted = torch.jit.script(unet) # Export to TorchScript
    # model_scripted.save('unet.pt')
    # print(unet)
    latent_model_input = torch.randn(2, 4, 64, 64)
    t = 1
    text_embeddings = torch.randn(2, 77, 768)
    # dummy_input = (torch.randn(2, 4, 64, 64),1,torch.randn(2, 77, 768))
    inputs = (latent_model_input,t,text_embeddings)
    # output = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    model_graph = draw_graph(unet, input_data=inputs,roll=False)
    # model_graph.resize_graph(scale=5.0)
    model_graph.visual_graph.render('stable_diffusion_unet',format='svg')
    # make_dot(output, params=dict(list(unet.named_parameters()))).render("unet", format="png")


def get_outlier_percent():
    percents = []
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    for name, param in unet.named_parameters():
        if 'weight' in name:
            weights = param.detach().numpy().reshape(-1,1)
            gm = GaussianMixture(n_components=1, random_state=0).fit(weights)
            log_prob = gm.score_samples(weights)
            outliers = weights[log_prob < -4]
            percents.append(outliers.size/weights.size)
    return percents

            
def get_ssd(num_clusters):
    distances = []
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    for name, param in unet.named_parameters():
        if 'weight' in name:
            weights = param.detach().numpy().reshape(-1,1)
            gm = GaussianMixture(n_components=1, random_state=0).fit(weights)
            log_prob = gm.score_samples(weights)
            g = weights[log_prob >= -4]
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(g)

            cluster_assignments = kmeans.labels_
            cluster_centroids = kmeans.cluster_centers_

            replaced_data = cluster_centroids[cluster_assignments]

            squared_distances = np.sum((g - replaced_data) ** 2)
            distances.append(squared_distances)
    return sum(distances)/len(distances)

if __name__ == "__main__":
    visulize_unet()
    # percents = get_outlier_percent()
    # plt.plot(list(range(len(percents))), percents)
    # plt.xlabel("weight_tensor")
    # plt.ylabel("outlier_percent")
    # plt.title("outlier_percent")
    # plt.savefig('outlier_percent.png')
    # nums = [2,4,8,16,32,64]
    # distances = []
    # for num in nums:
    #     print(num)
    #     distances.append(get_ssd(num_clusters=num))

    # plt.bar(nums, distances)
    # plt.xlabel('num_clusters')
    # plt.ylabel('SSD')
    # plt.title('num_clusters VS SSD')
    # plt.savefig('num_clusters_VS_SSD.png')

    
