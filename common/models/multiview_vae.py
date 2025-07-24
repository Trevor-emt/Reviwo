import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from tqdm import tqdm, trange
from PIL import Image
from .modules.stransformer import STransformer
try:
    from sklearn.cluster import KMeans
except Exception as e:
    print(f"Unable to import KMeans: {e}")

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, init_kmeans=True):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.init_kmeans = init_kmeans

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        if not init_kmeans:
            self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def kmeans_init(self, data):
        """Initialize embeddings using KMeans"""
        print("Start init k-means!")
        flat_inputs = data.reshape(-1, self.embedding_dim).cpu().detach().numpy()
        kmeans = KMeans(n_clusters=self.num_embeddings, n_init=10)
        kmeans.fit(flat_inputs[:min(self.num_embeddings * 2, flat_inputs.shape[0])])
        init_embeddings = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        self.embeddings.weight.data.copy_(init_embeddings)
        print("K-means init successfully!")

    def forward(self, inputs):
        if self.init_kmeans:
            self.kmeans_init(inputs)
            self.init_kmeans = False  # Only initialize once

        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        distances = torch.sum((flat_inputs.unsqueeze(1) - self.embeddings.weight)**2, dim=2)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encoding_onehot = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encoding_onehot.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encoding_onehot, self.embeddings.weight).view(inputs.shape)
        
        # avg_probs = torch.mean(encoding_onehot, dim=0)
        # code_diversity_loss = torch.mean(avg_probs * torch.log(avg_probs + 1e-10))

        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss # + 5 * code_diversity_loss

        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices
    
        
class MultiViewBetaVAE(nn.Module):
    def __init__(self, view_encoder_config, latent_encoder_config, decoder_config, view_cb_config, latent_cb_config, img_size, patch_size, fusion_style: str="plus", use_latent_vq: bool=True, is_latent_ae: bool=True, use_view_vq: bool=True, is_view_ae: bool=True):
        super().__init__()

        view_encoder_config = view_encoder_config.update(
            {
                "n_tokens_per_frame": (img_size // patch_size) ** 2,
                "block_size": (img_size // patch_size) ** 2,
                "vocab_size": 0
            }
        )

        latent_encoder_config = latent_encoder_config.update(
            {
                "n_tokens_per_frame": (img_size // patch_size) ** 2,
                "block_size": (img_size // patch_size) ** 2,
                "vocab_size": 0
            }
        )

        decoder_config = decoder_config.update(
            {
                "n_tokens_per_frame": (img_size // patch_size) ** 2,
                "block_size": (img_size // patch_size) ** 2,
                "vocab_size": 0
            }
        )

        self.view_encoder = STransformer(view_encoder_config)
        self.latent_encoder = STransformer(latent_encoder_config)
        self.decoder = STransformer(decoder_config)
        self.to_patch_embed = nn.Sequential(
            nn.Conv2d(3, view_encoder_config.n_embed, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.view_encoder_output_proj = nn.Linear(view_encoder_config.n_embed, view_cb_config.embed_dim, bias=view_encoder_config.bias)
        self.latent_encoder_output_proj = nn.Linear(latent_encoder_config.n_embed, latent_cb_config.embed_dim, bias=latent_encoder_config.bias)
        
        ### Below is the change
        if fusion_style == 'plus':
            self.decoder_input_proj = nn.Linear(view_cb_config.embed_dim, decoder_config.n_embed, bias=decoder_config.bias)
            # self.decoder_input_proj = nn.Linear(64, decoder_config.n_embed, bias=decoder_config.bias)
        elif fusion_style == 'cat':
            self.decoder_input_proj = nn.Linear(view_cb_config.embed_dim + latent_cb_config.embed_dim, decoder_config.n_embed, bias=decoder_config.bias)
            # self.decoder_input_proj = nn.Linear(128, decoder_config.n_embed, bias=decoder_config.bias)
        else:
            raise NotImplementedError
        
        self.to_pixel = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=img_size//patch_size, w=img_size//patch_size),
            nn.ConvTranspose2d(decoder_config.n_embed, 3, kernel_size=patch_size, stride=patch_size),
        )
        
        #### Here is the change
        self.view_output_head = VectorQuantizer(view_cb_config.n_embed, view_cb_config.embed_dim, view_cb_config.beta, init_kmeans=True) if use_view_vq else nn.Linear(view_cb_config.embed_dim, 2 * view_cb_config.embed_dim)
        
        self.latent_output_head = VectorQuantizer(latent_cb_config.n_embed, latent_cb_config.embed_dim, latent_cb_config.beta, init_kmeans=True) if use_latent_vq else nn.Linear(latent_cb_config.embed_dim, 2 * latent_cb_config.embed_dim)
        
        ### Here is the change
        # self.latent_output_final_proj = nn.Linear(latent_cb_config.embed_dim, 64)
        self.latent_output_final_proj = nn.Identity()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.fusion_style = fusion_style
        self.use_latent_vq = use_latent_vq
        self.is_latent_ae = is_latent_ae
        self.use_view_vq = use_view_vq
        self.is_view_ae = is_view_ae
        
    def encode(self, x, deterministic_view=False, deterministic_latent=False):   
        # x shape (B C H W)
        # to_patch_embed is shared by view and latent encoder
        patch_embed = self.to_patch_embed(x).contiguous()
        h_v = self.view_encoder_output_proj(self.view_encoder(patch_embed))
        h_l = self.latent_encoder_output_proj(self.latent_encoder(patch_embed))
        if self.use_view_vq:
            z_v, view_embed_loss, view_encoding_indices = self.view_output_head(h_v) 
        else:
            z_v_output = self.view_output_head(h_v)
            z_v_mu = torch.tanh(z_v_output[..., :z_v_output.shape[-1] // 2])
            if self.is_view_ae or deterministic_view:
                z_v = z_v_mu
                view_embed_loss = torch.tensor((0), dtype=torch.float).to(z_v.device)
            else:
                z_v_sigma = torch.exp(z_v_output[..., z_v_output.shape[-1] // 2:].clip(-20, 2))
                view_embed_loss = -0.5 * torch.mean((1 + torch.log(z_v_sigma ** 2) - z_v_mu ** 2 - z_v_sigma ** 2))
                if self.training:
                    z_v_dist = torch.distributions.Normal(z_v_mu, z_v_sigma)
                    z_v = z_v_dist.rsample()
                else:
                    z_v_dist = torch.distributions.Normal(z_v_mu, z_v_sigma)
                    z_v = z_v_dist.sample()
            view_encoding_indices = torch.zeros((6)).to(z_v.device) 
                              
        if self.use_latent_vq:
            z_l, latent_embed_loss, latent_encoding_indices = self.latent_output_head(h_l)
            z_l = self.latent_output_final_proj(z_l)
        else:
            z_l_output = torch.tanh(self.latent_output_head(h_l))
            z_l_mu = z_l_output[..., :z_l_output.shape[-1] // 2]
            if self.is_latent_ae or deterministic_latent:
                z_l = z_l_mu
                latent_embed_loss = torch.tensor((0), dtype=torch.float).to(z_l.device)
            else:
                z_l_sigma = torch.exp(z_l_output[..., z_l_output.shape[-1] // 2:].clip(-20, 2))
                latent_embed_loss = -0.5 * torch.mean((1 + torch.log(z_l_sigma ** 2) - z_l_mu ** 2 - z_l_sigma ** 2))
                if self.training:
                    z_l_dist = torch.distributions.Normal(z_l_mu, z_l_sigma)
                    z_l = z_l_dist.rsample()
                else:
                    z_l_dist = torch.distributions.Normal(z_l_mu, z_l_sigma)
                    z_l = z_l_dist.sample()
            latent_encoding_indices = torch.zeros((6)).to(z_l.device) 
        return z_v, view_embed_loss, z_l, latent_embed_loss, (latent_encoding_indices, view_encoding_indices)
    
    def decode(self, z_v, z_l):
        quant = self.fusion(z_v, z_l)
        B = quant.shape[0]
        dec = self.decoder(self.decoder_input_proj(quant)) # (B, H*W, C)
        dec = self.to_pixel(dec).contiguous() # (B, C, H, W)
        dec = dec.reshape(B, -1, *dec.shape[-2:]) # (B, C, H, W)
        return dec
    
    def decode_by_encoding(self, z_v, l_encodings):
        assert self.use_latent_vq is True
        z_l = self.latent_output_final_proj(self.latent_output_head.embeddings(l_encodings))
        z_l = z_l.reshape(z_v.shape[0], -1, z_l.shape[-1])
        return self.decode(z_v, z_l)
    
    def forward(self, x):
        z_v_mu, z_v_sigma, z_l, latent_embed_loss, latent_encoding_indices = self.encode(x)
        z_v_dist = torch.distributions.Normal(z_v_mu, z_v_sigma)
        z_v = z_v_dist.rsample()
        dec = self.decode(z_v, z_l)
        return dec, latent_embed_loss
    
    def fusion(self, z_v, z_l):
        if self.fusion_style == 'plus':
            return z_v + z_l
        elif self.fusion_style == 'cat':
            return torch.cat((z_v, z_l), dim=-1)
        else:
            raise NotImplementedError

        
    def visualize(self, x, save_dir=None):
        self.eval()
        if len(x.shape) == 5:
            B, A, C, H, W = x.shape
            origin_imgs = x.reshape(B * A, C, H, W)
        else:
            origin_imgs = x
            B, C, H, W = x.shape
            A = 1
        
        with torch.no_grad():
            z_v_mu, _, z_l, _, encoding_indices = self.encode(x.reshape(B*A, C, H, W), deterministic_view=False)
            z_v = z_v_mu.clone()
            xrec = self.decode(z_v, z_l)
        
        latent_encoding_indices, view_encoding_indices = encoding_indices
        
        shuffle_indices = torch.tensor(list(range(1, B)) + [0])
        shuffled_z_v = z_v.reshape(B, A, -1)[shuffle_indices, ...].reshape(B*A, -1, z_v.shape[-1])
        shuffled_xrec_v = self.decode(shuffled_z_v, z_l)
        
        shuffle_indices_l = torch.tensor(list(range(1, B)) + [0])
        shuffled_z_l = z_l.reshape(B, A, -1)[shuffle_indices_l, ...].reshape(B*A, -1, z_l.shape[-1])
        shuffled_xrec_l = self.decode(z_v, shuffled_z_l)
        # shuffled_xrec_vl = self.decode(shuffled_z_v, shuffled_z_l)

        rec_imgs = rearrange(xrec, 'b c h w -> c h (b w)').contiguous()
        shuffled_rec_imgs_v = rearrange(shuffled_xrec_v, 'b c h w -> c h (b w)').contiguous()
        shuffled_rec_imgs_l = rearrange(shuffled_xrec_l, 'b c h w -> c h (b w)').contiguous()
        # shuffled_rec_imgs_vl = rearrange(shuffled_xrec_vl, 'b c h w -> c h (b w)').contiguous()
        origin_imgs = rearrange(origin_imgs, 'b c h w -> c h (b w)').contiguous()

        cat_imgs = torch.cat([origin_imgs, rec_imgs, shuffled_rec_imgs_v, shuffled_rec_imgs_l], dim=-2).cpu().detach().numpy() # (B, C, 3*H, T*W)
        cat_imgs = ((cat_imgs + 1) * 127.5).clip(0, 255).astype(np.uint8)
        cat_imgs = cat_imgs.transpose(1, 2, 0)

        if save_dir is not None:
            for i in range(cat_imgs.shape[0]):
                Image.fromarray(cat_imgs[i]).save(save_dir+f'/vis_{i}.png')
        else:
            return cat_imgs, latent_encoding_indices.cpu().numpy(), view_encoding_indices.cpu().numpy()
     

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)
        
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file)) 
        if hasattr(self.latent_output_head, "init_kmeans"):
            self.latent_output_head.init_kmeans = False