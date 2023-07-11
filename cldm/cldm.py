import einops
import torch
import torch as th
import torch.nn as nn
import torchvision.models as models
import kornia
import torchvision

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

############################################################################################################
class ControlledUnetModel(UNetModel):
    """
    This class is like a maestro, it organizes and uses the output of the controlNet class
    It is assumed that all results from the forward method in controlnet are available in 
    a list before hand and this class determines how the controlNet influences the Unet
    """
    #def __init__(self, image_size, in_channels, model_channels, control_seq, out_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=..., conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=True, disable_self_attentions=None, num_attention_blocks=None, disable_middle_self_attn=False, use_linear_in_transformer=False, control_type='decoder_mid'):
    #    super().__init__(image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, conv_resample, dims, num_classes, use_checkpoint, use_fp16, num_heads, num_head_channels, num_heads_upsample, use_scale_shift_norm, resblock_updown, use_new_attention_order, use_spatial_transformer, transformer_depth, context_dim, n_embed, legacy, disable_self_attentions, num_attention_blocks, disable_middle_self_attn, use_linear_in_transformer)
    #    self.control_seq = control_seq
    #    self.control_type = control_type
    
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        self.control_type = 'decoder_mid'
        if self.control_type == 'decoder_mid':
            with torch.no_grad():
                t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
                emb = self.time_embed(t_emb)
                h = x.type(self.dtype)
                #print('INPUT SHAPE: ', h.shape, '*'*20)
                for module in self.input_blocks:
                    h = module(h, emb, context)
                    #print('INPUT BLOCK SHAPE out: ', h.shape, '*'*20)
                    hs.append(h)
                #print('BEFORE MIDDLE BLOCK SHAPE: ', h.shape, '*'*20)
            h = self.middle_block(h, emb, context)
                #print('AFTER MIDDLE BLOCK SHAPE: ', h.shape, '*'*20)

            if control is not None:
                h += control.pop()
                #print('CONTROL AFTER MIDDLE BLOCK SHAPE: ', mid_control.shape, '*'*20)

            for i, module in enumerate(self.output_blocks):
                if only_mid_control or control is None:
                    h = torch.cat([h, hs.pop()], dim=1)
                else:
                    h = torch.cat([h, hs.pop() + control.pop()], dim=1)
                    #print('h shape before cat:', h.shape)
                    #print('hs input block copy popping (TEMP): ', temp.shape)
                    #print('CONTROL OUT BLOCK SHAPE (ADDED TO TEMP): ', next_layer_control.shape, '*'*20)
                    #print('h shape after cat (before input to out block) :', h.shape)

                h = module(h, emb, context)
                #print('AFTER OUTPUT BLOCK SHAPE: ', h.shape, '*'*20)
                #print('CONTROL OUT BLOCK SHAPE: ', next_layer_control_resized.shape, '*'*20)

        elif self.control_type == 'encoder_mid':
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)

            for module in self.input_blocks:
                h = module(h, emb, context)
                temp = control.pop()
                h += temp
                hs.append(h)
            
            h = self.middle_block(h, emb, context)
            if control is not None:
                h += control.pop()
            
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context)

        h = h.type(x.dtype)
        #n = self.out(h)
        #print('-'*100)
        #print(n.mean())
        #print(n.grad)
        #print('FINAL BLOCK SHAPE: ', n.shape, '*'*20)
        return self.out(h)
###############################################################################################################
class ResNet34(nn.Module):
    """
    thanks: https://gitlab.com/vail-uvm/geodtr
    """
    def __init__(self, model_channels):
        super().__init__()
        net = models.resnet34(pretrained = True) #weights = RESNET.WEIGHTS to fix the warning

        layers_in = list(net.children())[:3]    #remove pooling layer
        layers_out = list(net.children())[4:-2] #remove classifiers

        #add last conv layer so the channel width becomes flexible (rn it matches the SD model, model_channels = 320)
        layers_conv = [nn.Sequential(nn.Conv2d(layers_out[-1][-1].conv2.in_channels, model_channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(model_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))] 
        self.layers = nn.Sequential(*layers_in, *layers_out, *layers_conv)

    def forward(self, x):
        return self.layers(x)
    
class ConvAlignBlock(nn.Module):
    """
    Mini conv block to align feature map's dimensions with the ControlledUnet 
    controlNey used a copy of the Unet encoder so this was not needed
    """
    def __init__(self,
                dims,
                channel_mult,
                model_channels,
                control_type='decoder_mid'
                ):
        super().__init__()
        self.dims = dims
        self.channel_mult = channel_mult
        self.model_channels = model_channels
        self.control_type = control_type

        #conv layers
        self.conv_layers = nn.ModuleDict()
        self.conv_layers['downsample_32'] = conv_nd(self.dims, in_channels=channel_mult[0]*model_channels, out_channels=channel_mult[1]*model_channels, kernel_size=3, padding=1, stride=2)
        #self.conv_layers['BN_downsample_32'] = nn.BatchNorm2d(channel_mult[1]*model_channels)
        self.conv_layers['ch_reduction_320'] = conv_nd(self.dims, in_channels=channel_mult[1]*model_channels, out_channels=channel_mult[0]*model_channels, kernel_size=1, padding=0, stride=1)
        #self.conv_layers['BN_ch_reduction_320'] = nn.BatchNorm2d(channel_mult[0]*model_channels)

        self.conv_layers['downsample_16'] = conv_nd(self.dims, in_channels=channel_mult[1]*model_channels, out_channels=channel_mult[2]*model_channels, kernel_size=3, padding=1, stride=2)
        #self.conv_layers['BN_downsample_16'] = nn.BatchNorm2d(channel_mult[2]*model_channels)
        self.conv_layers['ch_reduction_640'] = conv_nd(self.dims, in_channels=channel_mult[2]*model_channels, out_channels=channel_mult[1]*model_channels, kernel_size=1, padding=0, stride=1)
        #self.conv_layers['BN_ch_reduction_640'] = nn.BatchNorm2d(channel_mult[1]*model_channels)

        self.conv_layers['downsample_8'] = conv_nd(self.dims, in_channels=channel_mult[2]*model_channels, out_channels=channel_mult[3]*model_channels, kernel_size=3, padding=1, stride=2)     
        #self.conv_layers['BN_downsample_8'] = nn.BatchNorm2d(channel_mult[3]*model_channels)

        #zero convs for output block
        self.zero_convs = nn.ModuleList()
        self.zero_convs.append(self.make_zero_conv(in_channels=model_channels, out_channels=model_channels))
        for ch in channel_mult:
            for _ in range(3):
                self.zero_convs.append(self.make_zero_conv(in_channels=ch*model_channels, out_channels=ch*model_channels))

        if control_type == 'mid':
            self.zero_conv_middle = self.make_zero_conv(in_channels=channel_mult[-1]*model_channels, out_channels=channel_mult[-1]*model_channels)
    
    def make_zero_conv(self, in_channels, out_channels):
        return zero_module(conv_nd(self.dims, in_channels, out_channels, 1, padding=0))
    
    def forward(self, z):
        if self.control_type == 'mid':
            z = nn.functional.silu(self.conv_layers['BN_downsample_32'](self.conv_layers['downsample_32'](z)))
            z = nn.functional.silu(self.conv_layers['BN_downsample_16'](self.conv_layers['downsample_16'](z)))
            z = nn.functional.silu(self.conv_layers['BN_downsample_8'](self.conv_layers['downsample_8'](z)))
            outs = [self.zero_conv_middle(z)]

        elif (self.control_type == 'decoder_mid') or (self.control_type == 'encoder_mid'):
            outs = {}

            # if you don't want zero convs uncomment
            #outs[0] = z
            #outs[1] = z
            #outs[2] = z
            #z = nn.functional.silu(self.conv_layers['BN_downsample_32'](self.conv_layers['downsample_32'](z)))
            #outs[4] = z
            #outs[5] = z
            #outs[3] = nn.functional.silu(self.conv_layers['BN_ch_reduction_320'](self.conv_layers['ch_reduction_320'](z)))
            #
            #z = nn.functional.silu(self.conv_layers['BN_downsample_16'](self.conv_layers['downsample_16'](z)))
            #outs[7] = z
            #outs[8] = z
            #outs[6] = nn.functional.silu(self.conv_layers['BN_ch_reduction_640'](self.conv_layers['ch_reduction_640'](z)))
            #
            #z = nn.functional.silu(self.conv_layers['BN_downsample_8'](self.conv_layers['downsample_8'](z)))
            #outs[9] =  z
            #outs[10] = z
            #outs[11] = z
            #outs[12] = z

            outs[0] = self.zero_convs[0](z)
            outs[1] = self.zero_convs[1](z)
            outs[2] = self.zero_convs[2](z)

            z = nn.functional.silu(self.conv_layers['downsample_32'](z))
            outs[4] = self.zero_convs[4](z)
            outs[5] = self.zero_convs[5](z)

            outs[3] = self.zero_convs[3](nn.functional.silu(self.conv_layers['ch_reduction_320'](z)))
            
            z = nn.functional.silu(self.conv_layers['downsample_16'](z))
            outs[7] =self.zero_convs[7](z)
            outs[8] =self.zero_convs[8](z)

            outs[6] = self.zero_convs[6](nn.functional.silu(self.conv_layers['ch_reduction_640'](z)))
            
            z = nn.functional.silu(self.conv_layers['downsample_8'](z))
            outs[9] = self.zero_convs[9](z)
            outs[10] = self.zero_convs[10](z)
            outs[11] = self.zero_convs[11](z)
            outs[12] = self.zero_convs[12](z)
            
            outs = [i[1] for i in sorted(list(outs.items()))]
        
        if self.control_type == 'encoder_mid': outs.reverse()

        #for x in outs: print(x.shape)
        return outs
        

class ControlSeq(nn.Module):
    """
    ControlSeq expects a dict input with sequence data, it'll transform the sequence into a 
    latent representation and perform the geomatching scheme to control the SD model
    The output is expected to be a list that is passed to the ControledUnetModel

    args:
    seq_padding: fixed sequence length used after padding (= 14) 
    model_channels: channel width of latent space 
    transformation: not yet implemented (name of tranformation to be used)
    latent_ratio: ratio between latent and aerial image to be used in geomapping (e.g., 512/64 = 0.125)
    dims: dimensions used for Conv layers
    """
    def __init__(
            self,
            seq_padding,
            model_channels,
            channel_mult,
            transformation,
            img_size=512,
            dims=2,
            device='cuda'            
        ):
        super().__init__()

        self.latent_size = None
        self.seq_padding = seq_padding
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.transformation = transformation
        self.img_size = img_size
        self.dims = dims 
        self.device = device
        
        self.backbone_base = ResNet34(model_channels) #TODO: convnext / VAE
        #self.backbone_base = None
        #self.VAE_conv = conv_nd(dims, in_channels=4, out_channels=model_channels, kernel_size=3, padding=1, stride=2)
        # more conv        
        #self.geo_conv = conv_nd(3, in_channels=14, out_channels=1, kernel_size=1) #more methods than 1x1 conv
        self.geo_conv = conv_nd(dims, 14*320, model_channels, 3, padding=1)
        self.conv_align_block = ConvAlignBlock(dims, channel_mult, model_channels)
        
    def log_polar_transform(self, x:str):
        #TODO: kornia or some way to implement log transformation ... working on it
        pass

    def make_zero_conv(self, in_channels, out_channels):
        return zero_module(conv_nd(self.dims, in_channels, out_channels, 1, padding=0))
    
    def geo_mapping(self, hint_latent, seq_pos, B):
        """
        Creates a feature map where each vector from the sequence is matched to its x,y position
        args:
        hint_latent: latent representation of image seq, expected shape = (B, Seq, C, H, W)
        seq_pos: x,y position of each image in seq in pixel space, expected shape = (B, Seq, 2)
        """

        desired_height = self.latent_size[2]
        desired_width = self.latent_size[3]
        #print('HINT LATENT BEFORE ANYTHING JUST AFTER THE RESNET: ', hint_latent.shape)
        #print('DESIRED HEIGHT AND WIDTH IN GEO MAPPING: ', desired_height, desired_width)

        vertical_pad = (desired_height - hint_latent.shape[2])//2
        horizontal_pad = (desired_width - hint_latent.shape[3])//2
        #print('vertical_pad: ', vertical_pad, '*'*20)
        #print('horizontal_pad: ', horizontal_pad, '*'*20)

        hint_padded = torch.nn.functional.pad(hint_latent, (horizontal_pad, horizontal_pad,vertical_pad, vertical_pad)).to(self.device)
        #print("HINT PADDED SHAPE: ", hint_padded.shape)
        
        # pixel to latent space
        latent_ratio = desired_height/self.img_size

        x_shift = (seq_pos[:,0]*latent_ratio).round()
        y_shift = (seq_pos[:,1]*latent_ratio).round()
        #x_shift = (seq_pos[:,:,0]*latent_ratio).round()
        #y_shift = (seq_pos[:,:,1]*latent_ratio).round()
        #x_shift = torch.clip((seq_pos[:,:,0]*latent_ratio).round(), min=-horizontal_pad, max=horizontal_pad)
        #y_shift = torch.clip((seq_pos[:,:,1]*latent_ratio).round(), min=-vertical_pad, max=vertical_pad)
        #print('X SHIFT: ', x_shift.shape, '*'*20)
        #print('Y SHIFT: ', y_shift.shape, '*'*20)
        #print('X SHIFT: ', x_shift, '*'*20)
        #print('Y SHIFT: ', y_shift, '*'*20)

        affine_matrix = torch.tensor([[1, 0, 0.0], [0, 1, 0]], dtype=hint_padded.dtype).to('cuda')
        affine_matrix = affine_matrix.unsqueeze(0).repeat(B*14,1,1)   #for each seq

        affine_matrix[:,0,-1] = x_shift
        affine_matrix[:,1,-1] = -y_shift
        
        #affine_matrix = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=hint_padded.dtype)
        #affine_matrix = affine_matrix.unsqueeze(0).repeat(hint_padded.shape[1],1,1)   #for each seq
        #affine_matrix = affine_matrix.unsqueeze(0).repeat(hint_padded.shape[0],1,1,1).to(self.device) #for each batch

        #affine_matrix[:,:,0,-1] = x_shift
        #affine_matrix[:,:,1,-1] = y_shift
        
        #we can do the same as before spliting the tensor and then cat instead of indexing
        #hint_shifted = hint_padded.clone()
        #for seq in range(self.seq_padding):
        #    hint_shifted[:,seq,] =  kornia.geometry.transform.warp_affine(hint_padded[:,seq,], affine_matrix[:,seq,], dsize=(desired_height, desired_width))
        #print("HINT SHIFTED SHAPE: ", hint_shifted.shape)
        
        #mask = hint_shifted < 1e-06 #should be zero but the tranformation may switch 0 to 1e-07
        #return ((hint_shifted*mask).sum(dim=1)/mask.sum(dim=1)) #mean without zeros along channel dimension 
        #return self.geo_conv(hint_shifted).squeeze()

        return kornia.geometry.transform.warp_affine(hint_padded, affine_matrix, dsize=(desired_height, desired_width))
    
    
    def prep_input(self, x):
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def forward(self, cond):
        #hint = cond['c_concat'][0]
        #seq_pos = cond['c_seq_pos'][0]
        #seq_mask = cond['c_seq_mask'][0]
        
        B = cond['c_concat'][0].shape[0]
        if B == 14: B = 1
        hint = cond['c_concat'][0].view(B*14, 3, 512, 512)
        seq_pos = cond['c_seq_pos'][0].view(B*14, 2)

        #print('-'*75)
        #print('c_concat shape: ', cond['c_concat'][0].shape)
        #print('HINT shape: ', hint.shape)
        guided_hint = self.backbone_base(hint) #[B*14, 320, 32, 32]
        #print('HINT shape: ', guided_hint.shape)
        guided_hint = self.geo_mapping(guided_hint, seq_pos, B) #[2*14, 320, 64, 64]
        #print('HINT shape: ', guided_hint.shape)
        guided_hint = guided_hint.view(B, 14*320, 64, 64) #[2, 14*320, 64, 64]
        #print('HINT shape: ', guided_hint.shape)
        guided_hint = self.geo_conv(guided_hint) #[2, 320, 64, 64]
        #print('HINT shape: ', guided_hint.shape)
        return self.conv_align_block(guided_hint)
        #print('HINT shape: ', guided_hint.shape)
        #print('-'*75)

        #print('-'*100)
        #print('HINT: ', hint)
        
        #BACKBONE
        #splited_seq = torch.split(hint, dim=1, split_size_or_sections=1)
        #latents = []

        #(BS, C, H, W))
        #for i, split in enumerate(splited_seq):
            #latents.append(self.backbone_base.forward(split.squeeze()).unsqueeze(0))
            #split = self.prep_input(split.squeeze())
            #z = (self.backbone_base.encode(split).sample() * 0.18215).detach()
            #if i == 1:
            #    z = 1. / 0.18215 * z
            #    img = self.backbone_base.decode(z)
            #    torchvision.utils.save_image(img, 'DECODED Z !!!!!!!!!.png')
            #z = self.VAE_conv(z)
            #latents.append(z.unsqueeze(0))
        #hint_latent = einops.rearrange(torch.cat(latents), ('seq b c h w -> b seq c h w'))
        #print('-'*100)
        #print('HINT LATENT: ', hint_latent)

        #PERSPECTIVE TRANSFORMATION
        #TODO: apply transformation

        #MASKING
        #hint_masked = hint_latent*seq_mask[(..., ) + (None, ) * 3] #equivalent to doing unsqueeze(-1) three times  
        #print('-'*100)
        #print('HINT MASKED: ', hint_masked)
        
        #GEOMAPPING
        #feature_map = self.geo_mapping(hint_masked, seq_pos)
        #print('AFTER GEOMAPING: ',feature_map.shape, '*'*50)
        #print('-'*100)
        #print('FEATUR MAP: ', feature_map)
        #print('VAE convs: ',self.VAE_conv.weight)
        #print('-'*100)
        #print('GEO new 1x1 added: ',self.geo_conv.weight)

        #return self.conv_align_block(feature_map)
#############################################################################################################
class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch
        
    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs
    
############################################################################################################
class ControlLDM(LatentDiffusion):
    def __init__(self, control_stage_config,
                control_key,
                only_mid_control,
                control_seq, 
                control_seq_length, 
                control_seq_position,
                control_seq_mask,
                control_type,
                *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.model.diffusion_model.control_type = control_type
        self.control_seq = control_seq

        if control_seq!='single_image':
            self.control_seq_length = control_seq_length
            self.control_seq_position = control_seq_position
            self.control_seq_mask = control_seq_mask
            #self.control_model.conv_align_block.control_type = control_type
            #self.control_model.backbone_base = self.first_stage_model

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        """
        returns the latent space representation along with a dictionary of the control input
        In case of sequence (control_seq was 'seq_images'), the dict is as follows:
            {'c_crossattn': [txt embedding tensor],
             'c_concat': [control images],
             'c_seq_len': [number of images in the sequence]
             'c_seq_pos': [relative coordinates of each image in the sequence] (x,y position from the center of the aerial image),
             'c_seq_mask': [mask of seq length]}
        if control_seq was 'single_image', only crossattn and concat are present in the dictionary
        """
        # latent z 'jpg' , CLIP embedding of 'txt' = LDM.get_input()
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs) 
        
        control = batch[self.control_key] #hint (seq of gnd imgs)
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)

        if self.control_seq=='single_image': #standard controlNet
            control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float()
       
            return x, dict(c_crossattn=[c], c_concat=[control])
        
        else: #get the dict of gnd images seq
            if self.control_seq=='seq_images':
                self.control_model.latent_size = x.shape
            #elif self.control_seq=='seq_images_one_cond':
            #    self.control_model.input_hint_block.latent_size = x.shape

            seq_length = batch[self.control_seq_length]
            seq_position = batch[self.control_seq_position]
            seq_mask = batch[self.control_seq_mask]

            if bs is not None:
                seq_length = seq_length[:bs]
                seq_position = seq_position[:bs]
                seq_mask = seq_mask[:bs]

            seq_length = seq_length.to(self.device)
            seq_position = seq_position.to(self.device)
            seq_mask = seq_mask.to(self.device)

            seq_length = seq_length.to(memory_format=torch.contiguous_format).float()
            seq_position = seq_position.to(memory_format=torch.contiguous_format).float()
            seq_mask = seq_mask.to(memory_format=torch.contiguous_format).float()
            
            control = control.to(memory_format=torch.contiguous_format).float()

            #print('*'*50, '\ncross atn', c.shape, c.requires_grad, c.grad, '\n',
            #      'c_concat', control.shape, control.requires_grad, control.grad, '\n',
            #      'c_seq_len ', seq_length.shape, seq_length.requires_grad ,seq_length.grad, '\n',
            #      'c_seq_pos ', seq_position.shape, seq_position.requires_grad ,seq_position.grad, '\n',
            #      'c_seq_mask atn', seq_mask.shape, seq_mask.requires_grad ,seq_mask.grad, '\n',
            #      'Z', x.shape, x.requires_grad, x.grad, '\n')        
            return x, dict(c_crossattn=[c], c_concat=[control], c_seq_len=[seq_length], c_seq_pos=[seq_position], c_seq_mask=[seq_mask])
    
    def disable_SD(self):
        #self.model.diffusion_model.eval()
        #self.model.diffusion_model.train = disabled_train
        for param in self.model.diffusion_model.input_blocks.parameters():
            param.requires_grad = False
            param = param.detach()
    
        #for param in self.model.diffusion_model.middle_block.parameters():
        #    param.requires_grad = False

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """
        cond is expected to be a dict:
        cond = {'c_crossattn': [txt embedding tensor],
                'c_concat': [control images],
                'c_seq_len': [number of images in the sequence]
                'c_seq_pos': [relative coordinates of each image in the sequence] (x,y position from the center of the aerial image),
                'c_seq_mask': [mask of seq length]}
        if control_seq was 'single_image', only crossattn and concat are present in the dictionary
        """
        assert isinstance(cond, dict)

        #self.disable_SD()
        diffusion_model = self.model.diffusion_model #controlledUnetModule

        cond_txt = torch.cat(cond['c_crossattn'], 1) 

        if cond['c_concat'] is None:
            #unconditional
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)

        elif self.control_seq=='single_image':
            #self.control_model = controlNet
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)] #output of forward process of controlNet
            #print('-'*100)
            #for i, c in enumerate(control):
            #    print(i, 'CONTROL: ', c.mean(), '\nSHAPE: ', c.shape, '\nGRAD: ', c.grad)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            #print('-'*100)
            #print('EPS: ', eps, '\nSHAPE: ', eps.shape, '\nGRAD: ', eps.grad)

        elif self.control_seq=='seq_images':
            #self.control_model = ControlSeq
            control = self.control_model(cond=cond)
            #print('CONTROL', '-'*100, control)
            #print(self.control_model.named_parameters())
            control = [c * scale for c, scale in zip(control, self.control_scales)] #output of forward process of ControlSeq
            #for i, c in enumerate(control):
            #    print(i, 'CONTROL: ', c.mean(), '\nSHAPE: ', c.shape, '\nGRAD: ', c.grad)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            #print('PREDICTION: ', eps)
            #print(diffusion_model.named_parameters())

        elif self.control_seq=='seq_images_one_cond':
            #self.control_model = ControlNet_seq
            control = self.control_model(x=x_noisy, timesteps=t, cond=cond)
            #control = [c * scale for c, scale in zip(control, self.control_scales)] #output of forward process of controlNet
            #print('-'*100)
            #for i, c in enumerate(control):
            #    print(i, 'CONTROL: ', c.mean(), '\nSHAPE: ', c.shape, '\nGRAD: ', c.grad)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            
        #s = dict(list(self.named_parameters()))
        #s_ = [(key, val.grad) for key, val in s.items() if 'control_model' in key]
        #print('*'*70)
        #for l in s_: print(l)
        #for key, item in s.items():
        #    print(key)
        #    if(item.requires_grad):
        #        print('requiresed grad')
        #    else: print('does not requiresed grad')
        #    if(item.grad is None):
        #        print('This is also None')
        #    else: print('NOT None')
        
        #print('-'*70)
        #print('EPS gr: ', eps.grad)
        #print('EPS mn: ', eps.mean())
        #print('EPS mx: ', eps.max())

        #for name, param in self.control_model.named_parameters():
        #    #print(name, param.shape, param.requires_grad, param.grad_fn, param.is_leaf, param.grad.mean())
        #    #if 'zero' in name:
        #        print(name, param.shape, param.mean(), param.requires_grad, param.is_leaf)
        #        try:
        #            print(param.grad.mean())
        #        except:
        #            print("no grad found to get the mean of _|_")

        #for name, param in self.model.diffusion_model.named_parameters():
        #    #print(name, param.shape, param.requires_grad, param.grad_fn, param.is_leaf, param.grad.mean())
        #    #if 'zero' in name:
        #    print(name, param.shape, param.mean(), param.requires_grad, param.is_leaf)
        #    try:
        #        print(param.grad.mean())
        #    except:
        #        print("no grad found to get the mean of _|_")


        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log
    
    @torch.no_grad()
    def log_images_seq(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N) #bs = select N samples
        
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["aerial"] = einops.rearrange(batch["jpg"], 'b h w c -> b c h w')

        if unconditional_guidance_scale > 1.0:
            uc_full = c
            uc_full['c_crossattn'] = [self.get_unconditional_conditioning(N)]
            uc_full['c_concat'] = torch.zeros_like(c['c_concat'][0])
            uc_full['c_seq_pos'] = torch.zeros_like(c['c_seq_pos'][0])
            uc_full['c_seq_len'] = torch.zeros_like(c['c_seq_len'][0])
            uc_full['c_seq_mask'] = torch.zeros_like(c['c_seq_mask'][0])

            samples_cfg, _ = self.sample_log(cond=c,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)

        if self.control_seq=='single_image':
            b, c, h, w = cond["c_concat"][0].shape
        elif self.control_seq=='seq_images':
            h, w = self.control_model.img_size, self.control_model.img_size
        elif self.control_seq=='seq_images_one_cond':
            h, w = 512, 512 #self.control_model.input_hint_block.img_size, self.control_model.input_hint_block.img_size

        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate

        #trainable_param = []
        #for name, param in self.control_model.named_parameters():
        #    if 'backbone_base' not in name:
        #        trainable_param.append(param)
        #        print(name, param.shape, param.requires_grad)

        #params = [item for item in params if not 'backbone_base' in item]
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)

        #for name, param in self.control_model.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.shape, param.requires_grad, param.grad_fn)

        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
##########################################################################################################
class ControlNet_seq(ControlNet):
    """
    replica of ControlNet but takes condition as sequence and applies the geomapping
    so the feature map from the geomapping is the condition on the controlNet
    """
    def __init__(self, image_size, in_channels, model_channels, hint_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=True, disable_self_attentions=None, num_attention_blocks=None, disable_middle_self_attn=False, use_linear_in_transformer=False):
        super().__init__(image_size, in_channels, model_channels, hint_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, conv_resample, dims, use_checkpoint, use_fp16, num_heads, num_head_channels, num_heads_upsample, use_scale_shift_norm, resblock_updown, use_new_attention_order, use_spatial_transformer, transformer_depth, context_dim, n_embed, legacy, disable_self_attentions, num_attention_blocks, disable_middle_self_attn, use_linear_in_transformer)
        
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 256, 256, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 256, 320, 3, padding=1, stride=2),
            nn.SiLU()
            #zero_module(conv_nd(dims, 256, model_channels, 3, padding=1, stride=2))
        )
        
        self.input_hint_block_2 = TimestepEmbedSequential(
            zero_module(conv_nd(dims, 14*320, model_channels, 3, padding=1))
        )

    def geo_mapping(self, hint_latent, seq_pos, B):
        desired_height = 64
        desired_width = 64

        vertical_pad = (desired_height - 32)//2
        horizontal_pad = (desired_width - 32)//2

        hint_padded = torch.nn.functional.pad(hint_latent, (horizontal_pad, horizontal_pad,vertical_pad, vertical_pad))
        
        # pixel to latent space
        latent_ratio = 64/512
        seq_pos = seq_pos.view(B*14, 2)

        x_shift = (seq_pos[:,0]*latent_ratio).round()
        y_shift = (seq_pos[:,1]*latent_ratio).round()
        
        affine_matrix = torch.tensor([[1, 0, 0.0], [0, 1, 0]], dtype=hint_padded.dtype).to('cuda')
        affine_matrix = affine_matrix.unsqueeze(0).repeat(B*14,1,1)   #for each seq

        affine_matrix[:,0,-1] = x_shift
        affine_matrix[:,1,-1] = -y_shift

        #print('AFFINE MATRIX SHAPE: ', affine_matrix.shape)
        #print('seq pos SHAPE: ', seq_pos.shape)
        #print('hinr padded SHAPE: ', hint_padded.shape)
            
        return kornia.geometry.transform.warp_affine(hint_padded, affine_matrix, dsize=(desired_height, desired_width))
    
    def forward(self, x, timesteps, cond, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        #print('c_concat shape: ', cond['c_concat'][0].shape)
        B = cond['c_concat'][0].shape[0]
        if B == 14: B = 1
        hint = cond['c_concat'][0].view(B*14, 3, 512, 512)
        context = torch.cat(cond['c_crossattn'], 1) 

        #print('-'*75)
        #print('c_concat shape: ', cond['c_concat'][0].shape)
        #print('HINT shape: ', hint.shape)
        guided_hint = self.input_hint_block(hint, emb, context) #[2*14, 320, 32, 32]
        #print('HINT shape: ', guided_hint.shape)
        guided_hint = self.geo_mapping(guided_hint, cond['c_seq_pos'][0], B) #[2*14, 320, 64, 64]
        #print('HINT shape: ', guided_hint.shape)
        guided_hint = guided_hint.view(B, 14*320, 64, 64)
        #print('HINT shape: ', guided_hint.shape)
        guided_hint = self.input_hint_block_2(guided_hint, emb) #[2, 320, 64, 64]
        #print('HINT shape: ', guided_hint.shape)
        #print('-'*75)

        outs = []
        
        h = x.type(self.dtype)
        
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))
        
        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs

    