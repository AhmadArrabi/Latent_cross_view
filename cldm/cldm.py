import einops
import torch
import torch as th
import torch.nn as nn
import torchvision.models as models
import kornia

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
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from CVUSA_dataset import TARGET_SIZE, W_TARGET, H_TARGET

############################################################################################################
class ControlledUnetModel(UNetModel):
    """
    This class is like a maestro, it organizes and uses the output of the controlNet class
    It is assumed that all results from the forward method in controlnet are available in 
    a list before hand and this class determines how the controlNet influences the Unet

    TODO: experiment with different methods of conditioning, e.g., condition decoder, encoder, 
    middle block, or any other part of the Unet 
    """
    def __init__(self, image_size, in_channels, model_channels, control_seq, control_block, out_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=..., conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=True, disable_self_attentions=None, num_attention_blocks=None, disable_middle_self_attn=False, use_linear_in_transformer=False):
        super().__init__(image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, conv_resample, dims, num_classes, use_checkpoint, use_fp16, num_heads, num_head_channels, num_heads_upsample, use_scale_shift_norm, resblock_updown, use_new_attention_order, use_spatial_transformer, transformer_depth, context_dim, n_embed, legacy, disable_self_attentions, num_attention_blocks, disable_middle_self_attn, use_linear_in_transformer)
        self.control_seq = control_seq
        self.control_block = control_block

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            mid_control = control.pop()
            if self.control_seq=='seq_images':
                mid_control_resized = torch.nn.functional.interpolate(mid_control, size=(h.shape[2], h.shape[3]), mode="bilinear") 
            h += mid_control_resized

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                next_layer_control = control.pop()
                if self.control_seq=='seq_images':
                    next_layer_control_resized = torch.nn.functional.interpolate(next_layer_control, size=(h.shape[2], h.shape[3]), mode="bilinear") 
                h = torch.cat([h, hs.pop() + next_layer_control_resized], dim=1)
                
            h = module(h, emb, context)
            
        h = h.type(x.dtype)
        return self.out(h)
###############################################################################################################
class ResNet18(nn.Module):
    """
    thanks: https://gitlab.com/vail-uvm/geodtr
    """
    def __init__(self, model_channels):
        super().__init__()
        net = models.resnet18(pretrained = True) #weights = RESNET.WEIGHTS to fix the warning

        layers_in = list(net.children())[:3]    #remove pooling layer
        layers_out = list(net.children())[4:-2] #remove classifiers

        #add last conv layer so the channel width becomes flexible (rn it matches the SD model, model_channels = 320)
        layers_conv = [nn.Sequential(nn.Conv2d(layers_out[-1][-1].conv2.in_channels, model_channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(model_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))] 
        self.layers = nn.Sequential(*layers_in, *layers_out, *layers_conv)

    def forward(self, x):
        return self.layers(x)
    
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
        
        self.backbone_base = ResNet18(model_channels) 
        self.zero_convs = nn.ModuleList()

        #TODO: to be changed when determing how to influence the unet, rn it is applied to the decoder which has 13 blocks (control_block = 'decoder' in .yaml file)
        self.zero_convs.append(self.make_zero_conv(in_channels=model_channels, out_channels=model_channels))
        for ch in self.channel_mult:
            for _ in range(3):
                self.zero_convs.append(self.make_zero_conv(in_channels=model_channels, out_channels=ch*model_channels))
    
    def log_polar_transform(self, x:str):
        #TODO: kornia or some way to implement log transformation ... working on it
        pass

    def make_zero_conv(self, in_channels, out_channels):
        return zero_module(conv_nd(self.dims, in_channels, out_channels, 1, padding=0))
    
    def geo_mapping(self, hint_latent, seq_pos):
        """
        Creates a feature map where each vector from the sequence is matched to its x,y position
        args:
        hint_latent: latent representation of image seq, expected shape = (B, Seq, C, H, W)
        seq_pos: x,y position of each image in seq in pixel space, expected shape = (B, Seq, 2)
        """
        desired_height = self.latent_size[2]
        desired_width = self.latent_size[3]

        vertical_pad = (desired_height - hint_latent.shape[3])//2
        horizontal_pad = (desired_width - hint_latent.shape[4])//2

        hint_padded = torch.nn.functional.pad(hint_latent, (horizontal_pad, horizontal_pad,vertical_pad, vertical_pad)).to(self.device)
        
        # pixel to latent space
        latent_ratio = self.img_size/desired_height
        x_shift = (seq_pos[:,:,0]*latent_ratio).round()
        y_shift = (seq_pos[:,:,1]*-latent_ratio).round()
        
        affine_matrix = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=hint_padded.dtype).detach()
        affine_matrix = affine_matrix.unsqueeze(0).repeat(hint_padded.shape[1],1,1)   #for each seq
        affine_matrix = affine_matrix.unsqueeze(0).repeat(hint_padded.shape[0],1,1,1).to(self.device) #for each batch

        affine_matrix[:,:,0,-1] = x_shift
        affine_matrix[:,:,1,-1] = y_shift
        
        #we can do the same as before spliting the tensor and then cat instead of indexing
        hint_shifted = hint_padded.clone()
        for seq in range(self.seq_padding):
            hint_shifted[:,seq,] =  kornia.geometry.transform.warp_affine(hint_padded[:,seq,], affine_matrix[:,seq,], dsize=(desired_height, desired_width))

        mask = hint_shifted < 1e-06 #should be zero but the tranformation may switch 0 to 1e-07
        return ((hint_shifted*mask).sum(dim=1)/mask.sum(dim=1)) #mean without zeros along channel dimension 
    
    def forward(self, cond):
        hint = cond['c_concat'][0]
        #seq_len = cond['c_seq_len'][0] 
        seq_pos = cond['c_seq_pos'][0]
        seq_mask = cond['c_seq_mask'][0]
        
        #BACKBONE
        #method1
        #hint_latent_ = torch.cat([backbone(hint[:,seq,]).unsqueeze(0) for seq, backbone in zip(range(self.seq_padding), self.backbone_block)])
        #hint_latent = einops.rearrange(hint_latent_, ('seq b c h w -> b seq c h w'))

        #mthod2: maybe more reliable as we split the seq into multi tensors and then concat them instead of direct access with indexing
        splited_seq = torch.split(hint, dim=1, split_size_or_sections=1)
        latents = []
        for split in splited_seq:
            latents.append(self.backbone_base.forward(split.squeeze()).unsqueeze(0))
        hint_latent = einops.rearrange(torch.cat(latents), ('seq b c h w -> b seq c h w'))

        #PERSPECTIVE TRANSFORMATION
        #TODO: apply transformation

        #MASKING
        #method 1:
        #for sample in range(hint.shape[0]):
        #    seq_len_sample = seq_len[sample].type(torch.int)
        #    hint_latent[sample, seq_len_sample:].zero_()

        #method 2:
        hint_masked = hint_latent*seq_mask[(..., ) + (None, ) * 3] #equivalent to doing unsqueeze(-1) three times  
        
        #GEOMAPPING
        feature_map = self.geo_mapping(hint_masked, seq_pos)
        #will be removed probably
        #resized_feature_map = torch.nn.functional.interpolate(
        #    feature_map,
        #    size=(self.latent_size[2],
        #          self.latent_size[3]),
        #    mode="bilinear")

        outs = []
        for zero_conv in self.zero_convs:
            outs.append(zero_conv(feature_map))

        return outs
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
                # resize the guided signal to main SD size
                resized_guided_hint = torch.nn.functional.interpolate(
                    guided_hint, 
                    size=(h.shape[2], 
                          h.shape[3]), 
                    mode="bilinear")
                h += resized_guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs

#############################################################################################################
class ControlNet_seq(ControlNet):
    """
    replica of ControlNet but takes condition as sequence and applies the geomapping
    so the feature map from the geomapping is the condition on the controlNet
    """
    def __init__(self, image_size, in_channels, model_channels, hint_channels, num_res_blocks, attention_resolutions, hint_block_config, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=True, disable_self_attentions=None, num_attention_blocks=None, disable_middle_self_attn=False, use_linear_in_transformer=False):
        super().__init__(image_size, in_channels, model_channels, hint_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, conv_resample, dims, use_checkpoint, use_fp16, num_heads, num_head_channels, num_heads_upsample, use_scale_shift_norm, resblock_updown, use_new_attention_order, use_spatial_transformer, transformer_depth, context_dim, n_embed, legacy, disable_self_attentions, num_attention_blocks, disable_middle_self_attn, use_linear_in_transformer)
        self.input_hint_block = instantiate_from_config(hint_block_config)

    def forward(self, x, hint, timesteps, context, cond, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        guided_hint = self.input_hint_block(cond=cond)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                # resize the guided signal to main SD size
                resized_guided_hint = torch.nn.functional.interpolate(
                    guided_hint, 
                    size=(h.shape[2], 
                          h.shape[3]), 
                    mode="bilinear")
                h += resized_guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlSeq_condition(ControlSeq):
    def __init__(self, seq_padding, model_channels, channel_mult, transformation, img_size=512, dims=2, device='cuda'):
        super().__init__(seq_padding, model_channels, channel_mult, transformation, img_size, dims, device)

    def forward(self, cond):
        hint = cond['c_concat'][0]
        seq_len = cond['c_seq_len'][0] 
        seq_pos = cond['c_seq_pos'][0]
        seq_mask = cond['c_seq_mask'][0]
        
        #BACKBONE
        splited_seq = torch.split(hint, dim=1, split_size_or_sections=1)
        latents = []
        for split in splited_seq:
            latents.append(self.backbone_base.forward(split.squeeze()).unsqueeze(0))
        hint_latent = einops.rearrange(torch.cat(latents), ('seq b c h w -> b seq c h w'))

        #PERSPECTIVE TRANSFORMATION
        #TODO: apply transformation

        #MASKING
        hint_masked = hint_latent*seq_mask[(..., ) + (None, ) * 3] #equivalent to doing unsqueeze(-1) three times  
        
        #GEOMAPPING
        feature_map = self.geo_mapping(hint_masked, seq_pos)
        #will be removed probably
        resized_feature_map = torch.nn.functional.interpolate(
            feature_map,
            size=(self.latent_size[2],
                  self.latent_size[3]),
            mode="bilinear")
        
        return resized_feature_map
############################################################################################################
class ControlLDM(LatentDiffusion):
    def __init__(self, control_stage_config,
                control_key,
                only_mid_control,
                control_seq, 
                control_seq_length, 
                control_seq_position,
                control_seq_mask,
                *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.control_seq = control_seq
        self.control_seq_length = control_seq_length
        self.control_seq_position = control_seq_position
        self.control_seq_mask = control_seq_mask

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
        self.control_model.latent_size = x.shape
        control = batch[self.control_key] #hint (seq of gnd imgs)
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)

        if self.control_seq=='single_image':
            control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float()
       
            return x, dict(c_crossattn=[c], c_concat=[control])
        elif self.control_seq=='seq_images':
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
       
            return x, dict(c_crossattn=[c], c_concat=[control], c_seq_len=[seq_length], c_seq_pos=[seq_position], c_seq_mask=[seq_mask])

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
        diffusion_model = self.model.diffusion_model #controlledunetmodule

        cond_txt = torch.cat(cond['c_crossattn'], 1) 

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        elif self.control_seq=='single_image':
            #self.control_model = controlNet
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)] #output of forward process of controlNet
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        elif self.control_seq=='seq_images':
            #self.control_model = ControlSeq
            control = self.control_model.forward(cond=cond)
            control = [c * scale for c, scale in zip(control, self.control_scales)] #output of forward process of ControlSeq
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        elif self.control_seq=='seq_images_one_cond':
            #self.control_model = ControlNet_seq
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, cond=cond)
            control = [c * scale for c, scale in zip(control, self.control_scales)] #output of forward process of controlNet
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
    
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
        #log["control"] = c_cat * 2.0 - 1.0
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
        #log["control"] = c["c_concat"][0][:N] * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        #if plot_diffusion_rows:
        #    # get diffusion row
        #    diffusion_row = list()
        #    z_start = z[:n_row]
        #    for t in range(self.num_timesteps):
        #        if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
        #            t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
        #            t = t.to(self.device).long()
        #            noise = torch.randn_like(z_start)
        #            z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
        #            diffusion_row.append(self.decode_first_stage(z_noisy))
#
        #    diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
        #    diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
        #    diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
        #    diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
        #    log["diffusion_row"] = diffusion_grid
#
        #if sample:
        #    # get denoise row
        #    samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c_cross]},
        #                                             batch_size=N, ddim=use_ddim,
        #                                             ddim_steps=ddim_steps, eta=ddim_eta)
        #    x_samples = self.decode_first_stage(samples)
        #    log["samples"] = x_samples
        #    if plot_denoise_rows:
        #        denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
        #        log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_full = c
            uc_full['c_crossattn'] = [self.get_unconditional_conditioning(N)]
            #uc_full['c_concat'] = torch.zeros_like(c_cat)
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

        #b, c, h, w = cond["c_concat"][0].shape
        h, w = self.control_model.img_size, self.control_model.img_size
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
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
###############################################################################
