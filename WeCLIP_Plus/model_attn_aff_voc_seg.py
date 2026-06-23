import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head_seg import SegFormerHead
import clip
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_clip_fts
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder_seg import DecoderTransformer


def Normalize_clip():
    return Compose([
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)


# =====================================================================
# ZPA-Batch (零参数批次级语义原型引导)
# =====================================================================
class CMCA_Module(nn.Module):
    def __init__(self, in_channels=None):
        super(CMCA_Module, self).__init__()
        self.temp_semantic = 0.05
        self.temp_physical = 0.05

    def forward(self, clip_fts, dino_fts):
        b, c, h, w = clip_fts.shape
        N = h * w

        # 1. 提取 Batch 级全局共识原型 (在 CLIP 空间)
        clip_gap = F.adaptive_avg_pool2d(clip_fts, 1).view(b, c)
        clip_gap_norm = F.normalize(clip_gap, p=2, dim=1)

        batch_sim = torch.mm(clip_gap_norm, clip_gap_norm.t())
        batch_attn = F.softmax(batch_sim / self.temp_semantic, dim=-1)

        batch_protos = torch.mm(batch_attn, clip_gap)
        batch_protos_norm = F.normalize(batch_protos, p=2, dim=1).view(b, c, 1, 1)

        # 2. 生成语义前景掩码草图 (Semantic Mask)
        clip_norm = F.normalize(clip_fts, p=2, dim=1)
        semantic_sim = torch.sum(clip_norm * batch_protos_norm, dim=1, keepdim=True)
        semantic_mask = torch.sigmoid(semantic_sim / self.temp_semantic)  # [b, 1, h, w]
        semantic_mask_flat = semantic_mask.view(b, N, 1)

        # 3. 提取纯物理的像素边界亲和力 (在 DINO 空间)
        dino_norm = F.normalize(dino_fts, p=2, dim=1).view(b, c, N)
        dino_aff = torch.bmm(dino_norm.transpose(1, 2), dino_norm)
        dino_aff = F.softmax(dino_aff / self.temp_physical, dim=-1)  # [b, N, N]

        # 4. 跨模态贴合
        refined_mask_flat = torch.bmm(dino_aff, semantic_mask_flat)
        refined_mask = refined_mask_flat.view(b, 1, h, w)

        # 5. 特征提纯
        clip_refined = clip_fts * (1.0 + refined_mask)
        dino_refined = dino_fts * (1.0 + refined_mask)

        return clip_refined, dino_refined


# =====================================================================


class WeCLIP_Plus(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, dino_model=None, dino_fts_dim=768, decoder_layers=3,
                 embedding_dim=256, in_channels=512, dataset_root_path=None, clip_flag=16, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.dino_fts_fuse_dim = dino_fts_dim  # 384 for small, 768 for b, 1024 for l
        self.clip_flag = clip_flag

        self.encoder, _ = clip.load(clip_model, device=device)

        # 【修改点1】参照非seg文件：CLIP 解冻指定层以进行微调
        for name, param in self.encoder.named_parameters():
            if clip_flag == 14 and '23' not in name:
                param.requires_grad = False
            if clip_flag == 16 and "11" not in name:
                param.requires_grad = False

        # 【修改点2】参照非seg文件：使用本地路径加载 DINOv2
        self.dino_encoder = torch.hub.load('/home/share/priv/wangmengxiang/code/dinov2-main', dino_model,
                                           source='local')

        for name, param in self.dino_encoder.named_parameters():
            param.requires_grad = False

        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels, embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=1)

        self.dino_decoder_fts_fuse = SegFormerHead(
            in_channels=[self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim,
                         self.dino_fts_fuse_dim], embedding_dim=self.embedding_dim,
            num_classes=self.num_classes, index=1)

        self.decoder = DecoderTransformer(width=self.embedding_dim * 1, layers=decoder_layers, heads=8,
                                          output_dim=self.num_classes)

        # 【修改点3】参照非seg文件：挂载 ZPA-Batch 模块
        self.cmca = CMCA_Module(self.embedding_dim)

        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers,
                                reshape_transform=reshape_transform)
        self.root_path = os.path.join(dataset_root_path, 'JPEGImages')
        self.cam_bg_thres = 1
        self.encoder.eval()
        self.iter_num = 0
        self.require_all_fts = True

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        for param in list(self.dino_decoder_fts_fuse.parameters()):
            param_groups[3].append(param)

        return param_groups

    def forward(self, img, img_names='2007_000032', mode='train'):
        b, c, h, w = img.shape
        self.encoder.eval()
        self.iter_num += 1

        # 【修改点4】参照非seg文件：因为 CLIP 部分层已解冻，移除 torch.no_grad() 允许梯度回传
        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True, clip_flag=self.clip_flag)

        with torch.no_grad():
            dino_img_h, dino_img_w = (h // 14) * 14, (w // 14) * 14
            dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)
            dino_ftses = self.dino_encoder.forward_features(dino_img)
            dino_fts = dino_ftses['x_norm_patchtokens']

        fts_all_stack = torch.stack(fts_all, dim=0)  # (11, hw, b, c)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h // self.clip_flag,
                                                w // self.clip_flag)  # (11, b, c, h, w)

        all_img_tokens = all_img_tokens[-1].unsqueeze(0)

        fts = self.decoder_fts_fuse(all_img_tokens)

        _, _, fts_h, fts_w = fts.shape  # 24

        if isinstance(dino_fts, list):
            for d_i, dino_fts_single in enumerate(dino_fts):
                dino_fts_single = dino_fts_single.reshape([b, dino_img_h // 14, dino_img_w // 14, -1]).permute(0, 3, 1,
                                                                                                               2)
                dino_fts[d_i] = dino_fts_single

            dino_fts = torch.stack(dino_fts)
            dino_fts = self.dino_decoder_fts_fuse(dino_fts)
        else:
            dino_fts = dino_fts.reshape([b, dino_img_h // 14, dino_img_w // 14, -1]).permute(0, 3, 1, 2)
            _, _, dino_h, dino_w = dino_fts.shape  # 32
            dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))

        dino_fts = F.interpolate(dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)

        # 【修改点5】参照非seg文件：状态感知延迟门控特征增强
        if not self.training or self.iter_num > 10000:
            fts_enhanced, dino_fts_enhanced = self.cmca(fts, dino_fts)
        else:
            fts_enhanced, dino_fts_enhanced = fts, dino_fts

        seg_clip, seg_attn_weight_list_clip = self.decoder(fts_enhanced)

        seg_dino, seg_attn_weight_list_dino = self.decoder(dino_fts_enhanced)

        return seg_clip, seg_dino


if __name__ == "__main__":
    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    model = WeCLIP_Plus('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    model._param_groups()
    dummy_input = torch.rand(2, 3, 512, 512)
    model(dummy_input)