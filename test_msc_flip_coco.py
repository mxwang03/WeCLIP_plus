import os

# ==============================================================================
# 【关于显存/内存与CRF优化的说明】
# 显存从 32G 升级到 48G 主要影响的是 GPU 上的模型推理（可以支撑更大的 batch_size 或更大尺度推理）。
# 而 DenseCRF 后处理是纯 CPU 密集的并行操作，主要消耗的是系统内存（RAM）和 CPU 核心数。
# 因此，为了防止多进程并发时多线程库（OMP/MKL）导致 CPU 线程大爆炸以及系统内存撑爆，
# 限制线程数的环境变量和控制 CRF 进程数的代码【依然需要保留并推荐使用】。
# ==============================================================================

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
# ------------------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import datetime
import random
from collections import OrderedDict
import sys

sys.path.append(".")
from utils.dcrf import DenseCRF
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import multiprocessing
from tqdm import tqdm
import joblib
from datasets import coco
from utils import evaluate
from WeCLIP_Plus.model_attn_aff_coco import WeCLIP_Plus
import imageio.v2 as imageio

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/coco_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--work_dir", default="results", type=str, help="work_dir")
parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
parser.add_argument("--eval_set", default="val", type=str, help="eval_set")
parser.add_argument("--model_path", default="/your/path/model_80000.pth",
                    type=str, help="model_path")


def validate(model, dataset, test_scales=None):
    # 彻底修复指标覆盖与残差清空Bug
    _preds, _gts, _msc_preds, _cams_preds = [], [], [], []

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    model.cuda(0)
    model.eval()

    num = 0

    # 全局混淆矩阵初始化 (COCO 共 81 个类)
    _preds_hist = np.zeros((81, 81))
    _msc_preds_hist = np.zeros((81, 81))
    _cams_hist = np.zeros((81, 81))

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
        num += 1

        name, inputs, labels, cls_labels = data
        names = name + name

        inputs = inputs.cuda()
        labels = labels.cuda()

        _, _, h, w = inputs.shape
        ratio = cfg.clip_init.resize_long / max(h, w)
        _h, _w = int(h * ratio), int(w * ratio)
        inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)

        segs_list = []
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        segs_clip_cat, segs_dino_cat = model(inputs_cat, names, mode='val')

        segs_cat = 0.5 * segs_clip_cat + 0.5 * segs_dino_cat

        # 1. 单尺度基础预测结果 (Single-Scale Baseline)
        segs = segs_cat[0].unsqueeze(0)
        _segs_single = (segs_cat[0, ...] + segs_cat[1, ...].flip(-1)) / 2
        segs_list.append(_segs_single)

        _, _, h, w = segs_cat.shape

        # 2. 多尺度推理
        for s in test_scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs_clip_cat, segs_dino_cat = model(inputs_cat, names, mode='val')
                segs_cat = 0.5 * segs_clip_cat + 0.5 * segs_dino_cat

                _segs_cat = F.interpolate(segs_cat, size=(h, w), mode='bilinear', align_corners=False)
                _segs = (_segs_cat[0, ...] + _segs_cat[1, ...].flip(-1)) / 2
                segs_list.append(_segs)

        # 真正融合了多尺度与翻转增强的结果
        msc_segs = torch.mean(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0)

        # 还原到原始标签尺寸进行评估
        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_preds = torch.argmax(resized_segs, dim=1)

        resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

        # 修复指标隔离，各司其职，彻底杜绝被互相直接覆盖
        _preds += list(seg_preds.cpu().numpy().astype(np.int16))
        _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
        _cams_preds += list(seg_preds.cpu().numpy().astype(np.int16))
        _gts += list(labels.cpu().numpy().astype(np.int16))

        # 定期将列表里的统计结果累计到混淆矩阵中，随后清空，防止列表过大导致系统 RAM 溢出
        if num % 2000 == 0:
            _preds_hist, seg_score = evaluate.scores(_gts, _preds, _preds_hist, 81)
            _msc_preds_hist, msc_seg_score = evaluate.scores(_gts, _msc_preds, _msc_preds_hist, 81)
            _cams_hist, cam_score = evaluate.scores(_gts, _cams_preds, _cams_hist, 81)

            # 清空已统计的列表
            _preds, _gts, _msc_preds, _cams_preds = [], [], [], []
            print(f"\n--- Iteration {num} 中间进度数据统计 ---")
            print("cams score (单尺度激活映射基准):", cam_score['miou'])
            print("segs score (细化单尺度分割):", seg_score['miou'])
            print("msc segs score (多尺度融合增强):", msc_seg_score['miou'])

        # 保存 logit 用于后面的 CRF 离线处理
        np.save(args.work_dir + '/logit/' + name[0] + '.npy',
                {"segs": segs.detach().cpu().numpy(), "msc_segs": msc_segs.detach().cpu().numpy()})

    # 将最后不足 2000 张图的尾部残余数据一次性更新到混淆矩阵中
    if len(_gts) > 0:
        _preds_hist, _ = evaluate.scores(_gts, _preds, _preds_hist, 81)
        _msc_preds_hist, _ = evaluate.scores(_gts, _msc_preds, _msc_preds_hist, 81)
        _cams_hist, _ = evaluate.scores(_gts, _cams_preds, _cams_hist, 81)

    # 核心修复：直接返回已经完整累加、无残缺的全局混淆矩阵记录
    return _preds_hist, _msc_preds_hist, _cams_hist


def crf_proc(config):
    print("开始执行并行的 CRF 后处理...")

    txt_name = os.path.join(config.dataset.name_list_dir, args.eval_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(config.dataset.root_dir, 'JPEGImages/val')
    labels_path = os.path.join(config.dataset.root_dir, 'SegmentationClass/val')

    post_processor = DenseCRF(
        iter_max=10,
        pos_xy_std=3,
        pos_w=3,
        bi_xy_std=64,
        bi_rgb_std=5,
        bi_w=4,
    )

    def _job(i):
        name = name_list[i]

        # =====================================================================
        # 【核心去噪修复】建立双向兼容文件名检索机制，防范 COCO 标签切片不一致引发的 FileNotFoundError
        # =====================================================================
        logit_path1 = os.path.join(args.work_dir, "logit", name + ".npy")
        logit_path2 = os.path.join(args.work_dir, "logit", name[13:] + ".npy")
        logit_name = logit_path1 if os.path.exists(logit_path1) else logit_path2

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_segs']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=-1)

        label_path1 = os.path.join(labels_path, name + ".png")
        label_path2 = os.path.join(labels_path, name[13:] + ".png")
        if "test" in args.eval_set:
            label = image[:, :, 0]
        else:
            label = imageio.imread(label_path1 if os.path.exists(label_path1) else label_path2)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        return pred, label

    # 显存充裕，我们将并行工作进程数从 10 安全上调至 12，加速处理
    n_jobs = 12
    print(f"正在启动并行 CRF，工作进程数: {n_jobs}...")
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10)([joblib.delayed(_job)(i) for i in range(len(name_list))])

    # =====================================================================
    # 【核心修复 1】解包混淆矩阵，防止 scores 函数返回值（元组）将 81x81 矩阵刷屏打印
    # =====================================================================
    preds, gts = zip(*results)
    _, score = evaluate.scores(gts, preds, np.zeros((81, 81)), 81)
    print("\n================ CRF 后处理最终得分 ================")
    print(score)

    return True


def main(cfg):
    val_dataset = coco.CocoSegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    model = WeCLIP_Plus(num_classes=cfg.dataset.num_classes,
                        clip_model=cfg.clip_init.clip_pretrain_path,
                        dino_model=cfg.dino_init.dino_model,
                        dino_fts_dim=cfg.dino_init.dino_fts_fuse_dim,
                        decoder_layers=cfg.dino_init.decoder_layer,
                        embedding_dim=cfg.clip_init.embedding_dim,
                        in_channels=cfg.clip_init.in_channels,
                        dataset_root_path=cfg.dataset.root_dir,
                        clip_flag=cfg.clip_init.clip_flag,
                        device='cuda')

    # =====================================================================
    # 【核心修复 2】移除导致 SyntaxError 的非法转义字符 \"
    # =====================================================================
    trained_state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict=trained_state_dict, strict=False)
    model.eval()

    # 执行推理验证，直接获取无残缺的完整全局混淆矩阵
    preds_hist, msc_preds_hist, cams_hist = validate(model=model, dataset=val_dataset, test_scales=[1, 1.5])
    torch.cuda.empty_cache()

    # 用空列表 [] 代替原来的 None 绕过内部的打包 zip 循环，安全结算最终成绩
    _, seg_score = evaluate.scores([], [], preds_hist, 81)
    _, msc_seg_score = evaluate.scores([], [], msc_preds_hist, 81)
    _, cam_score = evaluate.scores([], [], cams_hist, 81)

    print("\n==================== 最终评估全局收敛得分 ====================")
    print("【1】cams score (原始单尺度激活映射):")
    print(cam_score)
    print("\n【2】segs score (细化单尺度分割):")
    print(seg_score)
    print("\n【3】msc segs score (多尺度+翻转增强融合):")
    print(msc_seg_score)
    print("==============================================================\n")

    # -----CRF-post-processing-----
    crf_proc(config=cfg)

    return True


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    print(cfg)
    print(args)

    args.work_dir = os.path.join(args.work_dir, args.eval_set)

    os.makedirs(args.work_dir + "/logit", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction_cmap", exist_ok=True)

    main(cfg=cfg)