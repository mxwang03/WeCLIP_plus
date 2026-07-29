import os
from PIL import Image
# ==============================================================================
# CRF 是 CPU/RAM 密集，不主要吃 GPU 显存。
# 限制多线程库线程数，避免 joblib 多进程时线程爆炸。
# ==============================================================================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

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

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


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


def get_color_map_list(num_classes=256):
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)

    for i in range(num_classes):
        r, g, b = 0, 0, 0
        cid = i

        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)
            cid >>= 3

        color_map[i] = np.array([r, g, b])

    return color_map


COLOR_MAP = get_color_map_list(256)


def encode_cmap(label):
    label = np.asarray(label, dtype=np.uint8)
    return COLOR_MAP[label]


def resolve_existing_path(candidates):
    for p in candidates:
        if p is not None and os.path.exists(p):
            return p

    raise FileNotFoundError(
        "None of candidate paths exists:\n" + "\n".join([str(x) for x in candidates])
    )


def validate(model, dataset, test_scales=None):

    _preds, _gts, _msc_preds, cams = [], [], [], []

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    model.cuda(0)
    model.eval()

    num = 0

    _preds_hist = np.zeros((81, 81))
    _msc_preds_hist = np.zeros((81, 81))
    _cams_hist = np.zeros((81, 81))

    if test_scales is None:
        test_scales = [1.0]

    with torch.no_grad():
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


            segs_clip_cat, segs_dino_cat, cam, attn_loss = model(inputs_cat, names, mode='val')

            segs_cat = 0.5 * segs_clip_cat + 0.5 * segs_dino_cat
            segs = segs_cat[0].unsqueeze(0)

            _segs = (segs_cat[0, ...] + segs_cat[1, ...].flip(-1)) / 2
            segs_list.append(_segs)

            _, _, h, w = segs_cat.shape

            for s in test_scales:
                if s != 1.0:
                    _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                    inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                    # 这里同样直接接收 4 个变量
                    segs_clip_cat, segs_dino_cat, cam_cat, attn_loss = model(inputs_cat, names, mode='val')

                    segs_cat = 0.5 * segs_clip_cat + 0.5 * segs_dino_cat

                    _segs_cat = F.interpolate(segs_cat, size=(h, w), mode='bilinear', align_corners=False)
                    _segs = (_segs_cat[0, ...] + _segs_cat[1, ...].flip(-1)) / 2
                    segs_list.append(_segs)

            msc_segs = torch.mean(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0)

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            seg_preds = torch.argmax(resized_segs, dim=1)

            resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

            cams += list(msc_seg_preds.cpu().numpy().astype(np.int16))
            _preds += list(seg_preds.cpu().numpy().astype(np.int16))
            _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
            _gts += list(labels.cpu().numpy().astype(np.int16))

            if num % 2000 == 0:
                _preds_hist, seg_score = evaluate.scores(_gts, _preds, _preds_hist, 81)
                _msc_preds_hist, msc_seg_score = evaluate.scores(_gts, _msc_preds, _msc_preds_hist, 81)
                _cams_hist, cam_score = evaluate.scores(_gts, cams, _cams_hist, 81)

                _preds, _gts, _msc_preds, cams = [], [], [], []

                print("cams score:")
                print(cam_score)
                print("segs score:")
                print(seg_score)
                print("msc segs score:")
                print(msc_seg_score)

            np.save(
                args.work_dir + '/logit/' + name[0] + '.npy',
                {
                    "segs": segs.detach().cpu().numpy(),
                    "msc_segs": msc_segs.detach().cpu().numpy()
                }
            )

    return _gts, _preds, _msc_preds, cams, _preds_hist, _msc_preds_hist, _cams_hist


def crf_proc(config):
    print("crf post-processing...")

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

    def read_label_mask(label_path):
        label = np.array(Image.open(label_path))

        # 防止某些 png 被读成 RGB
        if label.ndim == 3:
            label = label[:, :, 0]

        return label.astype(np.int16)

    def read_pred_mask(pred_path):
        pred = imageio.imread(pred_path)

        # 防止已保存的 prediction 被异常读成 RGB
        if pred.ndim == 3:
            pred = pred[:, :, 0]

        return pred.astype(np.uint8)

    def _job(i):
        name = name_list[i]
        short_name = name[13:] if len(name) > 13 else name

        pred_path = os.path.join(args.work_dir, "prediction", name + ".png")
        cmap_path = os.path.join(args.work_dir, "prediction_cmap", name + ".png")

        image_name = resolve_existing_path([
            os.path.join(images_path, name + ".jpg"),
            os.path.join(images_path, short_name + ".jpg"),
        ])

        image = imageio.imread(image_name).astype(np.float32)

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=-1)

        if "test" in args.eval_set:
            label = image[:, :, 0].astype(np.int16)
        else:
            label_name = resolve_existing_path([
                os.path.join(labels_path, name + ".png"),
                os.path.join(labels_path, short_name + ".png"),
            ])
            label = read_label_mask(label_name)

        if os.path.exists(pred_path):
            pred = read_pred_mask(pred_path)

            if pred.shape != label.shape:
                raise ValueError(
                    "Shape mismatch for {}: pred shape {}, label shape {}".format(
                        name, pred.shape, label.shape
                    )
                )

            return pred, label

        logit_name = resolve_existing_path([
            os.path.join(args.work_dir, "logit", name + ".npy"),
            os.path.join(args.work_dir, "logit", short_name + ".npy"),
        ])

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_segs']

        H, W, _ = image.shape

        logit = torch.FloatTensor(logit)
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0).astype(np.uint8)

        if pred.shape != label.shape:
            raise ValueError(
                "Shape mismatch for {}: pred shape {}, label shape {}".format(
                    name, pred.shape, label.shape
                )
            )

        imageio.imsave(pred_path, np.squeeze(pred).astype(np.uint8))
        imageio.imsave(
            cmap_path,
            encode_cmap(np.squeeze(pred)).astype(np.uint8)
        )

        return pred, label

    n_jobs = 4
    batch_size = 500

    print("CRF n_jobs:", n_jobs)
    print("CRF batch_size:", batch_size)
    print("CRF total images:", len(name_list))

    crf_hist = np.zeros((81, 81))

    for start in range(0, len(name_list), batch_size):
        end = min(start + batch_size, len(name_list))
        print(f"CRF processing {start} - {end} / {len(name_list)}")

        results = joblib.Parallel(
            n_jobs=n_jobs,
            verbose=10,
            pre_dispatch=n_jobs
        )(
            joblib.delayed(_job)(i) for i in range(start, end)
        )

        preds, gts = zip(*results)
        crf_hist, crf_score = evaluate.scores(gts, preds, crf_hist, 81)

        print("current crf score:")
        print(crf_score)

        del results, preds, gts

    _, score = evaluate.scores([], [], crf_hist, 81)

    print("crf score:")
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

    model = WeCLIP_Plus(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        dino_model=cfg.dino_init.dino_model,
        dino_fts_dim=cfg.dino_init.dino_fts_fuse_dim,
        decoder_layers=cfg.dino_init.decoder_layer,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        clip_flag=cfg.clip_init.clip_flag,
        device='cuda'
    )

    trained_state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict=trained_state_dict, strict=False)
    model.eval()

    gts, preds, msc_preds, cams, preds_hist, msc_preds_hist, cams_hist = validate(
        model=model,
        dataset=val_dataset,
        test_scales=[1, 1.5]
    )

    torch.cuda.empty_cache()

    preds_hist, seg_score = evaluate.scores(gts, preds, preds_hist, 81)
    msc_preds_hist, msc_seg_score = evaluate.scores(gts, msc_preds, msc_preds_hist, 81)
    cams_hist, cam_score = evaluate.scores(gts, cams, cams_hist, 81)

    print("cams score:")
    print(cam_score)
    print("segs score:")
    print(seg_score)
    print("msc segs score:")
    print(msc_seg_score)

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