import copy
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from utils.box_ops import box_cxcywh_to_xyxy, box_iou, box_xyxy_to_cxcywh
from utils.constants import smpl_root_idx
from utils.map import build_z_map
from utils.transforms import to_zorder


DEFAULT_GEO2O_CFG = {
    "enabled": False,
    "mosaic_prob": 0.0,
    "copy_blend_prob": 0.0,
    "final_no_aug_epochs": 0,
    "mosaic_datasets": ["coco", "mpii", "crowdpose"],
    "copy_blend_datasets": ["agora", "bedlam"],
    "copy_blend_max_persons": 1,
    "copy_blend_attempts": 8,
    "collision_depth_threshold": 0.5,
    "min_visible_joints": 4,
    "scale_map_patch_size": 28,
    "max_targets": 50,
}

PERSON_KEYS = (
    "boxes",
    "labels",
    "poses",
    "betas",
    "transl",
    "verts",
    "j3ds",
    "j2ds",
    "j2ds_mask",
    "depths",
    "focals",
    "genders",
)


def _cfg_with_defaults(cfg: Optional[Dict]) -> Dict:
    merged = copy.deepcopy(DEFAULT_GEO2O_CFG)
    if cfg:
        merged.update(cfg)
    return merged


def _as_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def _pnum(target: Dict) -> int:
    if "pnum" in target:
        return int(target["pnum"])
    if "boxes" in target:
        return int(len(target["boxes"]))
    return 0


class DenseO2OAugmentor:
    """Training-time target densification for Geo2O-HMR.

    The implementation keeps the inference path untouched. 2D-safe Mosaic
    only applies to projected/pseudo-supervised datasets, while Copy-Blend only
    applies to full-3D synthetic datasets configured in ``copy_blend_datasets``.
    """

    def __init__(self, dbs: Sequence, input_size: int, sat_cfg: Dict, geo2o_cfg: Optional[Dict]):
        self.dbs = list(dbs)
        self.input_size = int(input_size)
        self.sat_cfg = sat_cfg or {"use_sat": False}
        self.cfg = _cfg_with_defaults(geo2o_cfg)
        self.enabled = bool(self.cfg["enabled"])
        self.epoch = 0
        self.num_epochs = None
        self.patch_size = int(self.cfg["scale_map_patch_size"])

        z_depth = math.ceil(math.log2(math.ceil(self.input_size / self.patch_size)))
        self.z_order_map, self.y_coords, self.x_coords = build_z_map(z_depth)

        self.mosaic_datasets = set(self.cfg["mosaic_datasets"])
        self.copy_blend_datasets = set(self.cfg["copy_blend_datasets"])
        self.mosaic_db_indices = [
            i for i, db in enumerate(self.dbs) if getattr(db, "ds_name", None) in self.mosaic_datasets
        ]
        self.copy_blend_db_indices = [
            i for i, db in enumerate(self.dbs) if getattr(db, "ds_name", None) in self.copy_blend_datasets
        ]

    def set_epoch(self, epoch: int, num_epochs: Optional[int] = None):
        self.epoch = int(epoch)
        self.num_epochs = None if num_epochs is None else int(num_epochs)

    def active(self) -> bool:
        if not self.enabled:
            return False
        final_epochs = int(self.cfg["final_no_aug_epochs"])
        if self.num_epochs is not None and final_epochs > 0:
            if self.epoch >= max(0, self.num_epochs - final_epochs):
                return False
        return True

    def __call__(self, owner, db_idx: int, data_idx: int, norm_img: torch.Tensor, target: Dict):
        if not self.active() or _pnum(target) == 0:
            return norm_img, target

        ds_name = target.get("ds")
        is_full3d_copy = _as_bool(target.get("3d_valid", False)) and ds_name in self.copy_blend_datasets
        is_mosaic_safe = ds_name in self.mosaic_datasets or not _as_bool(target.get("3d_valid", False))

        if is_mosaic_safe and self.mosaic_db_indices and random.random() < float(self.cfg["mosaic_prob"]):
            return self._mosaic(owner, norm_img, target)

        if is_full3d_copy and self.copy_blend_db_indices and random.random() < float(self.cfg["copy_blend_prob"]):
            return self._copy_blend(owner, norm_img, target)

        return norm_img, target

    def _sample_from_dbs(self, db_indices: Sequence[int]):
        for _ in range(20):
            db_idx = random.choice(db_indices)
            db = self.dbs[db_idx]
            if len(db) == 0:
                continue
            img, target = db[random.randint(0, len(db) - 1)]
            if _pnum(target) > 0:
                return img, target
        return None, None

    def _mosaic(self, owner, norm_img: torch.Tensor, target: Dict):
        samples = [(norm_img, target)]
        for _ in range(3):
            img_i, tgt_i = self._sample_from_dbs(self.mosaic_db_indices)
            if img_i is None:
                return norm_img, target
            samples.append((img_i, tgt_i))

        size = self.input_size
        half = size // 2
        canvas = norm_img.new_zeros((norm_img.shape[0], size, size))
        offsets = [(0, 0), (half, 0), (0, half), (half, half)]
        transformed_targets = []

        for (img_i, tgt_i), (x0, y0) in zip(samples, offsets):
            resized = F.interpolate(
                img_i.unsqueeze(0), size=(half, half), mode="bilinear", align_corners=False
            ).squeeze(0)
            canvas[:, y0 : y0 + half, x0 : x0 + half] = resized
            transformed_targets.append(self._transform_projected_target(tgt_i, scale=0.5, shift=(x0, y0)))

        merged = self._merge_targets(transformed_targets, template=target, three_d_valid=False)
        merged["geo2o_aug"] = "mosaic"
        return canvas, merged

    def _transform_projected_target(self, target: Dict, scale: float, shift: Tuple[int, int]) -> Dict:
        transformed = copy.deepcopy(target)
        pnum = _pnum(transformed)
        if pnum == 0:
            return transformed

        shift_tensor = torch.tensor(shift, dtype=transformed["boxes"].dtype, device=transformed["boxes"].device)
        shift_norm = shift_tensor / float(self.input_size)

        boxes = transformed["boxes"].clone()
        boxes[:, :2] = boxes[:, :2] * scale + shift_norm
        boxes[:, 2:] = boxes[:, 2:] * scale
        boxes_xyxy = box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
        boxes = box_xyxy_to_cxcywh(boxes_xyxy)
        keep = (boxes[:, 2] > 1.0 / self.input_size) & (boxes[:, 3] > 1.0 / self.input_size)
        transformed["boxes"] = boxes

        if "j2ds" in transformed:
            transformed["j2ds"] = transformed["j2ds"] * scale + shift_tensor
            in_x = (transformed["j2ds"][..., 0] >= 0) & (transformed["j2ds"][..., 0] < self.input_size)
            in_y = (transformed["j2ds"][..., 1] >= 0) & (transformed["j2ds"][..., 1] < self.input_size)
            visible = (in_x & in_y).unsqueeze(-1)
            if "j2ds_mask" in transformed:
                transformed["j2ds_mask"] = transformed["j2ds_mask"] & visible
            else:
                transformed["j2ds_mask"] = visible.repeat(1, 1, 2)

        return self._filter_persons(transformed, keep)

    def _copy_blend(self, owner, norm_img: torch.Tensor, target: Dict):
        out_img = norm_img.clone()
        out_target = copy.deepcopy(target)
        max_persons = int(self.cfg["copy_blend_max_persons"])

        inserted = 0
        for _ in range(max_persons):
            if self._max_targets_reached(out_target):
                break
            accepted = False
            for _ in range(int(self.cfg["copy_blend_attempts"])):
                src_img, src_target = self._sample_from_dbs(self.copy_blend_db_indices)
                if src_img is None or not _as_bool(src_target.get("3d_valid", False)):
                    continue
                person_idx = random.randrange(_pnum(src_target))
                candidate = self._make_copy_blend_candidate(
                    src_img, src_target, person_idx, out_target, out_img.shape[-2:]
                )
                if candidate is None or self._collides(candidate, out_target):
                    continue

                out_img = self._paste_candidate(out_img, src_img, candidate)
                self._append_person(out_target, candidate["person"])
                accepted = True
                inserted += 1
                break

            if not accepted:
                break

        if inserted > 0:
            out_target["geo2o_aug"] = "copy_blend"
            self._rebuild_scale_map(out_target)
        return out_img, out_target

    def _make_copy_blend_candidate(
        self, src_img, src_target: Dict, person_idx: int, target: Dict, target_hw: Tuple[int, int]
    ):
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_target["boxes"][[person_idx]])[0] * self.input_size
        sx0, sy0, sx1, sy1 = src_boxes_xyxy.round().long().tolist()
        sx0, sy0 = max(0, sx0), max(0, sy0)
        sx1, sy1 = min(src_img.shape[-1] - 1, sx1), min(src_img.shape[-2] - 1, sy1)
        if sx1 <= sx0 + 2 or sy1 <= sy0 + 2:
            return None

        src_depth = src_target["depths"][person_idx, 0].clamp(min=1e-3)
        src_focal = src_target["focals"][person_idx, 0].clamp(min=1e-3)
        tgt_focal = self._target_focal(target, dtype=src_depth.dtype, device=src_depth.device)

        tgt_depth = self._sample_target_depth(target, dtype=src_depth.dtype, device=src_depth.device)
        resize = (tgt_focal * src_depth / (src_focal * tgt_depth)).clamp(0.35, 2.0)

        crop_w = max(2, int(round((sx1 - sx0) * float(resize))))
        crop_h = max(2, int(round((sy1 - sy0) * float(resize))))
        if crop_w >= self.input_size or crop_h >= self.input_size:
            return None

        root_px = self._sample_root_pixel(target, dtype=src_depth.dtype, device=src_depth.device)
        src_root_px = src_target["j2ds"][person_idx, smpl_root_idx]
        src_crop_origin = torch.tensor([sx0, sy0], dtype=src_root_px.dtype, device=src_root_px.device)
        scaled_root_offset = (src_root_px - src_crop_origin) * resize
        paste_xy = torch.round(root_px - scaled_root_offset).long()
        px0, py0 = paste_xy.tolist()
        px1, py1 = px0 + crop_w, py0 + crop_h
        target_h, target_w = target_hw
        if px0 < 0 or py0 < 0 or px1 > target_w or py1 > target_h:
            return None

        cam = self._target_camera(target, dtype=src_depth.dtype, device=src_depth.device)
        root_h = torch.stack([root_px[0], root_px[1], torch.ones_like(root_px[0])])
        new_root = tgt_depth * torch.linalg.solve(cam, root_h)

        src_root_3d = src_target["j3ds"][person_idx, smpl_root_idx]
        new_j3ds = src_target["j3ds"][person_idx] - src_root_3d + new_root
        new_verts = src_target["verts"][person_idx] - src_root_3d + new_root
        new_j2ds_h = torch.matmul(new_j3ds, cam.transpose(0, 1))
        new_j2ds = new_j2ds_h[:, :2] / new_j2ds_h[:, 2:].clamp(min=1e-6)

        visible = self._visible_joints(new_j2ds, new_j2ds_h[:, 2], target)
        if int(visible[:22].sum()) < int(self.cfg["min_visible_joints"]):
            return None

        new_box = self._box_from_joints(new_j2ds, visible)
        if new_box is None:
            return None

        person = self._extract_person(src_target, person_idx)
        person.update(
            {
                "boxes": new_box,
                "j2ds": new_j2ds,
                "j2ds_mask": visible.unsqueeze(-1).repeat(1, 2),
                "j3ds": new_j3ds,
                "verts": new_verts,
                "depths": torch.stack([tgt_depth, tgt_depth / tgt_focal]),
                "focals": tgt_focal.view(1),
                "transl": new_root,
            }
        )

        return {
            "person": person,
            "src_crop": (sx0, sy0, sx1, sy1),
            "dst_crop": (px0, py0, px1, py1),
            "box": new_box,
            "depth": tgt_depth,
        }

    def _target_camera(self, target: Dict, dtype, device):
        cam = target["cam_intrinsics"]
        if cam.ndim == 3:
            cam = cam[0]
        return cam.to(dtype=dtype, device=device)

    def _target_focal(self, target: Dict, dtype, device):
        cam = self._target_camera(target, dtype=dtype, device=device)
        return cam[0, 0]

    def _sample_target_depth(self, target: Dict, dtype, device):
        if "depths" in target and len(target["depths"]) > 0:
            depths = target["depths"][:, 0].to(dtype=dtype, device=device)
            base = depths[random.randrange(len(depths))]
            jitter = torch.empty((), dtype=dtype, device=device).uniform_(0.75, 1.35)
            return (base * jitter).clamp(min=1.0)
        return torch.empty((), dtype=dtype, device=device).uniform_(2.0, 8.0)

    def _sample_root_pixel(self, target: Dict, dtype, device):
        img_size = target.get("img_size", torch.tensor([self.input_size, self.input_size]))
        h = float(img_size[0])
        w = float(img_size[1])
        x = torch.empty((), dtype=dtype, device=device).uniform_(0.1 * w, 0.9 * w)
        y = torch.empty((), dtype=dtype, device=device).uniform_(0.1 * h, 0.9 * h)
        return torch.stack([x, y])

    def _visible_joints(self, j2ds: torch.Tensor, z: torch.Tensor, target: Dict):
        img_size = target.get("img_size", torch.tensor([self.input_size, self.input_size]))
        h = float(img_size[0])
        w = float(img_size[1])
        in_x = (j2ds[:, 0] >= 0) & (j2ds[:, 0] < w)
        in_y = (j2ds[:, 1] >= 0) & (j2ds[:, 1] < h)
        return in_x & in_y & (z > 0)

    def _box_from_joints(self, j2ds: torch.Tensor, visible: torch.Tensor):
        visible_joints = j2ds[visible]
        if len(visible_joints) == 0:
            return None
        xy_min = visible_joints.min(dim=0)[0]
        xy_max = visible_joints.max(dim=0)[0]
        box_xyxy = torch.cat([xy_min, xy_max]) / self.input_size
        box = box_xyxy_to_cxcywh(box_xyxy.unsqueeze(0))[0]
        box[2:] *= 1.2
        box_xyxy = box_cxcywh_to_xyxy(box.unsqueeze(0))[0].clamp(0.0, 1.0)
        box = box_xyxy_to_cxcywh(box_xyxy.unsqueeze(0))[0]
        if box[2] <= 1.0 / self.input_size or box[3] <= 1.0 / self.input_size:
            return None
        return box

    def _collides(self, candidate: Dict, target: Dict) -> bool:
        if "boxes" not in target or len(target["boxes"]) == 0:
            return False
        cand_xyxy = box_cxcywh_to_xyxy(candidate["box"].view(1, 4))
        tgt_xyxy = box_cxcywh_to_xyxy(target["boxes"])
        ious, _ = box_iou(cand_xyxy, tgt_xyxy)
        overlap = ious[0] > 0
        if not overlap.any() or "depths" not in target:
            return False
        tgt_depths = target["depths"][:, 0].to(candidate["depth"].device)
        similar_depth = torch.abs(tgt_depths - candidate["depth"]) <= float(self.cfg["collision_depth_threshold"])
        return bool((overlap & similar_depth).any())

    def _paste_candidate(self, target_img: torch.Tensor, src_img: torch.Tensor, candidate: Dict):
        sx0, sy0, sx1, sy1 = candidate["src_crop"]
        px0, py0, px1, py1 = candidate["dst_crop"]
        crop = src_img[:, sy0:sy1, sx0:sx1].unsqueeze(0)
        resized = F.interpolate(crop, size=(py1 - py0, px1 - px0), mode="bilinear", align_corners=False).squeeze(0)
        alpha = self._soft_rect_alpha(py1 - py0, px1 - px0, resized.device, resized.dtype)
        out = target_img.clone()
        out[:, py0:py1, px0:px1] = resized * alpha + out[:, py0:py1, px0:px1] * (1.0 - alpha)
        return out

    @staticmethod
    def _soft_rect_alpha(h: int, w: int, device, dtype):
        feather = min(8, max(1, h // 8), max(1, w // 8))
        if feather <= 1:
            return torch.ones((1, h, w), device=device, dtype=dtype)
        alpha = torch.ones((1, h, w), device=device, dtype=dtype)
        ramp_y = torch.linspace(0.0, 1.0, feather + 2, device=device, dtype=dtype)[1:-1].view(1, feather, 1)
        ramp_x = torch.linspace(0.0, 1.0, feather + 2, device=device, dtype=dtype)[1:-1].view(1, 1, feather)
        alpha[:, :feather, :] *= ramp_y
        alpha[:, -feather:, :] *= ramp_y.flip(1)
        alpha[:, :, :feather] *= ramp_x
        alpha[:, :, -feather:] *= ramp_x.flip(2)
        return alpha

    def _extract_person(self, target: Dict, idx: int):
        person = {}
        for key in PERSON_KEYS:
            if key not in target:
                continue
            value = target[key]
            if isinstance(value, torch.Tensor):
                person[key] = value[idx].clone()
            elif isinstance(value, list):
                person[key] = value[idx]
        if "labels" not in person:
            person["labels"] = torch.tensor(0, dtype=torch.long)
        return person

    def _append_person(self, target: Dict, person: Dict):
        for key, value in person.items():
            if key not in PERSON_KEYS:
                continue
            if isinstance(value, torch.Tensor):
                value = value.unsqueeze(0)
                if key in target and isinstance(target[key], torch.Tensor):
                    target[key] = torch.cat([target[key], value.to(target[key].device)], dim=0)
                else:
                    target[key] = value
            else:
                if key in target and isinstance(target[key], list):
                    target[key].append(value)
                else:
                    target[key] = [value]
        target["pnum"] = _pnum(target) + 1
        target["3d_valid"] = True
        target["detect_all_people"] = bool(target.get("detect_all_people", True))

    def _merge_targets(self, targets: List[Dict], template: Dict, three_d_valid: bool):
        merged = copy.deepcopy(template)
        for key in PERSON_KEYS:
            if key in merged:
                del merged[key]

        merged["pnum"] = sum(_pnum(t) for t in targets)
        merged["img_size"] = torch.tensor([self.input_size, self.input_size], dtype=torch.long)
        merged["resize_rate"] = 1.0
        merged["3d_valid"] = bool(three_d_valid)
        merged["detect_all_people"] = all(bool(t.get("detect_all_people", False)) for t in targets)
        merged["cam_intrinsics"] = copy.deepcopy(template["cam_intrinsics"])

        for key in PERSON_KEYS:
            values = [t[key] for t in targets if key in t and _pnum(t) > 0]
            if not values:
                continue
            if isinstance(values[0], torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], list):
                merged[key] = sum((list(v) for v in values), [])

        if "labels" not in merged and merged["pnum"] > 0:
            merged["labels"] = torch.zeros(merged["pnum"], dtype=torch.long)

        map_hw = (
            math.ceil(self.input_size / self.patch_size),
            math.ceil(self.input_size / self.patch_size),
        )
        self._cap_persons(merged)
        self._rebuild_scale_map(merged, map_hw=map_hw)
        return merged

    def _filter_persons(self, target: Dict, keep: torch.Tensor):
        keep = keep.to(dtype=torch.bool)
        for key in PERSON_KEYS:
            if key not in target:
                continue
            value = target[key]
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == len(keep):
                target[key] = value[keep]
            elif isinstance(value, list) and len(value) == len(keep):
                target[key] = [v for v, k in zip(value, keep.tolist()) if k]
        target["pnum"] = int(keep.sum().item())
        return target

    def _max_targets_reached(self, target: Dict) -> bool:
        max_targets = self.cfg.get("max_targets", None)
        return max_targets is not None and int(max_targets) > 0 and _pnum(target) >= int(max_targets)

    def _cap_persons(self, target: Dict):
        max_targets = self.cfg.get("max_targets", None)
        if max_targets is None or int(max_targets) <= 0 or _pnum(target) <= int(max_targets):
            return
        pnum = _pnum(target)
        keep_idx = torch.randperm(pnum)[: int(max_targets)]
        keep = torch.zeros(pnum, dtype=torch.bool)
        keep[keep_idx] = True
        self._filter_persons(target, keep)

    def _rebuild_scale_map(self, target: Dict, map_hw: Optional[Tuple[int, int]] = None):
        if not self.sat_cfg.get("use_sat", False) or "boxes" not in target:
            return

        if map_hw is None:
            map_hw = target.get("scale_map_hw", None)
        if map_hw is None:
            map_hw = (
                math.ceil(self.input_size / self.patch_size),
                math.ceil(self.input_size / self.patch_size),
            )
        map_h, map_w = int(map_hw[0]), int(map_hw[1])
        scale_map = target["boxes"].new_zeros((map_h, map_w, 2))
        boxes = target["boxes"].detach().clamp(0.0, 1.0)
        centers = boxes[:, :2]
        scales = boxes[:, 2:].norm(p=2, dim=1).clamp(0.0, 1.0)
        ys = torch.floor(centers[:, 1] * self.input_size / self.patch_size).long().clamp(0, map_h - 1)
        xs = torch.floor(centers[:, 0] * self.input_size / self.patch_size).long().clamp(0, map_w - 1)
        for y, x, scale in zip(ys.tolist(), xs.tolist(), scales):
            scale_map[y, x, 0] = 1.0
            scale_map[y, x, 1] = torch.maximum(scale_map[y, x, 1], scale)

        scale_map_z, _, pos_y, pos_x = to_zorder(
            scale_map.cpu(), z_order_map=self.z_order_map, y_coords=self.y_coords, x_coords=self.x_coords
        )
        target["scale_map"] = scale_map_z.to(device=target["boxes"].device, dtype=target["boxes"].dtype)
        target["scale_map_pos"] = {"pos_y": pos_y, "pos_x": pos_x}
        target["scale_map_hw"] = scale_map.shape[:2]
