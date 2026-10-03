# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image
import torch.nn as nn
import torch.nn.functional as F

from .modeling.sam2_base import SAM2Base

from .utils.transforms import SAM2Transforms


class SAM2ImagePredictor:
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        # 新增代码-----------------------------------------------------------------------------------------------------------

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps
        # 这里定义了3个特征层级的尺寸，分别是：
        # 最高分辨率特征: hires_size x hires_size
        # 中等分辨率特征: (hires_size//2) x (hires_size//2)
        # 最低分辨率特征: (hires_size//4) x (hires_size//4)
        hires_size = self.model.image_size // 4
        self._bb_feat_sizes = [[hires_size // (2**k)]*2 for k in range(3)]
        # self._bb_feat_sizes = [
        #     (256, 256),



        #     (128, 128),
        #     (64, 64),
        # ]
        # ---------- Decoder projection heads (Semantic Views) ----------
        C_base = 256
        self._proj_align = nn.ModuleDict({
            "128": nn.Conv2d(32, C_base, 1),
            "64": nn.Conv2d(64, C_base, 1),
            "32": nn.Conv2d(256, C_base, 1),
        }).to(self.device)
        self._semantic_gate = nn.Sequential(
            nn.Conv2d(C_base * 3, C_base, 1),
            nn.Sigmoid()
        ).to(self.device)
        self._decoder_proj = nn.ModuleDict({
            "D0": nn.Conv2d(C_base, 768, 1),
            "D1": nn.Conv2d(C_base, 384, 1),
            "D2": nn.Conv2d(C_base, 192, 1),
            "D3": nn.Conv2d(C_base, 96, 1),
        }).to(self.device)


    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2ImagePredictor): The loaded model.
        """
        from .build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def set_image(
        self,
        image: Union[np.ndarray, Image],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = \
            {"image_embed": feats[-1], # 主图像嵌入 tensor
             "high_res_feats": feats[:-1]} # 高分辨率特征列表
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[Union[np.ndarray]],
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.

        Arguments:
          image_list (List[np.ndarray]): The input images to embed in RGB format. The image should be in HWC format if np.ndarray
          with pixel values in [0, 255].
        """
        self.reset_predictor()
        assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            assert isinstance(
                image, np.ndarray
            ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
            self._orig_hw.append(image.shape[:2])
        # Transform the image to the form expected by the model
        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        # 特征被分为：
        # image_embed: 1个主要图像嵌入特征（最低分辨率）
        # high_res_feats: 2个高分辨率特征向量列表
        # 实际上，虽然注释说是"三个特征向量"，但从存储结构看：
        # feats 列表包含3个特征张量
        # 其中1个作为 image_embed 存储
        # 另外2个作为 high_res_feats 存储
        self._features = {"image_embed": feats[-1],  # 主特征 (低分辨率，高语义)
                          "high_res_feats": feats[:-1]} # 高分辨率特征 (2个，细节更丰富)
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """This function is very similar to predict(...), however it is used for batched mode, when the model is expected to generate predictions on multiple images.
        It returns a tuple of lists of masks, ious, and low_res_masks_logits.
        """
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        for img_idx in range(num_images):
            # Transform input prompts
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)

        return all_masks, all_ious, all_low_res_masks

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
          normalize_coords (bool): If true, the point coordinates will be normalized to the range [0,1] and point_coords is expected to be wrt. image dimensions.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Transform input prompts

        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )

        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        # 在 _predict 方法中使用所有三个特征
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"] # 使用两个高分辨率特征
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),# 使用主图像嵌入
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features, # 传递高分辨率特征给解码器
        )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    # 在 get_image_embedding 函数中，
    # 确实只返回了一个主要图像嵌入特征，
    # 而没有返回另外两个高分辨率特征向量
    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"] # 返回的是一个字典dic，返回的是字典中 "image_embed" 对应的单一 tensor（形状为 1xCxHxW）


    def get_multiscale_embeddings原(self) -> dict:
        """
        返回 MedSAM2 提取的多尺度特征字典（命名按实际空间分辨率）。
        对应层：
          feat128 → 128×128 → VMUNet 层1（192通道）
          feat64  →  64×64 → VMUNet 层2（384通道）
          feat32  →  32×32 → VMUNet 层3（768通道）
          同时通过上采样得到 feat256（256×256）用于 VMUNet 层0（96通道）
        """
        image_embed = self._features["image_embed"]  # [B, 256, 32, 32]
        high_res_feats = self._features["high_res_feats"]  # [[B,32,128,128], [B,64,64,64]]

        # --- 对应分辨率命名 ---
        feat128 = high_res_feats[0]  # [B, 32, 128, 128]
        feat64 = high_res_feats[1]  # [B, 64, 64, 64]
        feat32 = image_embed  # [B, 256, 32, 32]

        # print("===== [Debug] MedSAM2 feature shapes =====")
        # print("feat128:", feat128.shape)
        # print("feat64:", feat64.shape)
        # print("feat32:", feat32.shape)
        # print("==========================================")

        # --- 下采样到 VMUNet 对应分辨率 ---
        feat64_8 = F.adaptive_avg_pool2d(feat64, (8, 8))  # 对应 Decoder1
        feat64_16 = F.adaptive_avg_pool2d(feat64, (16, 16))  # 对应 Decoder2
        feat128_32 = F.adaptive_avg_pool2d(feat128, (32, 32))  # 对应 Decoder3
        feat256_64 = F.adaptive_avg_pool2d(feat128, (64, 64))  # 对应 Decoder4（浅层扩大）

        # --- 定义通道投影 ---
        if not hasattr(self, "_proj64_768"):
            self._proj64_768 = nn.Conv2d(64, 768, 1).to(self.device)
            self._proj64_384 = nn.Conv2d(64, 384, 1).to(self.device)
            self._proj128_192 = nn.Conv2d(32, 192, 1).to(self.device)
            self._proj256_96 = nn.Conv2d(32, 96, 1).to(self.device)

        # --- 投影到 VMUNet 对应通道 ---
        feat64_768 = self._proj64_768(feat64_8)  # [B,768,8,8]
        feat64_384 = self._proj64_384(feat64_16)  # [B,384,16,16]
        feat128_192 = self._proj128_192(feat128_32)  # [B,192,32,32]
        feat256_96 = self._proj256_96(feat256_64)  # [B,96,64,64]

        # print("===== [Debug] Projected feature shapes =====")
        # print("feat256_96:", feat256_96.shape)
        # print("feat128_192:", feat128_192.shape)
        # print("feat64_384:", feat64_384.shape)
        # print("feat64_768:", feat64_768.shape)
        # print("============================================")

        # == == = [Debug]== == =
        # feat256_96: torch.Size([1, 96, 64, 64])
        # feat128_192: torch.Size([1, 192, 32, 32])
        # feat64_384: torch.Size([1, 384, 16, 16])
        # feat64_768: torch.Size([1, 768, 8, 8])
        # == == == == == == == == == == == == == == == == == == == == == ==
        # --- 返回多尺度特征字典 ---
        return {
            "feat256_96": feat256_96.permute(0, 2, 3, 1),  # VMUNet 层0（输入）
            "feat128_192": feat128_192.permute(0, 2, 3, 1),  # VMUNet 层1
            "feat64_384": feat64_384.permute(0, 2, 3, 1),  # VMUNet 层2
            "feat64_768": feat64_768.permute(0, 2, 3, 1)  # VMUNet 层3
        }

    # -------- Semantic Views for Decoder --------
    def proj_and_resize(self, x, level: str, out_size: int):
        """
        level: one of ["D0", "D1", "D2", "D3"]
        """
        x = self._decoder_proj[level](x)
        if out_size != x.shape[-1]:
            x = F.interpolate(
                x,
                size=(out_size, out_size),
                mode="bilinear",
                align_corners=False
            )
        return x

    def get_multiscale_embeddings(self) -> dict:
        """
        返回 MedSAM2 提取的多尺度特征字典（命名按实际空间分辨率）。
        对应层：
          feat128 → 128×128 → VMUNet 层1（192通道）
          feat64  →  64×64 → VMUNet 层2（384通道）
          feat32  →  32×32 → VMUNet 层3（768通道）
          同时通过上采样得到 feat256（256×256）用于 VMUNet 层0（96通道）
        """
        image_embed = self._features["image_embed"]  # [B, 256, 32, 32]
        high_res_feats = self._features["high_res_feats"]  # [[B,32,128,128], [B,64,64,64]]

        # --- 对应分辨率命名 ---
        feat128 = high_res_feats[0]  # [1, 32, 128, 128]
        feat64 = high_res_feats[1]  # [1, 64, 64, 64]
        feat32 = image_embed  # [1, 256, 32, 32]

        # print("===== [Debug] MedSAM2 feature shapes =====")
        # print("feat128:", feat128.shape)
        # print("feat64:", feat64.shape)
        # print("feat32:", feat32.shape)
        # print("==========================================")

        # -------- Step 1: Spatial alignment --------
        feat128_32 = F.interpolate(feat128, size=(32, 32), mode="bilinear", align_corners=False)
        feat64_32 = F.interpolate(feat64, size=(32, 32), mode="bilinear", align_corners=False)
        feat32_32 = feat32  # already 32×32

        f128 = self._proj_align["128"](feat128_32)
        f64 = self._proj_align["64"](feat64_32)
        f32 = self._proj_align["32"](feat32_32)

        fusion_input = torch.cat([f128, f64, f32], dim=1)
        gate = self._semantic_gate(fusion_input)

        semantic_base = (
                gate * f32 +
                (1 - gate) * (0.5 * f64 + 0.5 * f128)
        )
        # print("semantic_base shape:", semantic_base.shape)
        feat_d3 = self.proj_and_resize(semantic_base, "D3", 64)
        feat_d2 = self.proj_and_resize(semantic_base, "D2", 32)
        feat_d1 = self.proj_and_resize(semantic_base, "D1", 16)
        feat_d0 = self.proj_and_resize(semantic_base, "D0", 8)
        # print("===== [Debug] Projected feature shapes =====")
        # print("feat256_96:", feat_d3.shape)
        # print("feat128_192:", feat_d2.shape)
        # print("feat64_384:", feat_d1.shape)
        # print("feat64_768:", feat_d0.shape)
        # print("============================================")

        # --- 返回多尺度特征字典 ---！！！！！！名字与.pt的要对应！！！！！！！------------------------------------------------
        # return {
        #     "feat256_96": feat256_96.permute(0, 2, 3, 1),  # VMUNet 层0（输入）
        #     "feat128_192": feat128_192.permute(0, 2, 3, 1),  # VMUNet 层1
        #     "feat64_384": feat64_384.permute(0, 2, 3, 1),  # VMUNet 层2
        #     "feat64_768": feat64_768.permute(0, 2, 3, 1)  # VMUNet 层3
        # }
        return {
            "feat256_96": feat_d3.permute(0, 2, 3, 1),  # [1,64,64,96]
            "feat128_192": feat_d2.permute(0, 2, 3, 1),  # [1,32,32,192]
            "feat64_384": feat_d1.permute(0, 2, 3, 1),  # [1,16,16,384]
            "feat64_768": feat_d0.permute(0, 2, 3, 1),  # [1,8,8,768]
        }

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False






# class SAM2ImagePredictor:
#     def __init__(
#         self,
#         sam_model: SAM2Base,
#         mask_threshold=0.0,
#         max_hole_area=0.0,
#         max_sprinkle_area=0.0,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.model = sam_model
#         self._transforms = SAM2Transforms(
#             resolution=self.model.image_size,
#             mask_threshold=mask_threshold,
#             max_hole_area=max_hole_area,
#             max_sprinkle_area=max_sprinkle_area,
#         )
#         # 新增代码-----------------------------------------------------------------------------------------------------------
#         self.fusion_mode = kwargs.get("fusion_mode", "concat")
#         in_channels = [256, 32, 64]   # 你打印出来的真实通道数！
#         proj_dim = 256  # 投影到同一维度，方便后续融合
#         self._proj_layers = nn.ModuleList([
#             nn.Conv2d(in_ch, proj_dim, kernel_size=1).to(self.device)
#             for in_ch in in_channels
#         ])
#         self._is_image_set = False
#         self._features = None
#         self._orig_hw = None
#         # Whether the predictor is set for single image or a batch of images
#         self._is_batch = False
#         self.mask_threshold = mask_threshold
#         hires_size = self.model.image_size // 4
#         self._bb_feat_sizes = [[hires_size // (2**k)]*2 for k in range(3)]
#     @classmethod
#     def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
#         from .build_sam import build_sam2_hf
#         sam_model = build_sam2_hf(model_id, **kwargs)
#         return cls(sam_model, **kwargs)
#     @torch.no_grad()
#     def set_image(
#         self,
#         image: Union[np.ndarray, Image],
#     ) -> None:
#         self.reset_predictor()
#         # Transform the image to the form expected by the model
#         if isinstance(image, np.ndarray):
#             logging.info("For numpy array image, we assume (HxWxC) format")
#             self._orig_hw = [image.shape[:2]]
#         elif isinstance(image, Image):
#             w, h = image.size
#             self._orig_hw = [(h, w)]
#         else:
#             raise NotImplementedError("Image format not supported")
#         input_image = self._transforms(image)
#         input_image = input_image[None, ...].to(self.device)
#         assert (
#             len(input_image.shape) == 4 and input_image.shape[1] == 3
#         ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
#         logging.info("Computing image embeddings for the provided image...")
#         backbone_out = self.model.forward_image(input_image)
#         _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
#         # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
#         if self.model.directly_add_no_mem_embed:
#             vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
#         feats = [
#             feat.permute(1, 2, 0).view(1, -1, *feat_size)
#             for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
#         ][::-1]
#         self._features = \
#             {"image_embed": feats[-1], # 主图像嵌入 tensor
#              "high_res_feats": feats[:-1]} # 高分辨率特征列表
#         self._is_image_set = True
#         logging.info("Image embeddings computed.")
#     @torch.no_grad()
#     def set_image_batch(
#         self,
#         image_list: List[Union[np.ndarray]],
#     ) -> None:
#         self.reset_predictor()
#         assert isinstance(image_list, list)
#         self._orig_hw = []
#         for image in image_list:
#             assert isinstance(
#                 image, np.ndarray
#             ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
#             self._orig_hw.append(image.shape[:2])
#         # Transform the image to the form expected by the model
#         img_batch = self._transforms.forward_batch(image_list)
#         img_batch = img_batch.to(self.device)
#         batch_size = img_batch.shape[0]
#         assert (
#             len(img_batch.shape) == 4 and img_batch.shape[1] == 3
#         ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
#         logging.info("Computing image embeddings for the provided images...")
#         backbone_out = self.model.forward_image(img_batch)
#         _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
#         # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
#         if self.model.directly_add_no_mem_embed:
#             vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
#         feats = [
#             feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
#             for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
#         ][::-1]
#         self._features = {"image_embed": feats[-1],  # 主特征 (低分辨率，高语义)
#                           "high_res_feats": feats[:-1]} # 高分辨率特征 (2个，细节更丰富)
#         self._is_image_set = True
#         self._is_batch = True
#         logging.info("Image embeddings computed.")
#     def predict_batch(
#         self,
#         point_coords_batch: List[np.ndarray] = None,
#         point_labels_batch: List[np.ndarray] = None,
#         box_batch: List[np.ndarray] = None,
#         mask_input_batch: List[np.ndarray] = None,
#         multimask_output: bool = True,
#         return_logits: bool = False,
#         normalize_coords=True,
#     ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
#         assert self._is_batch, "This function should only be used when in batched mode"
#         if not self._is_image_set:
#             raise RuntimeError(
#                 "An image must be set with .set_image_batch(...) before mask prediction."
#             )
#         num_images = len(self._features["image_embed"])
#         all_masks = []
#         all_ious = []
#         all_low_res_masks = []
#         for img_idx in range(num_images):
#             # Transform input prompts
#             point_coords = (
#                 point_coords_batch[img_idx] if point_coords_batch is not None else None
#             )
#             point_labels = (
#                 point_labels_batch[img_idx] if point_labels_batch is not None else None
#             )
#             box = box_batch[img_idx] if box_batch is not None else None
#             mask_input = (
#                 mask_input_batch[img_idx] if mask_input_batch is not None else None
#             )
#             mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
#                 point_coords,
#                 point_labels,
#                 box,
#                 mask_input,
#                 normalize_coords,
#                 img_idx=img_idx,
#             )
#             masks, iou_predictions, low_res_masks = self._predict(
#                 unnorm_coords,
#                 labels,
#                 unnorm_box,
#                 mask_input,
#                 multimask_output,
#                 return_logits=return_logits,
#                 img_idx=img_idx,
#             )
#             masks_np = masks.squeeze(0).float().detach().cpu().numpy()
#             iou_predictions_np = (
#                 iou_predictions.squeeze(0).float().detach().cpu().numpy()
#             )
#             low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
#             all_masks.append(masks_np)
#             all_ious.append(iou_predictions_np)
#             all_low_res_masks.append(low_res_masks_np)
#         return all_masks, all_ious, all_low_res_masks
#     def predict(
#         self,
#         point_coords: Optional[np.ndarray] = None,
#         point_labels: Optional[np.ndarray] = None,
#         box: Optional[np.ndarray] = None,
#         mask_input: Optional[np.ndarray] = None,
#         multimask_output: bool = True,
#         return_logits: bool = False,
#         normalize_coords=True,
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         if not self._is_image_set:
#             raise RuntimeError(
#                 "An image must be set with .set_image(...) before mask prediction."
#             )
#         mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
#             point_coords, point_labels, box, mask_input, normalize_coords
#         )
#         masks, iou_predictions, low_res_masks = self._predict(
#             unnorm_coords,
#             labels,
#             unnorm_box,
#             mask_input,
#             multimask_output,
#             return_logits=return_logits,
#         )
#         masks_np = masks.squeeze(0).float().detach().cpu().numpy()
#         iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
#         low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
#         return masks_np, iou_predictions_np, low_res_masks_np
#     def _prep_prompts(
#         self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
#     ):
#         unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
#         if point_coords is not None:
#             assert (
#                 point_labels is not None
#             ), "point_labels must be supplied if point_coords is supplied."
#             point_coords = torch.as_tensor(
#                 point_coords, dtype=torch.float, device=self.device
#             )
#             unnorm_coords = self._transforms.transform_coords(
#                 point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
#             )
#             labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
#             if len(unnorm_coords.shape) == 2:
#                 unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
#         if box is not None:
#             box = torch.as_tensor(box, dtype=torch.float, device=self.device)
#             unnorm_box = self._transforms.transform_boxes(
#                 box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
#             )  # Bx2x2
#         if mask_logits is not None:
#             mask_input = torch.as_tensor(
#                 mask_logits, dtype=torch.float, device=self.device
#             )
#             if len(mask_input.shape) == 3:
#                 mask_input = mask_input[None, :, :, :]
#         return mask_input, unnorm_coords, labels, unnorm_box
#     @torch.no_grad()
#     def _predict(
#         self,
#         point_coords: Optional[torch.Tensor],
#         point_labels: Optional[torch.Tensor],
#         boxes: Optional[torch.Tensor] = None,
#         mask_input: Optional[torch.Tensor] = None,
#         multimask_output: bool = True,
#         return_logits: bool = False,
#         img_idx: int = -1,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         if not self._is_image_set:
#             raise RuntimeError(
#                 "An image must be set with .set_image(...) before mask prediction."
#             )
#         if point_coords is not None:
#             concat_points = (point_coords, point_labels)
#         else:
#             concat_points = None
#         # Embed prompts
#         if boxes is not None:
#             box_coords = boxes.reshape(-1, 2, 2)
#             box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
#             box_labels = box_labels.repeat(boxes.size(0), 1)
#             # we merge "boxes" and "points" into a single "concat_points" input (where
#             # boxes are added at the beginning) to sam_prompt_encoder
#             if concat_points is not None:
#                 concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
#                 concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
#                 concat_points = (concat_coords, concat_labels)
#             else:
#                 concat_points = (box_coords, box_labels)
#         sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
#             points=concat_points,
#             boxes=None,
#             masks=mask_input,
#         )
#         # Predict masks
#         batched_mode = (
#             concat_points is not None and concat_points[0].shape[0] > 1
#         )  # multi object prediction
#         # 在 _predict 方法中使用所有三个特征
#         high_res_features = [
#             feat_level[img_idx].unsqueeze(0)
#             for feat_level in self._features["high_res_feats"] # 使用两个高分辨率特征
#         ]
#         low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
#             image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),# 使用主图像嵌入
#             image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=multimask_output,
#             repeat_image=batched_mode,
#             high_res_features=high_res_features, # 传递高分辨率特征给解码器
#         )
#         # Upscale the masks to the original image resolution
#         masks = self._transforms.postprocess_masks(
#             low_res_masks, self._orig_hw[img_idx]
#         )
#         low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
#         if not return_logits:
#             masks = masks > self.mask_threshold
#         return masks, iou_predictions, low_res_masks
#     def get_image_embedding(self) -> torch.Tensor:
#         """
#         Returns the image embeddings for the currently set image, with
#         shape 1xCxHxW, where C is the embedding dimension and (H,W) are
#         the embedding spatial dimension of SAM (typically C=256, H=W=64).
#         """
#         if not self._is_image_set:
#             raise RuntimeError(
#                 "An image must be set with .set_image(...) to generate an embedding."
#             )
#         assert (
#             self._features is not None
#         ), "Features must exist if an image has been set."
#         return self._features["image_embed"] # 返回的是一个字典dic，返回的是字典中 "image_embed" 对应的单一 tensor（形状为 1xCxHxW）
#     def get_fused_embedding(self) -> torch.Tensor:
#         image_embed = self._features["image_embed"]  # [B, C, H, W]
#         high_res_feats = self._features["high_res_feats"]  # list of 2 tensors
#         all_feats = [image_embed] + high_res_feats
#         B = image_embed.shape[0]
#         target_h, target_w = 8, 8
#         # --- 融合模式分支 ---
#         if self.fusion_mode == "concat":
#             fused_feats = []
#             for f, proj in zip(all_feats, self._proj_layers):
#                 f_proj = proj(f)
#                 f_down = F.adaptive_avg_pool2d(f_proj, (target_h, target_w))
#                 fused_feats.append(f_down)
#             fused = torch.cat(fused_feats, dim=1)
#         elif self.fusion_mode == "attention":
#             downs = [F.adaptive_avg_pool2d(proj(f), (target_h, target_w))
#                      for f, proj in zip(all_feats, self._proj_layers)]
#             seqs = [feat.flatten(2).transpose(1, 2) for feat in downs]  # [B, HW, C]
#             tokens = torch.cat(seqs, dim=1)  # [B, HW*3, C]
#             if not hasattr(self, "_attn"):
#                 self._attn = nn.MultiheadAttention(
#                     embed_dim=downs[0].shape[1],
#                     num_heads=4,
#                     batch_first=True
#                 ).to(self.device)
#             fused_tokens, _ = self._attn(tokens, tokens, tokens)
#             fused = fused_tokens.transpose(1, 2).reshape(B, -1, target_h, target_w)
#         elif self.fusion_mode == "residual":
#             feats_proj = [proj(f) for f, proj in zip(all_feats, self._proj_layers)]
#             f = feats_proj[-1]  # 从最高分辨率开始
#             for i in range(len(feats_proj) - 1, 0, -1):
#                 f = F.adaptive_avg_pool2d(f, feats_proj[i - 1].shape[-2:])
#                 f = f + feats_proj[i - 1]
#             fused = F.adaptive_avg_pool2d(f, (target_h, target_w))
#             # --- 新增：逐元素相加 ---
#         elif self.fusion_mode == "sum":
#             downs = [F.adaptive_avg_pool2d(proj(f), (target_h, target_w))
#                      for f, proj in zip(all_feats, self._proj_layers)]
#             fused = torch.stack(downs, dim=0).sum(dim=0)
#         # --- 新增：逐元素最大值 ---
#         elif self.fusion_mode == "maxpool":
#             downs = [F.adaptive_avg_pool2d(proj(f), (target_h, target_w))
#                      for f, proj in zip(all_feats, self._proj_layers)]
#             fused = torch.stack(downs, dim=0).max(dim=0).values
#         # --- 新增：可学习加权融合 ---
#         elif self.fusion_mode == "weighted":
#             downs = [F.adaptive_avg_pool2d(proj(f), (target_h, target_w))
#                      for f, proj in zip(all_feats, self._proj_layers)]
#             if not hasattr(self, "_fusion_weights"):
#                 self._fusion_weights = nn.Parameter(torch.ones(len(downs), device=self.device))
#             weights = F.softmax(self._fusion_weights, dim=0)  # 保证和为1
#             fused = sum(w * d for w, d in zip(weights, downs))
#         else:
#             raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
#         # --- 输出统一到 [B, 768, 8, 8] ---
#         if not hasattr(self, "_out_proj"):
#             self._out_proj = nn.Conv2d(fused.shape[1], 768, kernel_size=1).to(self.device
#         fused = self._out_proj(fused)
#         return fused
#     @property
#     def device(self) -> torch.device:
#         return self.model.device
#     def reset_predictor(self) -> None:
#         """
#         Resets the image embeddings and other state variables.
#         """
#         self._is_image_set = False
#         self._features = None
#         self._orig_hw = None
#         self._is_batch = False

