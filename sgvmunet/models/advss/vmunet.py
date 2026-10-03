# from .vmamba import VSSM_point,VSSM
from .vmamba1 import VSSM_point,VSSM
import torch
from torch import nn


class VMUNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.vmunet(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            try:
                pretrained_dict = modelCheckpoint['model']
            except:
                try:
                    pretrained_dict = modelCheckpoint['model_state_dict']
                except:
                    pretrained_dict = modelCheckpoint
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                         -                              len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            try:
                pretrained_odict = modelCheckpoint['model']
            except:
                try:
                    pretrained_odict = modelCheckpoint['model_state_dict']
                except:
                    pretrained_odict = modelCheckpoint
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")

class VMUNet_point(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM_point(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate
                           )
    def forward(self, x,feature):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.vmunet(x, feature)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits

    def load_from原(self):
        if self.load_ckpt_path is not None:
            print(f"[Info] Loading checkpoint from {self.load_ckpt_path}")

            # ----------- 读取 checkpoint ----------- #
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path, map_location="cpu")

            print(f"[DEBUG] Checkpoint top-level keys = {list(modelCheckpoint.keys())}")

            # ---- 尝试不同的键结构 ----
            try:
                pretrained_dict = modelCheckpoint['model']
                print("[DEBUG] Using checkpoint['model']")
            except:
                try:
                    pretrained_dict = modelCheckpoint['model_state_dict']
                    print("[DEBUG] Using checkpoint['model_state_dict']")
                except:
                    pretrained_dict = modelCheckpoint
                    print("[DEBUG] Using checkpoint itself as pretrained_dict")

            print(f"[DEBUG] Number of pretrained params = {len(pretrained_dict)}")
            print(f"[DEBUG] Sample keys: {list(pretrained_dict.keys())[:10]}")

            # ----------- Encoder 部分匹配 ----------- #
            matched_dict = {}
            skipped = []
            for k, v in pretrained_dict.items():
                # 去掉 "vmunet." 前缀
                k = k.replace("vmunet.", "")
                if k in model_dict and model_dict[k].shape == v.shape:
                    matched_dict[k] = v
                else:
                    skipped.append(k)

            model_dict.update(matched_dict)
            self.vmunet.load_state_dict(model_dict, strict=False)

            print(f"[Encoder] ✅ Loaded {len(matched_dict)} / {len(pretrained_dict)} params, skipped {len(skipped)}")
            if skipped:
                print("[Encoder] ⚠️ Skipped example keys:", skipped[:10], "...")

            # ----------- Decoder 部分映射 ----------- #
            model_dict = self.vmunet.state_dict()
            pretrained_odict = pretrained_dict
            decoder_dict = {}
            for k, v in pretrained_odict.items():
                # 去掉 "vmunet." 前缀
                k = k.replace("vmunet.", "")
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                else:
                    continue

                # 检查 shape
                if new_k in model_dict and model_dict[new_k].shape == v.shape:
                    decoder_dict[new_k] = v
                else:
                    print(f"[DEBUG] Decoder skip because shape mismatch: {k} -> {new_k}")

            model_dict.update(decoder_dict)
            self.vmunet.load_state_dict(model_dict, strict=False)

            print(f"[Decoder] ✅ Loaded {len(decoder_dict)} decoder params.")
            if len(decoder_dict) > 0:
                print(f"[DEBUG] Decoder matched example keys: {list(decoder_dict.keys())[:10]}")

            print("[Decoder] Load finished.")



    def load_from1(self):
        if self.load_ckpt_path is None:
            print("[Info] No checkpoint provided, skip loading.")
            return

        print(f"[Info] Loading clean checkpoint: {self.load_ckpt_path}")

        ckpt = torch.load(self.load_ckpt_path, map_location="cpu")


        if isinstance(ckpt, dict) and "model" in ckpt:
            print("[Info] Unwrapping ckpt['model']")
            pretrained_dict = ckpt["model"]
        else:
            pretrained_dict = ckpt
-
        fixed_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith("vmunet."):
                new_k = k[len("vmunet."):]  # 去前缀
            else:
                new_k = k
            fixed_dict[new_k] = v

        pretrained_dict = {
            k: v for k, v in fixed_dict.items()
            if ("total_ops" not in k and "total_params" not in k)
        }

        model_dict = self.vmunet.state_dict()

        matched_dict, skipped, mismatch_details = {}, [], []

        # ----------- 匹配 encoder -----------
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    matched_dict[k] = v
                else:
                    skipped.append(k)
                    mismatch_details.append(
                        f"[DEBUG] Shape mismatch: {k}, ckpt={v.shape}, model={model_dict[k].shape}"
                    )
            else:
                skipped.append(k)
                mismatch_details.append(
                    f"[DEBUG] Key not found in model: {k}"
                )

        self.vmunet.load_state_dict(matched_dict, strict=False)

        print(f"[Encoder] Loaded {len(matched_dict)} / {len(pretrained_dict)} params")
        print(f"[Encoder] Skipped {len(skipped)} params")

    def load_from(self):
        if self.load_ckpt_path is None:
            print("[Info] No checkpoint provided, skip loading.")
            return

        print(f"[Info] Loading clean checkpoint: {self.load_ckpt_path}")

        # ---------------- load checkpoint ----------------
        ckpt = torch.load(self.load_ckpt_path, map_location="cpu")

        # ---- unwrap {"model": ...} ----
        if isinstance(ckpt, dict) and "model" in ckpt:
            print("[Info] Unwrapping ckpt['model']")
            pretrained_dict = ckpt["model"]
        else:
            pretrained_dict = ckpt

        # ---- remove "vmunet." prefix ----
        fixed_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith("vmunet."):
                new_k = k[len("vmunet."):]
            else:
                new_k = k
            fixed_dict[new_k] = v

        # remove flops parameters
        pretrained_dict = {
            k: v for k, v in fixed_dict.items()
            if ("total_ops" not in k and "total_params" not in k)
        }

        model_dict = self.vmunet.state_dict()

        # ----------------------------------------------------------------------------------------------------
        # Split keys: Encoder keys = layers.0 ~ layers.3 ; Decoder keys = layers_up.0 ~ layers_up.3
        # ----------------------------------------------------------------------------------------------------

        encoder_keys = []
        decoder_keys = []
        other_keys = []

        for k in pretrained_dict.keys():
            if k.startswith("layers."):
                encoder_keys.append(k)
            elif k.startswith("layers_up."):
                decoder_keys.append(k)
            else:
                other_keys.append(k)  # patch_embed, final head, norms, etc.

        # Stats
        print(f"[Info] Keys: encoder={len(encoder_keys)}, decoder={len(decoder_keys)}, other={len(other_keys)}")

        # ---------------- Final matching ----------------
        matched, skipped, mismatch_details = {}, [], []

        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    matched[k] = v
                else:
                    skipped.append(k)
                    mismatch_details.append(
                        f"[Mismatch] {k}: ckpt={v.shape}, model={model_dict[k].shape}"
                    )
            else:
                skipped.append(k)
                mismatch_details.append(
                    f"[Missing] Key not found in model: {k}"
                )

        # ---- load ----
        self.vmunet.load_state_dict(matched, strict=False)

        # ----------------------------------------------------------------------------------------------------
        # Output advanced stats: how many encoder & decoder keys were successfully loaded
        # ----------------------------------------------------------------------------------------------------
        loaded_encoder = sum(1 for k in matched if k in encoder_keys)
        loaded_decoder = sum(1 for k in matched if k in decoder_keys)
        loaded_other = sum(1 for k in matched if k in other_keys)

        print("\n================= Weight Loading Summary =================")
        print(f"[Encoder] Loaded {loaded_encoder}/{len(encoder_keys)} keys")
        print(f"[Decoder] Loaded {loaded_decoder}/{len(decoder_keys)} keys")
        print(f"[Other]   Loaded {loaded_other}/{len(other_keys)} keys")
        print(f"[Total]   Loaded {len(matched)} / {len(pretrained_dict)} keys")
        print("==========================================================")

        if len(mismatch_details) > 0:
            print("\n[DEBUG] Mismatch / Missing key details:")
            for msg in mismatch_details[:20]:
                print("   ", msg)
            if len(mismatch_details) > 20:
                print(f"   ... ({len(mismatch_details) - 20} more)")





    def freeze_encoder_layers(self, freeze_level=2):
        """
        冻结 encoder 前 freeze_level 个阶段的参数
        """
        freeze_layers = [f"layers.{i}" for i in range(freeze_level)]
        for name, param in self.vmunet.named_parameters():
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False
        print(f"✅ 冻结 Encoder 前 {freeze_level} 层参数。")




    def freeze_except_cdfa(self):
        # 先冻结所有参数
        for p in self.parameters():
            p.requires_grad = False
        # 再解冻 CDFA 的参数
        if hasattr(self.vmunet, "CDFA"):
            for p in self.vmunet.CDFA.parameters():
                p.requires_grad = True
            trainable = [n for n, p in self.vmunet.CDFA.named_parameters() if p.requires_grad]
            print(f"✅ 仅训练以下 CDFA 层: {trainable}")
        else:
            print("⚠️ CDFA 不存在，跳过解冻操作")

    def unfreeze_decoder(self):
        """
        第二阶段：解冻 Decoder（layers_up + final_*）
        """
        for name, param in self.named_parameters():
            if ("vmunet.layers_up" in name) or ("vmunet.final_" in name):
                param.requires_grad = True
        print("🔓 已解冻 Decoder 层（layers_up + final_*）")

    def unfreeze_all(self):
        """
        第三阶段：解冻全部层，进入全模型微调
        """
        for param in self.parameters():
            param.requires_grad = True
        print("🔓 已解冻全模型（Encoder + Decoder）")