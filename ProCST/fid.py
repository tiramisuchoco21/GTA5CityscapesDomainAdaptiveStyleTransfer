import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import linalg
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.inception import inception_v3


# -------------------------------------------------------------
# 1) Load InceptionV3 (feature_dim = 2048)
# -------------------------------------------------------------
class InceptionV3Feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # remove final FC
        self.inception.eval()

    @torch.no_grad()
    def forward(self, x):
        # expects input 299x299
        return self.inception(x)


# -------------------------------------------------------------
# 2) Image → 2048-d feature
# -------------------------------------------------------------
def get_features(image_paths, device):
    model = InceptionV3Feature().to(device)
    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225])
    ])

    feats = []

    for img_path in tqdm(image_paths, desc="Extracting features"):
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(img).cpu().numpy().reshape(-1)
        feats.append(feat)

    feats = np.array(feats)
    return feats


# -------------------------------------------------------------
# 3) FID 계산
# -------------------------------------------------------------
def calculate_fid(feats1, feats2):
    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)

    # mu distance
    diff = mu1 - mu2

    # sqrt of product
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# -------------------------------------------------------------
# 4) 이미지 경로 수집
# -------------------------------------------------------------
def get_image_paths(root):
    exts = [".png", ".jpg", ".jpeg"]
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(dirpath, f))
    return paths


# -------------------------------------------------------------
# 5) Main
# -------------------------------------------------------------
if __name__ == "__main__":

    # ----- 수정해야 할 경로 -----
    cityscapes_root = "/home/work/inAI_plusStyle/ProCST/transfer/images" # 기준이 되는 실제 스타일 이미지
    sit_root        = "/home/work/inAI_plusStyle/ProCST/output_images"  # 방금 생성된 결과물 이미지
    # ------------------------------

    city_paths = get_image_paths(cityscapes_root)
    sit_paths  = get_image_paths(sit_root)[:3000]

    print("Cityscapes images:", len(city_paths))
    print("SIT images:", len(sit_paths))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # features
    feats_city = get_features(city_paths, device)
    feats_sit  = get_features(sit_paths, device)

    fid_value = calculate_fid(feats_city, feats_sit)

    print("\n==============================")
    print("          FID Score")
    print("==============================")
    print("FID =", fid_value)
