import os
import shutil
import subprocess
import numpy as np
from PIL import Image
from tqdm import tqdm

from enhance import (
    apply_gamma,
    apply_contrast,
    apply_sharpen,
    apply_clahe,
    apply_histogram_matching
)

############################################################
# 경로 설정
############################################################

PROCST_OUTPUT = "/home/work/inAI_plusStyle/ProCST/output_images"

# Deeplab 폴더 구조
DEEPLAB_TRANSFER = "/home/work/inAI_plusStyle/Deeplabv3/transfer"
DEEPLAB_TRANSFER_IMAGES = os.path.join(DEEPLAB_TRANSFER, "images")  # images 폴더 추가!
DEEPLAB_INFER = "/home/work/inAI_plusStyle/Deeplabv3/gta_infer.py"
DEEPLAB_CKPT = "/home/work/inAI_plusStyle/checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth"

# Deeplab 전용 python.exe
DEEPLAB_PYTHON = "/home/work/inAI_plusStyle/deeplab/bin/python"

# Reference image
REF_IMAGE = "/home/work/inAI_plusStyle/ref_cityscapes.png"


############################################################
# Reference Loading
############################################################
def load_reference(path="/home/work/inAI_plusStyle/ref_cityscapes.png"):
    """참조 이미지 로드"""
    if not os.path.exists(path):
        print(f"[Warning] Reference image not found: {path}")
        return None
    return np.array(Image.open(path).convert("RGB"))


############################################################
# 후처리 함수
############################################################
def postprocess_image(img, cfg, ref):
    """이미지 후처리 적용"""
    if ref is not None:
        img = apply_histogram_matching(img, ref, cfg["hm"])
    
    img = apply_gamma(img, cfg["gamma"])
    img = apply_contrast(img, cfg["contrast"])
    img = apply_sharpen(img, cfg["sharpen"])
    
    if cfg["clahe"]:
        img = apply_clahe(img)
    
    return img


############################################################
# 경로 검증
############################################################
def validate_paths():
    """모든 경로 검증"""
    errors = []
    
    if not os.path.exists(PROCST_OUTPUT):
        errors.append(f"ProCST output not found: {PROCST_OUTPUT}")
    
    if not os.path.exists(DEEPLAB_PYTHON):
        errors.append(f"Python not found: {DEEPLAB_PYTHON}")
    
    if not os.path.exists(DEEPLAB_INFER):
        errors.append(f"gta_infer.py not found: {DEEPLAB_INFER}")
    
    if not os.path.exists(DEEPLAB_CKPT):
        errors.append(f"Checkpoint not found: {DEEPLAB_CKPT}")
    
    if errors:
        print("\n[ERROR] Path validation failed:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    print("[OK] All paths validated")
    return True


############################################################
# DeepLab 실행 + mIoU 파싱
############################################################
def run_miou(log_file):
    """DeepLab mIoU 계산"""
    
    log_file.write("\n[DEBUG] Running Deeplab mIoU...\n")
    log_file.write(f"[DEBUG] PYTHON = {DEEPLAB_PYTHON}\n")
    log_file.write(f"[DEBUG] SCRIPT = {DEEPLAB_INFER}\n")
    log_file.write(f"[DEBUG] DATA_ROOT = {DEEPLAB_TRANSFER}\n")  # transfer/ 로 전달
    log_file.write(f"[DEBUG] IMAGES DIR = {DEEPLAB_TRANSFER_IMAGES}\n")

    cmd = [
        DEEPLAB_PYTHON,
        DEEPLAB_INFER,
        "--data_root", DEEPLAB_TRANSFER,  # transfer/ (images/의 부모)
        "--model", "deeplabv3plus_mobilenet",
        "--ckpt", DEEPLAB_CKPT,
        "--batch_size", "4"
    ]

    log_file.write(f"[DEBUG] CMD = {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        log_file.write("\n--- DeepLab Output ---\n")
        log_file.write(result.stdout + "\n")
        log_file.write("\n--- DeepLab Error Stream ---\n")
        log_file.write(result.stderr + "\n")

        if result.returncode != 0:
            log_file.write(f"[ERROR] Deeplab returned code {result.returncode}\n")
            return None

        # mIoU 파싱 - 다양한 형식 지원
        for line in result.stdout.split("\n"):
            line_lower = line.lower()
            
            # "Mean IoU: 0.3852" 형식
            if "mean iou" in line_lower:
                try:
                    # "Mean IoU: 0.3852104028886671" → 0.3852
                    parts = line.split(":")
                    if len(parts) >= 2:
                        score = float(parts[1].strip())
                        log_file.write(f"[Parsed] Mean IoU: {score:.4f}\n")
                        return score
                except Exception as e:
                    log_file.write(f"[ERROR] Mean IoU parsing failed: {e}\n")
                    continue
            
            # "mIoU: 0.4523" 또는 "miou = 45.23%" 형식
            if "miou" in line_lower:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "miou" in part.lower():
                            if i + 1 < len(parts):
                                score_str = parts[i + 1].replace(',', '').replace('%', '').replace(':', '')
                                score = float(score_str)
                                result_score = score / 100 if score > 1 else score
                                log_file.write(f"[Parsed] mIoU: {result_score:.4f}\n")
                                return result_score
                    # 마지막 숫자 시도
                    score = float(parts[-1].replace('%', ''))
                    result_score = score / 100 if score > 1 else score
                    log_file.write(f"[Parsed] Last number: {result_score:.4f}\n")
                    return result_score
                except Exception as e:
                    log_file.write(f"[ERROR] mIoU parsing failed: {e}\n")
                    continue

        log_file.write("[ERROR] No mIoU/Mean IoU found in output\n")
        return None

    except subprocess.TimeoutExpired:
        log_file.write("[ERROR] Timeout (5min)\n")
        return None
    except Exception as e:
        log_file.write(f"[EXCEPTION] {e}\n")
        return None


############################################################
# MAIN
############################################################
def main():
    
    print("="*60)
    print("Auto Postprocess Search")
    print("="*60)
    
    if not validate_paths():
        return
    
    log_file = open("search_log.txt", "w", encoding="utf-8")
    log_file.write("=== Auto Postprocess Search Log ===\n")
    log_file.write(f"ProCST Output: {PROCST_OUTPUT}\n")
    log_file.write(f"DeepLab Script: {DEEPLAB_INFER}\n")
    log_file.write(f"DeepLab Transfer: {DEEPLAB_TRANSFER}\n")
    log_file.write(f"DeepLab Images Dir: {DEEPLAB_TRANSFER_IMAGES}\n\n")

    # 파일 목록
    files = sorted([f for f in os.listdir(PROCST_OUTPUT) if f.endswith('.png')])

    print(f"\n[Info] Found {len(files)} images")
    
    if len(files) == 0:
        print("\n[ERROR] No images in ProCST output")
        log_file.close()
        return

    # Reference
    ref = load_reference(REF_IMAGE)
    if ref is None:
        print("[Warning] No reference - histogram matching disabled")

    # 탐색 공간
    hm_list = [0.2, 0.4, 0.6] if ref is not None else [0.0]
    gamma_list = [0.8, 1.0, 1.2]
    contrast_list = [0.8, 1.0, 1.2]
    sharpen_list = [0.0, 0.5]
    clahe_list = [False, True]

    total = len(hm_list) * len(gamma_list) * len(contrast_list) * len(sharpen_list) * len(clahe_list)
    print(f"[Info] Total experiments: {total}")
    print(f"[Info] Using {min(100, len(files))} images per experiment\n")

    best_score = -1
    best_cfg = None
    exp_id = 0

    for hm in hm_list:
        for g in gamma_list:
            for c in contrast_list:
                for sh in sharpen_list:
                    for cl in clahe_list:

                        cfg = {"hm": hm, "gamma": g, "contrast": c, "sharpen": sh, "clahe": cl}

                        print("\n" + "="*60)
                        print(f"[EXP {exp_id}/{total}] {cfg}")
                        print("="*60)

                        log_file.write(f"\n\n[EXP {exp_id}] Config = {cfg}\n")

                        # transfer/images 폴더만 정리 (labels는 유지)
                        if os.path.exists(DEEPLAB_TRANSFER_IMAGES):
                            shutil.rmtree(DEEPLAB_TRANSFER_IMAGES)
                        os.makedirs(DEEPLAB_TRANSFER_IMAGES, exist_ok=True)
                        
                        # labels 폴더는 있으면 유지, 없으면 생성
                        labels_dir = os.path.join(DEEPLAB_TRANSFER, "labels")
                        if not os.path.exists(labels_dir):
                            os.makedirs(labels_dir, exist_ok=True)

                        # 후처리 적용 - images/ 폴더에 저장!
                        num_images = min(100, len(files))
                        for f in tqdm(files[:num_images], desc=f"EXP {exp_id}"):
                            try:
                                raw = np.array(Image.open(os.path.join(PROCST_OUTPUT, f)))
                                out = postprocess_image(raw, cfg, ref)
                                # images/ 폴더에 저장!
                                Image.fromarray(out).save(os.path.join(DEEPLAB_TRANSFER_IMAGES, f))
                                
                                # 더미 라벨 생성 (없을 때만)
                                label_path = os.path.join(labels_dir, f)
                                if not os.path.exists(label_path):
                                    dummy_label = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
                                    Image.fromarray(dummy_label).save(label_path)
                            except Exception as e:
                                print(f"[Error] {f}: {e}")
                                log_file.write(f"[Error] {f}: {e}\n")
                                continue

                        # 저장된 파일 수 확인
                        saved_files = len(os.listdir(DEEPLAB_TRANSFER_IMAGES))
                        print(f"[Info] Saved {saved_files} images to {DEEPLAB_TRANSFER_IMAGES}")
                        log_file.write(f"[Info] Saved {saved_files} images\n")

                        # mIoU 계산
                        print(f"[Info] Running DeepLab...")
                        score = run_miou(log_file)

                        if score is None:
                            print(f"[EXP {exp_id}] ERROR - Skip")
                            log_file.write("[ERROR] Skipped\n")
                            exp_id += 1
                            continue

                        print(f"[Result] mIoU = {score:.4f}")
                        log_file.write(f"[mIoU] = {score:.4f}\n")

                        if score > best_score:
                            best_score = score
                            best_cfg = cfg
                            print(f"[New Best!] mIoU = {score:.4f}")

                        exp_id += 1

    # 결과
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Best Config: {best_cfg}")
    print(f"Best mIoU: {best_score:.4f}")
    print("="*60)

    log_file.write("\n" + "="*60 + "\n")
    log_file.write(f"BEST CONFIG = {best_cfg}\n")
    log_file.write(f"BEST mIoU = {best_score:.4f}\n")
    log_file.write("="*60 + "\n")

    log_file.close()
    print(f"\nLog: search_log.txt")


if __name__ == "__main__":
    main()