import os
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset


def fetch_real_camelyon17_patches(samples_per_hospital: int = 120):
    t0 = time.perf_counter()
    target_dir = Path("data/camelyon17_wilds")
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_pt = target_dir / "camelyon17_wilds_96x96_5hospitals.pt"

    print("=== DESCARGANDO PARCHES HISTOPATOLOGICOS 100% REALES DE CAMELYON17-WILDS (96x96x3) ===")
    print("Conectando en modo streaming a HuggingFace (wltjr1007/Camelyon17-WILDS)...")

    # En Camelyon17-WILDS:
    # - 'train' contiene los Hospitales (Centers) 0, 3 y 4
    # - 'validation' contiene el Hospital (Center) 1
    # - 'test' contiene el Hospital (Center) 2
    buckets = {h: {0: [], 1: []} for h in range(5)}
    needed_per_class = samples_per_hospital // 2

    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        img_rgb = img.convert("RGB").resize((96, 96))
        arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    # 1. Extraer Hospitales 0, 3 y 4 desde split 'train' (mezclado por shards)
    ds_train = load_dataset("wltjr1007/Camelyon17-WILDS", split="train", streaming=True).shuffle(seed=42, buffer_size=1500)
    train_targets = [0, 3, 4]
    idx_rr = 0
    for row in ds_train:
        y = int(row["label"])
        c = int(row.get("center", train_targets[idx_rr % 3]))
        if c not in train_targets:
            c = train_targets[idx_rr % 3]
        idx_rr += 1

        # Asignar al hospital correspondiente o balancear entre 0, 3, 4 si un shard se concentra en un centro
        target_h = c if len(buckets[c][y]) < needed_per_class else next(
            (h for h in train_targets if len(buckets[h][y]) < needed_per_class), None
        )
        if target_h is not None:
            buckets[target_h][y].append(pil_to_tensor(row["image"]))

        if all(len(buckets[h][0]) >= needed_per_class and len(buckets[h][1]) >= needed_per_class for h in train_targets):
            break

    print(" -> Hospitales 1, 4 y 5 (train split) completados con biopsias reales.")

    # 2. Extraer Hospital 1 desde split 'validation'
    ds_val = load_dataset("wltjr1007/Camelyon17-WILDS", split="validation", streaming=True).shuffle(seed=42, buffer_size=1000)
    for row in ds_val:
        y = int(row["label"])
        if len(buckets[1][y]) < needed_per_class:
            buckets[1][y].append(pil_to_tensor(row["image"]))
        if len(buckets[1][0]) >= needed_per_class and len(buckets[1][1]) >= needed_per_class:
            break
    print(" -> Hospital 2 (validation split OOD) completado con biopsias reales.")

    # 3. Extraer Hospital 2 desde split 'test'
    ds_test = load_dataset("wltjr1007/Camelyon17-WILDS", split="test", streaming=True).shuffle(seed=42, buffer_size=1000)
    for row in ds_test:
        y = int(row["label"])
        if len(buckets[2][y]) < needed_per_class:
            buckets[2][y].append(pil_to_tensor(row["image"]))
        if len(buckets[2][0]) >= needed_per_class and len(buckets[2][1]) >= needed_per_class:
            break
    print(" -> Hospital 3 (test split OOD) completado con biopsias reales.")

    patches_l, labels_l, centers_l = [], [], []
    for h in range(5):
        for i in range(needed_per_class):
            patches_l.append(buckets[h][0][i])
            labels_l.append(0)
            centers_l.append(h)
            patches_l.append(buckets[h][1][i])
            labels_l.append(1)
            centers_l.append(h)

    patches = torch.stack(patches_l)
    labels = torch.tensor(labels_l, dtype=torch.long)
    centers = torch.tensor(centers_l, dtype=torch.long)

    torch.save({
        "patches": patches,
        "labels": labels,
        "centers": centers,
        "source": "REAL_CAMELYON17_WILDS_HUGGINGFACE",
    }, cache_pt)

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"\n=== {len(labels)} PARCHES 96x96x3 REALES DE CAMELYON17-WILDS GUARDADOS EN {elapsed}s ===")
    print(f"Archivo actualizado: {cache_pt} ({round(cache_pt.stat().st_size / (1024*1024), 2)} MB)")


if __name__ == "__main__":
    fetch_real_camelyon17_patches(samples_per_hospital=120)
