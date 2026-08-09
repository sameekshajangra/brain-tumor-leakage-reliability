import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils_stage2 import (
    BrainTumorDataset, build_efficientnet_b0,
    get_val_test_transform
)

def main():
    print("=" * 70)
    print("STAGE 3 — FINAL TEST PREDICTIONS")
    print("=" * 70)
    
    BASE_DIR = PROJECT_ROOT
    RAW_DIR = BASE_DIR.parent / "data" / "raw"
    SPLITS_DIR = BASE_DIR / "results" / "stage3" / "splits"
    CKPT_DIR = BASE_DIR / "results" / "stage3" / "checkpoints"
    PRED_DIR = BASE_DIR / "results" / "stage3" / "predictions"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Device: {device}")
    
    for fold in range(1, 6):
        for cond in ['A', 'B', 'C']:
            print(f"Generating predictions for Fold {fold} Condition {cond}...")
            
            # Load test manifest (identical for A/B/C but we use the specific one for consistency)
            test_csv = SPLITS_DIR / f"fold{fold}_{cond}_test.csv"
            test_df = pd.read_csv(test_csv)
            
            # Load model
            ckpt_path = CKPT_DIR / f"fold{fold}_{cond}" / f"fold{fold}_condition{cond}_best.pt"
            ckpt = torch.load(ckpt_path, map_location='cpu')
            
            model = build_efficientnet_b0(num_classes=3, pretrained=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
            
            # DataLoader
            test_ds = BrainTumorDataset(test_df, RAW_DIR, transform=get_val_test_transform())
            test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
            
            all_probs = []
            all_preds = []
            
            with torch.no_grad():
                for imgs, labels, _ in test_loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    preds = probs.argmax(axis=1) + 1 # classes 1,2,3
                    all_probs.extend(probs)
                    all_preds.extend(preds)
                    
            # Save predictions
            out_df = test_df.copy()
            out_df['fold'] = fold
            out_df['condition'] = cond
            out_df['true_label'] = out_df['label']
            out_df['predicted_label'] = all_preds
            out_df['probability_meningioma'] = [p[0] for p in all_probs]
            out_df['probability_glioma'] = [p[1] for p in all_probs]
            out_df['probability_pituitary'] = [p[2] for p in all_probs]
            out_df['max_confidence'] = [max(p) for p in all_probs]
            out_df['correct'] = (out_df['true_label'] == out_df['predicted_label']).astype(int)
            
            out_path = PRED_DIR / f"fold{fold}_condition{cond}_predictions.csv"
            out_df.to_csv(out_path, index=False)
            
    print("All 15 test prediction files generated successfully.")

if __name__ == "__main__":
    main()
