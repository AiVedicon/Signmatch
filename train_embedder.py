"""Fine-tune the signature/handwriting embedder with triplet loss.

Data layout (one folder per writer):
    data/<writer_id>/genuine/*.png
    data/<writer_id>/forgery/*.png      <- optional but VERY valuable (skilled forgeries = hard negatives)

Public datasets to start from: CEDAR, GPDS Synthetic/960, BHSig260 (signatures);
IAM, CVL (handwriting / writer identification). Add your own labelled samples on top.

    python training/train_embedder.py --data data --out weights/sig_embedder.pt --epochs 40
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from signatures.embedder import SignatureNet
from signatures.preprocess import prepare, read_image

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_dir(d: Path):
    out = []
    if d.is_dir():
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in EXT:
                try:
                    out.append(prepare(read_image(f)).canvas)
                except ValueError:
                    print("skip", f)
    return out


def augment(c: np.ndarray) -> np.ndarray:
    h, w = c.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-4, 4), random.uniform(0.92, 1.08))
    M[0, 1] += random.uniform(-0.08, 0.08)                      # small shear
    M[:, 2] += (random.uniform(-0.04, 0.04) * w, random.uniform(-0.04, 0.04) * h)
    out = cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR)
    if random.random() < 0.3:                                    # pen-thickness jitter
        k = np.ones((2, 2), np.uint8)
        out = cv2.dilate(out, k) if random.random() < 0.5 else cv2.erode(out, k)
    return out


class TripletData(Dataset):
    def __init__(self, writers, length, aug=True):
        self.items = {}
        for p in writers:
            gen, forg = load_dir(p / "genuine"), load_dir(p / "forgery")
            if len(gen) >= 2:
                self.items[p.name] = {"gen": gen, "forg": forg}
        self.names, self.length, self.aug = list(self.items), length, aug
        if len(self.names) < 2:
            raise SystemExit("Need at least 2 writers with >= 2 genuine samples each")

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        name = random.choice(self.names)
        d = self.items[name]
        a, p = random.sample(d["gen"], 2)
        if d["forg"] and random.random() < 0.5:
            n = random.choice(d["forg"])                          # skilled forgery of same writer
        else:
            other = random.choice(self.names)
            while other == name:
                other = random.choice(self.names)
            n = random.choice(self.items[other]["gen"])           # another writer
        f = augment if self.aug else (lambda x: x)
        return tuple(torch.from_numpy(f(x).copy())[None] for x in (a, p, n))


@torch.no_grad()
def validate(net, loader, dev):
    net.eval()
    ok = tot = 0
    for a, p, n in loader:
        za, zp, zn = (net(t.to(dev)) for t in (a, p, n))
        ok += int(((za - zp).norm(dim=1) < (za - zn).norm(dim=1)).sum())
        tot += len(a)
    return ok / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="weights/sig_embedder.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=2000, help="triplets per epoch")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    writers = sorted(p for p in Path(args.data).iterdir() if p.is_dir())
    random.Random(0).shuffle(writers)
    n_val = max(2, len(writers) // 10) if len(writers) >= 12 else 0
    train_w, val_w = writers[n_val:], writers[:n_val]        # validate on UNSEEN writers

    train = DataLoader(TripletData(train_w, args.steps), batch_size=args.batch,
                       num_workers=args.workers, drop_last=True)
    val = DataLoader(TripletData(val_w, 500, aug=False), batch_size=args.batch) if val_w else None

    net = SignatureNet(args.dim, pretrained=True).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.TripletMarginLoss(margin=args.margin)
    use_amp = dev == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = -1.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        net.train()
        run = 0.0
        for a, p, n in train:
            a, p, n = a.to(dev), p.to(dev), n.to(dev)
            with torch.autocast(device_type=dev, enabled=use_amp):
                za, zp, zn = net(a), net(p), net(n)
            loss = loss_fn(za.float(), zp.float(), zn.float())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()
        acc = validate(net, val, dev) if val else float("nan")
        print(f"epoch {ep:3d}  loss {run / len(train):.4f}  val triplet-acc {acc:.3f}")
        score = acc if val else -run
        if score > best:
            best = score
            torch.save({"state_dict": net.state_dict(), "dim": args.dim}, args.out)
    print("saved best checkpoint to", args.out)


if __name__ == "__main__":
    main()
