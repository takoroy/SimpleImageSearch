import argparse
import os
import time

import cv2
import faiss
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset import PdfPageDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load("feature.pt", map_location=device)


def load_gallery(dataset: PdfPageDataset) -> torch.Tensor:
    if not os.path.exists("gallery.pt"):
        features = []
        with torch.no_grad():
            for x, _ in tqdm.tqdm(DataLoader(dataset, batch_size=1, shuffle=False)):
                x = x.to(device)
                feature = model(x).detach() > 0
                features.append(feature.cpu())

        gallery = torch.cat(features)
        torch.save(gallery, "gallery.pt")
    else:
        gallery = torch.load("gallery.pt")
    return gallery


def build_index(gallery: np.ndarray) -> faiss.IndexBinaryFlat:
    index = faiss.IndexBinaryFlat(256)
    db = np.packbits(gallery.numpy(), axis=1)
    index.add(db)
    return index


def load_index(dataset: PdfPageDataset) -> faiss.IndexBinaryFlat:
    gallery = load_gallery(dataset)
    if not os.path.exists("vector.index"):
        index = build_index(gallery)
        faiss.write_index_binary(index, "vector.index")
    else:
        index = faiss.read_index_binary("vector.index")
    return index


def load_image(path: str) -> Image:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (480, int(480 * image.shape[0] / image.shape[1])))
    image = cv2.fastNlMeansDenoising(image, h=10)
    return Image.fromarray(image).convert("RGB")


def extract_feature(image: Image, transform: Compose) -> np.ndarray:
    x = transform(image)
    with torch.no_grad():
        feat = model(x.unsqueeze(0).to(device)).cpu() > 0
    return np.packbits(feat.numpy(), axis=1)


def main(gallery_dir: str, query_path: str):
    dataset = PdfPageDataset(root_dir=gallery_dir, split="predict", augmentation=False)
    index = load_index(dataset)

    image = load_image(query_path)
    feat_start = time.perf_counter()
    feat = extract_feature(image, dataset.transform)
    feat_end = time.perf_counter()
    print(f"feature extraction: {feat_end - feat_start}")

    search_start = time.perf_counter()
    dist, result = index.search(feat, k=5)
    search_end = time.perf_counter()
    print(f"search: {search_end - search_start}")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery_dir", type=str, help="gallery directory.")
    parser.add_argument("--query", type=str, help="query image path.")
    parser.add_argument("--use-gpu", action="store_true", help="use gpu or not.")

    args = parser.parse_args()

    device = device if args.use_gpu else "cpu"
    model = model.to(device)
    print(args.query)
    main(args.gallery_dir, args.query)
