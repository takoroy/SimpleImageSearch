import argparse
import glob
import hashlib
import os

import tqdm
from pdf2image import convert_from_path


def main(pdf_dir: str, img_dir: str):
    pdfs = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    os.makedirs(img_dir, exist_ok=True)

    for _, fpath in tqdm.tqdm(enumerate(pdfs)):
        fname = os.path.basename(fpath)
        hash = hashlib.sha256(fname.encode())
        hash = hash.hexdigest()[:5]

        if len(list(glob.glob(os.path.join(img_dir, f"{hash}_*.jpg")))) > 0:
            continue
        pages = convert_from_path(fpath, thread_count=8)
        for j, page in enumerate(pages):
            output_path = os.path.join(img_dir, f"{hash}_{str(j).zfill(4)}.jpg")
            if os.path.exists(output_path):
                continue
            else:
                page.save(output_path, "JPEG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdf-dir", type=str, help="input pdfs' directory.")
    parser.add_argument("--img-dir", type=str, help="output images' directory.")

    args = parser.parse_args()

    main(args.pdf_dir, args.img_dir)
