# シンプルな画像検索

## シンプルな前処理

```bash
python search.py --pdf-dir path/to/pdfs --img-dir path/to/images
```

## シンプルな訓練

```bash
python train.py --dataset_dir path/to/images --binarize --accelerator gpu --gpus 1
```

## シンプルな検索

```bash
python search --gallery_dir path/to/images --query query.jpg
```
