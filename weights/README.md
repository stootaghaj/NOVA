# Model Weights

Place your fine-tuned NOVA model weights in this directory.

## Required Files

- `NOVA_merged.pt` - Fine-tuned DINOv2 model with merged LoRA weights

## Download

Model weights are available on our [project page](https://stootaghaj.github.io/nova-project/).

## Usage

Once weights are downloaded, run:

```bash
python nova.py --image-a samples/frame1.png --image-b samples/frame2.png \
    --checkpoint weights/NOVA_merged.pt
```

