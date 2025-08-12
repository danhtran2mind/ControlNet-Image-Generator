# ControlNet Image Generator 🖌️

```bash
git clone https://github.com/danhtran2mind/ControlNet-Image-Generator.git
cd ControlNet-Image-Generator
```
```bash
pip install -r requirements/requirements.txt
```
#### Download Model Checkpoints
```bash
python scripts/download_ckpts.py
```
#### Download Datasets

```bash
python scripts/download_datasets.py
```

#### Setup Third Party (Diffusers for ControlNet Training)
```bash
python scripts/setup_third_party.py
```
### Training
```bash
accelerate launch src/controlnet_image_generator/train.py
```
```bash
accelerate launch src/controlnet_image_generator/train.py \
    --dataset_name "HighCWu-open_pose_controlnet_subset"
```

### Inference

```bash
python src/controlnet_image_generator/infer.py
```