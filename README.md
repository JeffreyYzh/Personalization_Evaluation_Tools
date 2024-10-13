## Evaluation Tools for Customized Image Generation
### Set up
Our code mainly bases on [CelebBasis](https://github.com/ygtxr1997/CelebBasis). We use their environment to initialize.
```
conda env create -f environment.yaml
conda activate sd
pip install insightface
```

### Download Model
As we use PIPNet to extract face landmarks ([weights](https://huggingface.co/datasets/ygtxr1997/CelebBasis/tree/main/PIPNet)) for face region detection and computes cosine similarities using ArcFace ([weights](https://huggingface.co/datasets/ygtxr1997/CelebBasis/tree/main/glint360k_cosface_r100_fp16_0.1)). Please download their weights to the `/weights` directory.

### Download Source Image
We evaluate on the first 200 images in the CelebA-HQ dataset with 20 text prompts including 15 realistic prompts and 5 stylistic prompts. Refer to /src directory and unzip the [image](https://drive.google.com/file/d/1IUM7GGPoXnLlUPhkOVErnrl0cgFy2Uvq/view?usp=sharing) under the directory.

### Quick Use
1. Generate Images using 200 images and 20 text prompts. Label each image as {IMAGE_ID}_{PROMPT_ID}.jpg and put them all in `/Your/Path/To/OUTPUT`
2. Run the following code to generate evaluation results
```
python eval_imgs.py --src_root src --save_dir result.csv --eval_folder ouput --model_dir weights
```