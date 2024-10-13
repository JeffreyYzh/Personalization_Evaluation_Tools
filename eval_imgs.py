import os
import sys
import torch
import argparse
from parse_args import parser_eval
from base_class_ours import IDCLIPScoreCalculator
from clip_eval import IdCLIPEvaluator
os.chdir('/root/CelebBasis/evaluation')
sys.path.append('/root/CelebBasis/evaluation')


# 给src image, tgt image, prompt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root",
        type=str,
        default="src")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="result.csv")
    parser.add_argument(
        "--eval_folder",
        type=str,
        default="output")
    parser.add_argument(
        "--model_dir",
        default="weights",
        type=str)

    parser = parser_eval(parser)
    opt = parser.parse_args()
    src_root = opt.src_root
    id_clip_evaluator = IdCLIPEvaluator(
        torch.device('cuda:0'),
        # torch.device('cpu'),
        clip_model='ViT-B/32',
        model_dir=opt.model_dir
    )
    id_score_calculator = IDCLIPScoreCalculator(
        opt.eval_folder,
        id_clip_evaluator,
        opt.save_dir,
        prompt_dir=os.path.join(src_root, 'prompts.txt'),
        src_img_dir=os.path.join(src_root, '200'),
        src_img_id=os.path.join(src_root, 'image_id.txt'),
    )

    id_score_calculator.start_calc()