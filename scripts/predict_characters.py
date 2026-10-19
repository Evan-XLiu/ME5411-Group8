"""Load a trusted local checkpoint and classify a directory of chip crops."""
import argparse,json,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from task_07.cnn import build_model
from common.image_io import load_grayscale
from common.features import extract_features
from run_experiments import prepare

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--crops',type=Path,default=ROOT/'results/figures/characters');a=p.parse_args()
    paths=sorted(a.crops.glob('*.png'))
    if not paths:raise ValueError('No PNG character crops found')
    if a.model.suffix=='.pt':
        saved=torch.load(a.model,map_location='cpu',weights_only=True);model=build_model(len(saved['classes']),architecture=saved['architecture']);model.load_state_dict(saved['state_dict']);model.eval()
        inputs=torch.tensor(np.stack([prepare(255-load_grayscale(f),saved.get('preprocessing','resize')) for f in paths])[:,None]/255)
        with torch.no_grad():indices=model(inputs).argmax(1).tolist()
    else:
        with a.model.open('rb') as f:saved=pickle.load(f)
        indices=saved['classifier'].predict(extract_features([prepare(255-load_grayscale(f),saved.get('preprocessing','resize')) for f in paths]))
    print(json.dumps({'files':[f.name for f in paths],'predictions':[saved['classes'][i] for i in indices],'text':''.join(saved['classes'][i] for i in indices)},indent=2))
if __name__=='__main__':main()
