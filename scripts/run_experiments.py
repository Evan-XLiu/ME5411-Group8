"""Reproduce Task 7 validation selection, final testing and chip inference."""
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from common.datasets import discover_classification_dataset, stratified_split, IMAGE_EXTENSIONS
from common.features import resize_nearest, extract_features
from common.image_io import load_grayscale
from task_07.cnn import build_model

def prepare(image,mode):
    if mode=='pad':
        h,w=image.shape; canvas=np.zeros((max(h,w),max(h,w)),dtype=image.dtype)
        canvas[(max(h,w)-h)//2:(max(h,w)-h)//2+h,(max(h,w)-w)//2:(max(h,w)-w)//2+w]=image
        image=canvas
    return resize_nearest(image).astype('float32')

def fit_cnn(images,labels,train_idx,eval_idx,config,epochs,seed):
    torch.manual_seed(seed)
    x=torch.from_numpy(np.stack(images)[:,None]/255); y=torch.tensor(labels)
    model=build_model(len(np.unique(labels)),architecture=config['architecture'])
    opt=torch.optim.Adam(model.parameters(),lr=config['lr']); loss_fn=torch.nn.CrossEntropyLoss()
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[train_idx],y[train_idx]),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
    history=[]; start=time.perf_counter()
    for epoch in range(epochs):
        model.train(); total=0
        for bx,by in loader:
            opt.zero_grad(); loss=loss_fn(model(bx),by);loss.backward();opt.step();total+=loss.item()*len(by)
        history.append({'epoch':epoch+1,'train_loss':total/len(train_idx)})
    elapsed=time.perf_counter()-start
    model.eval()
    with torch.no_grad(): pred=torch.cat([model(b).argmax(1) for b in x[eval_idx].split(128)]).numpy()
    return model,pred,elapsed,history

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=20);parser.add_argument('--data',type=Path,default=ROOT/'dataset_2026');args=parser.parse_args()
    torch.set_num_threads(1); seed=42
    out=ROOT/'results/experiments';out.mkdir(exist_ok=True,parents=True)
    models=ROOT/'results/models';models.mkdir(exist_ok=True,parents=True)
    raw,labels,classes=discover_classification_dataset(args.data)
    paths=[p for d in sorted(p for p in args.data.iterdir() if p.is_dir()) for p in sorted(d.rglob('*')) if p.suffix.lower() in IMAGE_EXTENSIONS]
    train_idx,test_idx=stratified_split(labels,.25,seed)
    subtrain,val=stratified_split(labels[train_idx],.2,seed);fit_idx=train_idx[subtrain];val_idx=train_idx[val]
    manifest={'seed':seed,'classes':classes,'class_counts':{c:int(sum(labels==i)) for i,c in enumerate(classes)},'samples':[{'path':p.relative_to(args.data).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'label':int(l)} for p,l in zip(paths,labels)],'train_indices':train_idx.tolist(),'test_indices':test_idx.tolist(),'fit_indices':fit_idx.tolist(),'validation_indices':val_idx.tolist()}
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    data={mode:[prepare(im,mode) for im in raw] for mode in ('resize','pad')}
    validation=[];final=[]
    for arch in ('baseline','yjn'):
        trials=[]
        for mode in data:
            for lr in (.001,.0003):
                cfg={'architecture':arch,'preprocessing':mode,'lr':lr}
                model,pred,sec,hist=fit_cnn(data[mode],labels,fit_idx,val_idx,cfg,args.epochs,seed)
                row={**cfg,'validation_accuracy':accuracy_score(labels[val_idx],pred),'seconds':sec};trials.append(row);validation.append(row);print(row,flush=True)
        best=max(trials,key=lambda r:r['validation_accuracy']);cfg={k:best[k] for k in ('architecture','preprocessing','lr')}
        model,pred,sec,hist=fit_cnn(data[cfg['preprocessing']],labels,train_idx,test_idx,cfg,args.epochs,seed)
        torch.save({'state_dict':model.state_dict(),'classes':classes,'architecture':arch,'preprocessing':cfg['preprocessing'],'image_size':32},models/f'{arch}_final.pt')
        crop_paths=sorted((ROOT/'results/figures/characters').glob('*.png'))
        x=torch.tensor(np.stack([prepare(255-load_grayscale(p),cfg['preprocessing']) for p in crop_paths])[:,None]/255)
        start=time.perf_counter()
        with torch.no_grad():chip=model(x).argmax(1).tolist()
        infer=time.perf_counter()-start
        final.append({**cfg,'name':arch,'test_accuracy':accuracy_score(labels[test_idx],pred),'train_seconds':sec,'chip_inference_seconds':infer,'parameters':sum(p.numel() for p in model.parameters()),'chip_polarity':'inverted to dark glyph on white background','chip_prediction':''.join(classes[i] for i in chip),'history':hist,'confusion_matrix':confusion_matrix(labels[test_idx],pred).tolist(),'classification_report':classification_report(labels[test_idx],pred,target_names=classes,output_dict=True)})
    import pickle
    feature_sets={mode:extract_features(ims) for mode,ims in data.items()}
    trials=[]
    for mode,features in feature_sets.items():
        for c in (1.,10.):
            model=make_pipeline(StandardScaler(),SVC(C=c,kernel='rbf'));start=time.perf_counter();model.fit(features[fit_idx],labels[fit_idx]);pred=model.predict(features[val_idx])
            row={'architecture':'hog_svm','preprocessing':mode,'C':c,'validation_accuracy':accuracy_score(labels[val_idx],pred),'seconds':time.perf_counter()-start};trials.append(row);validation.append(row);print(row,flush=True)
    best=max(trials,key=lambda r:r['validation_accuracy']);features=feature_sets[best['preprocessing']]
    model=make_pipeline(StandardScaler(),SVC(C=best['C'],kernel='rbf'));start=time.perf_counter();model.fit(features[train_idx],labels[train_idx]);sec=time.perf_counter()-start;pred=model.predict(features[test_idx])
    with (models/'hog_svm_final.pkl').open('wb') as f:pickle.dump({'classifier':model,'classes':classes,'preprocessing':best['preprocessing']},f)
    start=time.perf_counter();chip=model.predict(extract_features([prepare(255-load_grayscale(p),best['preprocessing']) for p in crop_paths]));infer=time.perf_counter()-start
    final.append({'name':'hog_svm','preprocessing':best['preprocessing'],'C':best['C'],'test_accuracy':accuracy_score(labels[test_idx],pred),'train_seconds':sec,'chip_inference_seconds':infer,'chip_polarity':'inverted to dark glyph on white background','chip_prediction':''.join(classes[i] for i in chip),'confusion_matrix':confusion_matrix(labels[test_idx],pred).tolist(),'classification_report':classification_report(labels[test_idx],pred,target_names=classes,output_dict=True)})
    result={'seed':seed,'epochs':args.epochs,'batch_size':32,'threads':1,'environment':{'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},'timing_note':'CPU, one torch thread. CNN training excludes input preparation; SVM training excludes HOG extraction. Chip timing includes SVM HOG extraction but CNN input preparation is excluded; not directly comparable end-to-end latency.','train_samples':len(train_idx),'test_samples':len(test_idx),'validation':validation,'final':final}
    (out/'task7_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(final),flush=True)
if __name__=='__main__':main()
