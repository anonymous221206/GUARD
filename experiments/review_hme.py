"""HME official model, with a held-out, fixed-mask MOSEI evaluation harness.

This is a contextual retraining comparator, NOT a frozen-host baseline.
Train with complete modalities; choose one checkpoint using mean validation
loss over the six incomplete masks. Never evaluate test during training.
"""
import argparse, hashlib, importlib, json, os, pickle, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = BERT = DATA = None
MASKS = {'t':[1,0,0], 'a':[0,1,0], 'v':[0,0,1], 'ta':[1,1,0],
         'tv':[1,0,1], 'av':[0,1,1], 'tav':[1,1,1]}

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def save_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(path)

def init(seed):
    import numpy as np, torch, transformers, sklearn, einops
    sys.path.insert(0,str(REPO/'HME_MSA'))
    old_argv=sys.argv
    sys.argv=['HME_main.py','--dataset','mosei','--d_l','192','--layers','2',
              '--hyper_depth','3','--latent_layers','3','--latent_dim','192',
              '--learning_rate','2e-5','--missing_rate','0','--seed',str(seed)]
    m=importlib.import_module('HME_main')
    sys.argv=old_argv
    m.set_random_seed(seed)
    torch.set_num_threads(4)
    m.get_tokenizer=lambda _:transformers.BertTokenizer.from_pretrained(str(BERT),local_files_only=True)
    versions={k:v.__version__ for k,v in [('numpy',np),('torch',torch),
                  ('transformers',transformers),('sklearn',sklearn),('einops',einops)]}
    return m,versions

def get_datasets(m,out):
    import numpy as np, torch
    cache=out.parent/'prepared.pt'
    if cache.exists(): return torch.load(cache,map_location='cpu')
    with DATA.open('rb') as f: raw=pickle.load(f)
    arrays={k:m.get_appropriate_dataset(v).tensors for k,v in raw.items()}
    labels={k:np.asarray([float(np.asarray(r[1]).reshape(-1)[0]) for r in v]) for k,v in raw.items()}
    cmad=np.load(ROOT/'artifacts/mosei_cmad/dumps/raw_features.npz',allow_pickle=True)
    assert np.array_equal(labels['test'],cmad['test_y'].reshape(-1)), 'CMAD test label order differs'
    # ID matching is retained for independent auditing of the preprocessing order.
    ids={k:[list(map(str,r[2])) for r in v] for k,v in raw.items()}
    meta={'source_sha256':digest(DATA),'counts':{k:len(v) for k,v in raw.items()},
          'ids':ids,'label_order_matches_cmad':True}
    save_json(out.parent/'data_manifest.json',meta)
    cache.parent.mkdir(parents=True,exist_ok=True)
    tmp=cache.with_suffix('.tmp'); torch.save(arrays,tmp); tmp.replace(cache)
    print('PREPARED',meta['counts'],flush=True)
    return arrays

def load_model(m):
    model,info=m.HME.from_pretrained(str(BERT),num_labels=1,args=m.args,
                                   local_files_only=True,output_loading_info=True)
    missing=[k for k in info['missing_keys'] if k.startswith('bert.') and not k.endswith('position_ids')]
    assert not missing, ('pretrained BERT not loaded',missing)
    # Activation recomputation preserves the full 256-sample batch and objective.
    model.bert.encoder.gradient_checkpointing=True
    return model.to(m.DEVICE),info

def loss_of(outputs,y):
    import torch.nn.functional as F
    task=F.l1_loss(outputs['predictions'].view(-1),y.view(-1))
    aux=sum(F.l1_loss(outputs[k].view(-1),y.view(-1)) for k in
            ('logits_l_p','logits_a_p','logits_v_p'))/3
    return task+0.8*aux+0.01*outputs['info_losses']

def run_batch(m,model,batch,mask,training=False):
    import torch
    ids,vis,ac,att,seg,y=[t.to(m.DEVICE) for t in batch]
    ps=torch.tensor(mask,device=m.DEVICE).expand(len(ids),3)
    outputs=(model if training else model.test)(ids,vis.squeeze(1),ac.squeeze(1),ps,
                      attention_mask=att,token_type_ids=seg,labels=None)
    return outputs,y

def evaluate(m,model,tensors,mask,batch_size=128,seed=0):
    import numpy as np, torch
    # Fork RNG prevents validation from changing the subsequent training stream.
    model.eval(); losses=[]; pred=[]
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        for start in range(0,len(tensors[0]),batch_size):
            batch=tuple(t[start:start+batch_size] for t in tensors)
            assert len(batch[0])>1, 'upstream squeeze() cannot process a singleton'
            o,y=run_batch(m,model,batch,mask)
            losses.append((float(loss_of(o,y)),len(y)))
            pred.append(o['predictions'].cpu().numpy().reshape(-1))
    return sum(l*n for l,n in losses)/sum(n for l,n in losses),np.concatenate(pred)

def main():
    global REPO, BERT, DATA
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','smoke','train','evaluate'])
    p.add_argument('--seed',type=int,default=5576)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--hme-repo',type=Path,default=os.environ.get('HME_REPO'),
                   help='official HME repository containing HME_MSA/')
    p.add_argument('--bert-dir',type=Path,default=os.environ.get('HME_BERT_DIR'),
                   help='local pretrained BERT directory used by the HME release')
    p.add_argument('--dataset',type=Path,default=os.environ.get('HME_MOSEI_PICKLE'),
                   help='CMU-MOSEI pickle in the official HME/CMAD layout')
    p.add_argument('--output-root',type=Path,
                   default=ROOT/'results/external_review_20260917/hme')
    p.add_argument('--cuda-visible-devices',default=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   help='optional CUDA_VISIBLE_DEVICES value, set before importing HME')
    a=p.parse_args()
    missing=[name for name,value in (('--hme-repo',a.hme_repo),('--bert-dir',a.bert_dir),
                                      ('--dataset',a.dataset)) if value is None]
    if missing: p.error('required path(s) missing: '+', '.join(missing))
    REPO, BERT, DATA = a.hme_repo.resolve(), a.bert_dir.resolve(), a.dataset.resolve()
    for name,path in (('HME repository',REPO/'HME_MSA'),('BERT directory',BERT),
                      ('MOSEI pickle',DATA)):
        if not path.exists(): p.error(f'{name} not found: {path}')
    if a.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES']=str(a.cuda_visible_devices)
    out=a.output_root.resolve()/f'seed{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    m,versions=init(a.seed)
    import numpy as np, torch
    tensors=get_datasets(m,out)
    if a.mode=='prepare': return
    model,load_info=load_model(m)
    meta={'method':'HME official model with fixed-mask evaluation harness',
          'upstream_commit':'cdb1d60b30bddb5ba7fe7762dadaf7e21dc8619f',
          'seed':a.seed,'versions':versions,'args':vars(m.args),
          'upstream_loading_info':load_info,'data_sha256':digest(DATA),
          'driver_sha256':digest(__file__),
          'train_mask':'tav','validation_masks':list(MASKS)[:-1],
          'selection':'mean per-example official-code loss over six incomplete validation masks',
          'train_batch':256,'eval_batch':128,'gradient_checkpointing':'BERT encoder',
          'test_label_use':'evaluation only; no HME fitting/calibration',
          'test_context':'only current held-out partition; cross-sample HME context within each ordered batch',
          'reference_recipe':'released code lr=2e-5, task+0.8 aux+0.01 info, AdamW reset each epoch',
          'paper_difference':'paper appendix lists lr=4e-5 and a different loss notation; this run follows released code',
          'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
    if a.mode=='smoke':
        model.train(); opt=torch.optim.AdamW(model.parameters(),lr=2e-5)
        t0=time.time(); b=tuple(t[:256] for t in tensors['train'])
        o,y=run_batch(m,model,b,MASKS['tav'],True); loss=loss_of(o,y)
        assert torch.isfinite(loss); loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step(); torch.cuda.synchronize()
        meta.update(loss=float(loss),seconds=time.time()-t0,
                    max_allocated_bytes=torch.cuda.max_memory_allocated(),status='SMOKE_PASS')
        save_json(out/'smoke.json',meta); print(json.dumps(meta),flush=True); return
    if a.mode=='train':
        if (out/'best.pt').exists(): raise RuntimeError('Refuse to overwrite existing training; choose a new seed/directory')
        save_json(out/'protocol.json',meta)
        save_json(out/'status.json',{'status':'TRAINING','seed':a.seed,'epoch':-1})
        best=float('inf'); stale=0; hist=[]
        dataset=torch.utils.data.TensorDataset(*tensors['train'])
        loader=torch.utils.data.DataLoader(dataset,batch_size=256,shuffle=True,num_workers=0)
        for epoch in range(a.epochs):
            t0=time.time(); model.train()
            no_decay=['bias','LayerNorm.weight']
            groups=[{'params':[v for k,v in model.named_parameters() if not any(x in k for x in no_decay)],'weight_decay':m.args.weight_decay},
                    {'params':[v for k,v in model.named_parameters() if any(x in k for x in no_decay)],'weight_decay':0.0}]
            opt=m.AdamW(groups,lr=m.args.learning_rate,eps=m.args.adam_epsilon)
            m.adjust_learning_rate(opt,epoch,m.args)
            total=0.; n=0
            for step,batch in enumerate(loader):
                opt.zero_grad(); o,y=run_batch(m,model,batch,MASKS['tav'],True)
                loss=loss_of(o,y)
                if not torch.isfinite(loss): raise FloatingPointError('non-finite training loss')
                loss.backward(); torch.nn.utils.clip_grad_value_(model.parameters(),m.args.clip); opt.step()
                total+=float(loss)*len(y); n+=len(y)
                if step%16==0: print('TRAIN',a.seed,epoch,step,float(loss),flush=True)
            del opt
            vals={k:evaluate(m,model,tensors['dev'],v,seed=a.seed+100+j)[0]
                  for j,(k,v) in enumerate(list(MASKS.items())[:-1])}
            val=float(np.mean(list(vals.values())))
            row={'epoch':epoch,'train_loss':total/n,'validation_loss':val,
                 'validation_by_mask':vals,'seconds':time.time()-t0}
            if val<best:
                best=val; stale=0
                tmp=out/'best.tmp'; torch.save({'model':model.state_dict(),'epoch':epoch,'validation_loss':best},tmp); tmp.replace(out/'best.pt')
            else: stale+=1
            hist.append(row); save_json(out/'history.json',hist)
            save_json(out/'status.json',dict(status='TRAINING',seed=a.seed,**row,patience_used=stale))
            print('EPOCH',json.dumps(row),flush=True)
            if stale>=a.patience: break
        save_json(out/'status.json',{'status':'TRAINING_COMPLETE','seed':a.seed,'epochs':len(hist),'best_validation_loss':best})
        # The test split is evaluated only after checkpoint selection is closed.
    checkpoint=torch.load(out/'best.pt',map_location='cpu'); model.load_state_dict(checkpoint['model'])
    from sklearn.metrics import f1_score
    rows=[]; predictions={}
    raw_y=tensors['test'][-1].numpy().reshape(-1)
    for split_seed in range(5):
        _,_,_,idx=np.array_split(np.random.default_rng(split_seed).permutation(len(raw_y)),4)
        test=tuple(t[idx] for t in tensors['test']); y=raw_y[idx]; keep=y!=0
        for j,(mask,bits) in enumerate(MASKS.items()):
            _,pred=evaluate(m,model,test,bits,seed=a.seed+1000+split_seed*10+j)
            gold=y[keep]>0; decision=pred[keep]>0
            rows.append({'training_seed':a.seed,'split_seed':split_seed,'mask':mask,
                         'n':len(y),'n_nonzero':int(keep.sum()),'accuracy':float(np.mean(decision==gold)),
                         'f1_weighted':float(f1_score(gold,decision,average='weighted'))})
            predictions[f's{split_seed}_{mask}']=pred
        predictions[f's{split_seed}_indices']=idx
    save_json(out/'metrics.json',{'selected_epoch':checkpoint['epoch'],'rows':rows,'protocol':meta})
    np.savez_compressed(out/'predictions.npz',**predictions)
    save_json(out/'status.json',{'status':'COMPLETE','seed':a.seed,'selected_epoch':checkpoint['epoch'],'evaluation_cells':len(rows)})
    print('COMPLETE',a.seed,len(rows),flush=True)

if __name__=='__main__':
    try: main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
