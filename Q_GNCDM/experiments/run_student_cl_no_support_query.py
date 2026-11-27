"""Record-wise GNCDM training; joint-view interaction InfoNCE, no episodes."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from Q_GNCDM.experiments import run_simple_contrastive as core
from Q_GNCDM.experiments import run_prepared_dataset as prepared
from Q_GNCDM.experiments.controlled_student_cl import same_student_infonce

ROOT = Path(__file__).resolve().parents[2]
ID = 'student_cl_no_support_query_20261124'
WORK = ROOT/'Q_GNCDM/scratch'/ID
REPORT = ROOT/'Q_GNCDM/docs/experiments/student_cl_no_support_query'
DATASETS = ['math1','math2','assist2012','junyi2015','eedi_task34','synthetic_weak_q']
CONFIGS = {'R2':(.1,.01,.5,.1),'R3':(.01,.001,.5,.1),
           'R4':(.1,.01,.7,.5),'R5':(.01,.001,.7,.5)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def load(name):
    base=json.loads((ROOT/'Q_GNCDM/results/student_cl_control_math1_20261123/manifest.json').read_text())['args']
    base.update(experiment_id=ID,dataset=name,data_dir=str(ROOT/'Q_GNCDM/data'/name),batch_records=128)
    base.pop('query_fraction',None);base.pop('batch_students',None)
    args=argparse.Namespace(**base)
    if name=='synthetic_weak_q':
        args.dataset='synthetic';args.data_dir=str(Path(args.data_dir)/'seed_2026')
        args.q_noise=.2;args.split_seed=2026
        data=core.load_data(args)
    else:data=prepared.load_data(Path(args.data_dir))
    return data,args


def mix64(x):
    with np.errstate(over='ignore'):
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        return x^(x>>np.uint64(31))


def noise_view(x, coordinates, ratio, scale, key):
    """Coordinate-keyed uniform noise: shared row/column entries stay consistent.

    Local seed keys never consume global RNG. Missing positions remain zero.
    Noise is additive and clipped, not binary flipping or magnitude-thresholded.
    """
    if not 0<=ratio<=1 or not 0<=scale<1:raise ValueError('Invalid noise')
    out=x.copy();obs=x!=0
    if ratio==0 or scale==0:
        selected=np.zeros_like(obs);raw=out
    else:
        key=np.uint64(key)
        # Row repetition and missing entries need no repeated hashing. Restore
        # original order before statistics; values are bitwise identical.
        _,first,inverse=np.unique(coordinates[:,0],return_index=True,return_inverse=True)
        reduced=x[first].copy();valid=reduced!=0;coords=coordinates[first][valid]
        u=(mix64(coords^key)>>np.uint64(11)).astype(np.float64)/(2**53)
        v=(mix64(coords^key^np.uint64(0x9e3779b97f4a7c15))>>np.uint64(11)).astype(np.float64)/(2**53)
        selected_small=np.zeros_like(valid);selected_small[valid]=u<ratio
        values=reduced[valid];take=u<ratio
        values[take]+=((2*v[take]-1)*scale).astype(np.float32)
        reduced[valid]=values
        raw=reduced[inverse];selected=selected_small[inverse]
        out=np.clip(raw,-1,1).astype(np.float32)
    delta=np.abs(out-x);count=max(int(obs.sum()),1)
    return out,{'selected_ratio':float(selected.sum()/count),'changed_ratio':float((delta>0).sum()/count),
                'mae':float(delta.sum()/count),'max_change':float(delta.max()),
                'clipped_ratio':float(((raw<-1)|(raw>1)).sum()/count)}


def inputs(data, rows):
    # Deliberately use all training evidence, including the target training entry.
    users,items=rows[:,0],rows[:,1]
    return data['train'][users].copy(),data['train'][:,items].T.copy()


def interaction(model, student, item, ids):
    theta=model.diagnose_theta(student);psi=model.diagnose_psi(item)
    q=model.get_q_soft()[ids]
    return model.theta_agg_mat(theta*q)-model.psi_agg_mat(psi*q)


def views(data, rows, student, item, cfg, seed, epoch, step, view):
    ratio,scale=cfg[:2] if view==0 else cfg[2:]
    key=np.random.SeedSequence([1123,seed,epoch,step,view]).generate_state(1,dtype=np.uint64)[0]
    n=data['train'].shape[1]
    uc=rows[:,0,None].astype(np.uint64)*np.uint64(n)+np.arange(n,dtype=np.uint64)[None,:]
    ic=np.arange(len(data['train']),dtype=np.uint64)[None,:]*np.uint64(n)+rows[:,1,None].astype(np.uint64)
    u,us=noise_view(student,uc,ratio,scale,key)
    v,vs=noise_view(item,ic,ratio,scale,key)
    return u,v,{**{'student_'+k:x for k,x in us.items()},**{'item_'+k:x for k,x in vs.items()}}


def forward_loss(model,data,rows,coefficient,cfg,seed,epoch,step,reference=False):
    u,v=inputs(data,rows)
    student=torch.from_numpy(u);item=torch.from_numpy(v)
    users=torch.as_tensor(rows[:,0]);items=torch.as_tensor(rows[:,1])
    target=torch.as_tensor(rows[:,2],dtype=torch.float32).reshape(-1,1)
    captured=[]
    # At zero coefficient there is no hook, no augmentation or extra forward.
    hook=model.ncd.register_forward_pre_hook(lambda module,args:captured.append(args[0])) if coefficient else None
    try:pred=model(student,item,users,items)
    finally:
        if hook:hook.remove()
    losses=model.compute_loss(pred,target);original=losses['loss_total']
    stats={'loss_original':float(original.detach()),'loss_prediction':float(losses['loss_rec'].detach())}
    cl=None
    if coefficient:
        hs=[]
        for view in (0,1):
            up,vp,s=views(data,rows,u,v,cfg,seed,epoch,step,view)
            hs.append(interaction(model,torch.from_numpy(up),torch.from_numpy(vp),items))
            stats.update({('positive_' if view==0 else 'negative_')+k:x for k,x in s.items()})
        cl,diagnostics=same_student_infonce(captured[0],hs[0],hs[1],.2)
        stats.update(diagnostics)
    total=original if cl is None else original+coefficient*cl
    stats.update(loss_cl=0. if cl is None else float(cl.detach()),loss_total=float(total.detach()))
    return total,original,cl,stats,pred


def grad_norm(grads):
    return float(sum(g.detach().double().square().sum() for g in grads if g is not None).sqrt())


def gradient_stats(model,original,cl):
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad and not n.startswith('ncd.')]
    a=torch.autograd.grad(original,[p for _,p in named],retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(cl,[p for _,p in named],retain_graph=True,allow_unused=True)
    norm=grad_norm(b)
    assert np.isfinite(norm) and norm>0
    groups={prefix:grad_norm([g for (n,_),g in zip(named,b) if n.startswith(prefix)]) for prefix in ['f_nn.','g_nn.','theta_agg_mat.','psi_agg_mat.']}
    return {'cl_gradient':norm,'weighted_gradient_ratio':.1*norm/max(grad_norm(a),1e-30),**groups}


def reference_loss(model,data,rows):
    # Literal IDCDataset behavior: full training row/column per interaction.
    u=torch.stack([torch.tensor(data['train'][int(r[0])]) for r in rows])
    v=torch.stack([torch.tensor(data['train'][:,int(r[1])]) for r in rows])
    pred=model(u,v,torch.tensor(rows[:,0]),torch.tensor(rows[:,1]))
    return model.compute_loss(pred,torch.tensor(rows[:,2],dtype=torch.float32).reshape(-1,1))['loss_total'],pred


def train(name,mode,seed,condition):
    torch.set_num_threads(1)
    data,args=load(name);out=WORK/name/mode/f'seed_{seed}'/condition
    if (out/'complete.json').exists():return
    if out.exists():raise RuntimeError(f'Incomplete run must be reviewed: {out}')
    out.mkdir(parents=True)
    cfg=CONFIGS.get(condition,CONFIGS['R2']);weight=0. if condition in ['R0','R1'] else .1
    model=core.build_model(data,args,mode,seed);opt=core.build_optimizer(model,args)
    write(out/'config.json',{'args':vars(args),'mode':mode,'seed':seed,'condition':condition,'noise':cfg,'lambda_cl':weight})
    history=[];best=-float('inf');bad=0;started=time.time()
    for epoch in range(1,args.epochs+1):
        model.train();order=np.random.default_rng([seed,epoch,0]).permutation(len(data['rows'][0]));batch_stats=[]
        for step,start in enumerate(range(0,len(order),args.batch_records)):
            rows=data['rows'][0][order[start:start+args.batch_records]]
            if condition=='R0':
                loss,_=reference_loss(model,data,rows);stats={'loss_original':float(loss.detach()),'loss_cl':0.,'loss_total':float(loss.detach())}
            else:
                loss,original,cl,stats,_=forward_loss(model,data,rows,weight,cfg,seed,epoch,step)
                if cl is not None and step==0:probe=gradient_stats(model,original,cl)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();batch_stats.append(stats)
        valid,_,_=core.evaluate(model,data['train'],data['rows'][1])
        h={k:float(np.mean([s[k] for s in batch_stats])) for k in batch_stats[0]}
        h.update(epoch=epoch,**{'valid_'+k:v for k,v in valid.items()})
        if weight:h.update({'probe_'+k:v for k,v in probe.items()})
        history.append(h);core.save_csv(out/'history.csv',history)
        if valid['acc']>best:
            best=valid['acc'];bad=0;best_epoch=epoch;state=deepcopy(model.state_dict())
        else:bad+=1
        print(name,mode,seed,condition,'epoch',epoch,'valid_ACC',valid['acc'],flush=True)
        if bad>=args.patience:break
    model.load_state_dict(state)
    valid,_,_=core.evaluate(model,data['train'],data['rows'][1]);test,_,_=core.evaluate(model,data['train'],data['rows'][2])
    torch.save(state,out/'best.pt')
    metrics={'dataset':name,'mode':mode,'seed':seed,'condition':condition,'best_epoch':best_epoch,'epochs':len(history),
             'seconds':time.time()-started,**{'valid_'+k:v for k,v in valid.items()},**{'test_'+k:v for k,v in test.items()}}
    write(out/'metrics.json',metrics)
    # Disk checkpoint replay; a different RNG state cannot affect eval predictions.
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True))
    replay,_,_=core.evaluate(model,data['train'],data['rows'][2]);assert replay==test
    if condition=='R1':
        r0=out.parent/'R0';s0=torch.load(r0/'best.pt',weights_only=True)
        assert all(torch.equal(s0[k],state[k]) for k in state)
        baseline=json.loads((r0/'metrics.json').read_text())
        assert all(metrics[k]==baseline[k] for k in metrics if k.startswith(('valid_','test_')) or k in ['best_epoch','epochs'])
        old=list(csv.DictReader((r0/'history.csv').open(encoding='utf-8-sig')))
        assert all(float(a[k])==b[k] for a,b in zip(old,history) for k in a)
    write(out/'complete.json',{'replay':True,'zero_exact_parity':condition=='R1'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True)
    p.add_argument('--mode',choices=['A0','A1'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--condition',choices=['R0','R1',*CONFIGS],required=True);a=p.parse_args()
    try:train(a.dataset,a.mode,a.seed,a.condition)
    except Exception:
        folder=WORK/a.dataset/a.mode/f'seed_{a.seed}'/a.condition;folder.mkdir(parents=True,exist_ok=True)
        (folder/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
