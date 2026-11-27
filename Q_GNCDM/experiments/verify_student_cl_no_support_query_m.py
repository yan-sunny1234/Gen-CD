"""Preflight record-wise, dual-input interaction CL before any full training."""
import argparse
import hashlib
import importlib.util
import traceback
from copy import deepcopy
from unittest.mock import patch
import numpy as np
import torch
from Q_GNCDM.experiments import run_student_cl_no_support_query_m as r


def verify(name):
    torch.set_num_threads(1);data,args=r.load(name)
    rows=data['rows'][0][np.random.default_rng(2026).choice(len(data['rows'][0]),128,replace=False)]
    u,v=r.inputs(data,rows);original=data['train'].copy()
    assert np.array_equal(u[np.arange(len(rows)),rows[:,1]],2*rows[:,2]-1)
    assert np.array_equal(v[np.arange(len(rows)),rows[:,0]],2*rows[:,2]-1)
    checks={}
    for mode in ['A0','A1']:
        base=r.core.build_model(data,args,mode,2026)
        # Fixed-Q forward parity against original GNCDM class at identical parameters.
        if mode=='A0':
            spec=importlib.util.spec_from_file_location('original_gncdm_model',r.ROOT/'GNCDM/model.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            old=module.GNCDM(len(data['train']),len(data['q']),data['q'].shape[1],32,32,args.alpha,Q_mat=data['q'],device=torch.device('cpu'))
            old_params=dict(old.named_parameters())
            with torch.no_grad():
                for n,p in base.named_parameters():
                    if n in old_params:old_params[n].copy_(p)
            base.eval();old.eval()
            x,y=torch.from_numpy(u),torch.from_numpy(v);ids=torch.tensor(rows[:,1]).reshape(-1,1)
            assert torch.equal(base(x,y,torch.tensor(rows[:,0]),ids),old(x,y,torch.tensor(rows[:,0]),ids))
        base.train();reference=deepcopy(base);zero=deepcopy(base)
        rng=torch.get_rng_state().clone()
        l0,p0=r.reference_loss(reference,data,rows);l0.backward()
        r.core.build_optimizer(reference,args).step();after=torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        with patch.object(r,'student_view',side_effect=AssertionError('zero branch entered')):
            l1,_,cl,_,p1=r.forward_loss(zero,data,rows,0,r.CONFIGS['R2'],2026,1,0)
        assert cl is None and torch.equal(l0,l1) and torch.equal(p0,p1)
        l1.backward();r.core.build_optimizer(zero,args).step()
        assert torch.equal(after,torch.get_rng_state())
        assert all(torch.equal(reference.state_dict()[k],zero.state_dict()[k]) for k in zero.state_dict())
        checks[mode]={'zero_predictions_loss_gradients_updates_rng_exact':True,'original_fixedQ_forward_exact':mode=='A0'}
        for cond,cfg in r.CONFIGS.items():
            positive,ps0=r.student_view(data,rows,u,cfg,2026,1,0,0)
            pi=v.copy();ps={'student_'+k:x for k,x in ps0.items()}
            negative,ns0=r.student_view(data,rows,u,cfg,2026,1,0,1)
            ni=v.copy();ns={'student_'+k:x for k,x in ns0.items()}
            for source,view in [(u,positive),(u,negative)]:
                assert source.shape==view.shape and source.dtype==view.dtype==np.float32
                assert np.isfinite(view).all() and abs(view).max()<=1
                assert np.array_equal(view[source==0],source[source==0])
                assert np.any(view!=source)
            # Same coordinate in row/column receives exactly the same noise.
            assert np.array_equal(pi,v)
            assert np.array_equal(ni,v)
            assert ps['student_mae']<ns['student_mae']
            again,_=r.student_view(data,rows,u,cfg,2026,1,0,0);assert np.array_equal(positive,again)
            other,_=r.student_view(data,rows,u,cfg,2027,1,0,0);assert not np.array_equal(positive,other)
            coords=np.arange(u.size,dtype=np.uint64).reshape(u.shape)
            for ratio,scale in [(0,.1),(.5,0)]:assert np.array_equal(r.noise_view(u,coords,ratio,scale,1)[0],u)
            model=deepcopy(base);torch.set_rng_state(rng)
            total,original_loss,cl,stats,pred=r.forward_loss(model,data,rows,.1,cfg,2026,1,0)
            assert torch.equal(pred,p0) and torch.equal(total,original_loss+.1*cl)
            g=r.gradient_stats(model,original_loss,cl)
            assert g['f_nn.']>0
            assert all(g[k]==0 for k in ['g_nn.','theta_agg_mat.','psi_agg_mat.'])
            unrelated=[p for n,p in model.named_parameters() if n.startswith(('g_nn.','theta_agg_mat.','psi_agg_mat.','ncd.'))]
            assert all(x is None for x in torch.autograd.grad(cl,unrelated,allow_unused=True,retain_graph=True))
            total.backward();r.core.build_optimizer(model,args).step()
            differences={prefix:sum(float((p-zero.state_dict()[n]).double().square().sum()) for n,p in model.state_dict().items() if n.startswith(prefix))**.5 for prefix in ['f_nn.','g_nn.','theta_agg_mat.','psi_agg_mat.']}
            assert differences['f_nn.']>0
            assert all(differences[k]==0 for k in ['g_nn.','theta_agg_mat.','psi_agg_mat.'])
            checks[mode][cond]={'status':'passed','gradients':g,'no_direct_item_or_aggregation_cl_gradient':True,'update_difference_L2':differences,'positive':ps,'negative':ns,'cl_loss':float(cl.detach())}
    assert np.array_equal(original,data['train'])
    r.write(r.WORK/name/'preflight.json',checks)
    print(name,'ALL PREFLIGHTS PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=r.DATASETS,required=True);a=p.parse_args()
    try:verify(a.dataset)
    except Exception:
        r.WORK.mkdir(parents=True,exist_ok=True)
        (r.WORK/(a.dataset+'_preflight_failure.log')).write_text(traceback.format_exc(),encoding='utf-8');raise
