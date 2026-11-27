"""Frozen-training-loop 2x2 noise sensitivity; only observation hooks are added."""
import argparse
import contextlib
import csv
import hashlib
import json
import subprocess
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from Q_GNCDM.experiments import controlled_student_cl as cl
from Q_GNCDM.experiments import run_controlled_student_cl as prior

ROOT = prior.ROOT
OLD = ROOT / 'Q_GNCDM/results/student_cl_control_math1_20261123'
CONFIGS = {'R2': (.1, .01, .5, .1), 'R3': (.01, .001, .5, .1),
           'R4': (.1, .01, .7, .5), 'R5': (.01, .001, .7, .5)}
ORIGINAL_LOSS = cl.same_student_infonce


def norm(grads):
    return float(sum(g.detach().double().square().sum() for g in grads if g is not None).sqrt())


def observed_loss(model, anchor, positive, negative, temperature):
    loss, stats = ORIGINAL_LOSS(anchor, positive, negative, temperature)
    a, p, n = [F.normalize(t, dim=1) for t in (anchor, positive, negative)]
    pos_logit, neg_logit = (a*p).sum(1)/temperature, (a*n).sum(1)/temperature
    logits = torch.stack((pos_logit, neg_logit), dim=1)
    stats.update({'positive_energy': float(-pos_logit.mean().detach()),
                  'joint_log_partition': float(logits.logsumexp(1).mean().detach()),
                  'negative_relative_penalty': float(F.softplus(neg_logit-pos_logit).mean().detach())})
    # These are observations only; original loss is returned untouched.
    if torch.is_grad_enabled():
        gp, gn = torch.autograd.grad(loss, (positive, negative), retain_graph=True)
        pn, nn = gp.detach().double().norm(dim=1), gn.detach().double().norm(dim=1)
        params = [p for _, p in cl.shared_parameters(model)]
        weight = logits.softmax(1)[:, 1].detach()/len(anchor)
        pos_part = torch.autograd.grad(pos_logit, params, grad_outputs=-weight,
                                       retain_graph=True, allow_unused=True)
        neg_part = torch.autograd.grad(neg_logit, params, grad_outputs=weight,
                                       retain_graph=True, allow_unused=True)
        stats.update({'positive_view_grad_norm_mean': float(pn.mean()),
                      'negative_view_grad_norm_mean': float(nn.mean()),
                      'negative_view_grad_nonzero_fraction': float((nn > 0).double().mean()),
                      'negative_view_grad_above_1e_12_fraction': float((nn > 1e-12).double().mean()),
                      'positive_shared_partial_grad_norm': norm(pos_part),
                      'negative_shared_partial_grad_norm': norm(neg_part)})
    return loss, stats


def observed_augment(x, ratio, scale, rng, distribution='uniform'):
    out, stats = cl.augment(x, ratio, scale, rng, distribution)
    delta = np.abs(out-x)
    mask = x != 0
    stats.update({'input_mse_observed': float(np.square(delta).sum()/max(int(mask.sum()),1)),
                  'clipped_fraction_observed': stats['selected_fraction']*stats['clipped_fraction_selected'],
                  'input_mae_changed': float(delta.sum()/max(int((delta>0).sum()),1)),
                  'unchanged_student_fraction': float((delta.max(1)==0).mean())})
    return out, stats


def branch(model, clean_np, anchor, args, seed, epoch, step):
    pos, ps = observed_augment(clean_np, args.positive_noise_ratio, args.positive_noise_scale,
                               np.random.default_rng([args.contrastive_seed,seed,epoch,step,0]))
    neg, ns = observed_augment(clean_np, args.negative_noise_ratio, args.negative_noise_scale,
                               np.random.default_rng([args.contrastive_seed,seed,epoch,step,1]))
    positive = model.diagnose_theta(torch.as_tensor(pos, device=anchor.device))
    negative = model.diagnose_theta(torch.as_tensor(neg, device=anchor.device))
    loss, stats = observed_loss(model, anchor, positive, negative, args.temperature)
    stats.update({'positive_'+k:v for k,v in ps.items()})
    stats.update({'negative_'+k:v for k,v in ns.items()})
    return loss, stats


def config(base, condition):
    args = argparse.Namespace(**deepcopy(base))
    (args.positive_noise_ratio,args.positive_noise_scale,
     args.negative_noise_ratio,args.negative_noise_scale) = CONFIGS[condition]
    return args


def preflight(data, args, mode):
    result = prior.preflight(data,args,mode)
    model = prior.build_model(data,args,mode,2026)
    ids=np.flatnonzero((data['train']!=0).sum(1)>=2)[:128]
    episode=prior.make_student_episode(data['train'],ids,args.query_fraction,np.random.default_rng(19))
    clean,m,_,losses=prior.batch_forward(model,episode)
    rng=torch.get_rng_state().clone()
    state=deepcopy(model.state_dict())
    old_loss,_=cl.contrastive_branch(model,clean,m,args,2026,1,0)
    params=[p for _,p in cl.shared_parameters(model)]
    old_grad=torch.autograd.grad(old_loss,params,retain_graph=True)
    new_loss,diagnostics=branch(model,clean,m,args,2026,1,0)
    new_grad=torch.autograd.grad(new_loss,params,retain_graph=True)
    assert torch.equal(old_loss,new_loss)
    assert all(torch.equal(a,b) for a,b in zip(old_grad,new_grad))
    assert torch.equal(torch.get_rng_state(),rng)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert diagnostics['negative_view_grad_nonzero_fraction']>0
    assert abs(diagnostics['positive_energy']+diagnostics['joint_log_partition']-float(new_loss.detach()))<1e-6
    result.update({'observed_branch_loss_and_gradient_exact':True,'observation_state_and_rng_unchanged':True,
                   'extra_diagnostics':diagnostics})
    return result


@torch.no_grad()
def probe(model, clean, args, seed):
    theta=model.diagnose_theta(torch.as_tensor(clean,device=model.device))
    loss,st=branch(model,clean,theta,args,seed,0,0)
    return {'probe_loss_cl':float(loss),**{'probe_'+k:v for k,v in st.items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id',default='student_cl_noise_math1_20261123')
    cli=parser.parse_args()
    assert Path(cli.experiment_id).name==cli.experiment_id
    torch.set_num_threads(1)
    previous=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    assert json.loads((OLD/'completed.json').read_text(encoding='utf-8'))['status']=='passed'
    for path,digest in previous['code_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    destination=ROOT/'Q_GNCDM/results'/cli.experiment_id
    destination.mkdir(parents=True,exist_ok=False)
    manifest={**previous,'experiment_id':cli.experiment_id,'parent_experiment':str(OLD.relative_to(ROOT)),
              'noise_configs':CONFIGS,'seeds':[2026,2027,2028],
              'command':subprocess.list2cmdline([sys.executable,'-m','Q_GNCDM.experiments.run_student_cl_noise_sensitivity',*sys.argv[1:]]),
              'new_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'selection_between_seeds':'highest validation ACC, ties lowest seed; report both test metrics from same run',
              'changes':'only positive/negative ratios and scales; observations do not affect training',
              'loss_components':'positive_energy + joint_log_partition = InfoNCE; no independent negative loss or margin',
              'gradient_activation':'fraction of negative-view row gradients with L2 norm >0 and >1e-12',
              'epoch_diagnostics':'arithmetic batch means; probe uses fixed support/students/noise throughout epochs'}
    prior.legacy.write_json(destination/'manifest.json',manifest)
    try:
        data=prior.core.load_data(argparse.Namespace(**previous['args']))
        checks={}
        # All single-batch checks precede any full training.
        for mode in ('A0','A1'):
            for condition in CONFIGS:
                checks[mode+'_'+condition]=preflight(data,config(previous['args'],condition),mode)
                prior.legacy.write_json(destination/'preflight.json',checks)
        print('PREFLIGHT: all 8 Q/noise settings passed',flush=True)
        summaries,parity=[] , []
        for seed in (2026,2027,2028):
            for mode in ('A0','A1'):
                for condition in CONFIGS:
                    args=config(previous['args'],condition)
                    args.experiment_id=cli.experiment_id
                    out=destination/mode/f'seed_{seed}'/condition
                    out.parent.mkdir(parents=True,exist_ok=True)
                    ids=np.flatnonzero((data['train']!=0).sum(1)>=2)[:128]
                    ep=prior.make_student_episode(data['train'],ids,args.query_fraction,np.random.default_rng([seed,1,0,1]))
                    clean=ep.support[ids].copy()
                    model=prior.build_model(data,args,mode,seed)
                    probes=[{'epoch':0,**probe(model,clean,args,seed)}]
                    original_eval=prior.core.evaluate
                    def observed_eval(model,evidence,rows):
                        result=original_eval(model,evidence,rows)
                        probes.append({'epoch':len(probes),**probe(model,clean,args,seed)})
                        return result
                    # Prior function creates out itself; console log lives beside it.
                    log_path=out.parent/(condition+'_console.log')
                    with log_path.open('w',encoding='utf-8') as handle,contextlib.redirect_stdout(handle):
                        with patch.object(prior,'contrastive_branch',branch),patch.object(prior,'evaluate',observed_eval):
                            result=prior.controlled_train_one(data,args,mode,seed,out)
                    log_path.rename(out/'console.log')
                    result.update({'condition':condition,'lambda_cl':.1})
                    prior.legacy.write_json(out/'metrics.json',result)
                    # Remove two final best-checkpoint eval observations from epoch trajectory.
                    assert len(probes)==result['epochs_run']+3
                    prior.save_csv(out/'fixed_probe_curve.csv',probes[:result['epochs_run']+1])
                    prior.legacy.write_json(out/'best_checkpoint_probe.json',probes[-1])
                    state=prior.checkpoint_state(out)
                    if condition=='R2':
                        old_run=OLD/mode/f'seed_{seed}'/'R2'
                        old_state=prior.checkpoint_state(old_run)
                        assert all(torch.equal(v,old_state[k]) for k,v in state.items()),'R2 state changed'
                        a=list(csv.DictReader((old_run/'train_log.csv').open(encoding='utf-8-sig')))
                        b=list(csv.DictReader((out/'train_log.csv').open(encoding='utf-8-sig')))
                        assert len(a)==len(b)
                        assert all(all(row[k]==new[k] for k in row) for row,new in zip(a,b))
                        parity.append({'mode':mode,'seed':seed,'R2_checkpoint_and_previous_log_exact':True})
                        prior.legacy.write_json(destination/'R2_parity.json',parity)
                    flat={k:v for k,v in result.items() if not isinstance(v,dict)}
                    flat.update(result['training_means'])
                    summaries.append(flat)
                    prior.save_csv(destination/'summary.csv',summaries)
                    print(f'{mode} seed={seed} {condition}: ACC={result["test_acc"]:.9f} AUC={result["test_auc"]:.9f} best={result["best_epoch"]}',flush=True)
        for path,digest in previous['code_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
        for path,digest in previous['data_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==manifest['new_script_sha256']
        prior.legacy.write_json(destination/'completed.json',{'status':'passed','runs':len(summaries),
                               'R2_exact_parity_pairs':len(parity),'protected_code_and_data_unchanged':True})
    except BaseException:
        (destination/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':
    main()
