"""Scheduling-only fork: frozen worker loop restricted to seed 2028.

No model/optimizer/noise or selection changes. A complete verified six-condition
Q group is atomically published only if the primary worker has not started it.
"""
from Q_GNCDM.experiments.run_student_cl_all_datasets import *
from Q_GNCDM.experiments import run_student_cl_all_datasets as original
WORK=original.WORK/'_seed2028_helpers'


def publish(name,mode,seed,folder):
    source=folder/mode/f'seed_{seed}'
    target=original.WORK/name/mode/f'seed_{seed}'
    assert source.resolve().is_relative_to(WORK.resolve())
    assert target.resolve().is_relative_to(original.WORK.resolve())
    assert all((source/c/'verified.json').exists() for c in ('R0','R1','R2','R3','R4','R5'))
    event={'mode':mode,'seed':seed,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'source':str(Path(__file__).relative_to(ROOT)),'changes':'seed scheduling only, same frozen worker logic',
           'status':'primary_already_started; no overwrite'}
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        try:
            source.rename(target)
            event['status']='published_complete_verified_group'
        except FileExistsError:
            pass
    log=original.WORK/name/f'seed2028_{mode}_execution.json'
    prior.legacy.write_json(log,event)
    print('PUBLISH',name,mode,event['status'],flush=True)

def worker(name):
    torch.set_num_threads(1)
    folder=WORK/name;folder.mkdir(parents=True,exist_ok=True)
    try:
        parent=json.loads((noise.OLD/'manifest.json').read_text(encoding='utf-8'))
        data,args=load_dataset(name,parent['args'])
        sources={**parent['code_sha256'],str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 str(Path(noise.__file__).relative_to(ROOT)):hashlib.sha256(Path(noise.__file__).read_bytes()).hexdigest(),
                 str(Path(prepared.__file__).relative_to(ROOT)):hashlib.sha256(Path(prepared.__file__).read_bytes()).hexdigest()}
        for p,digest in sources.items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
        manifest={'experiment_id':f'student_cl_all_{name}_20261123','dataset':name,'args':vars(args),
                  'noise_configs':noise.CONFIGS,'source_sha256':sources,'data_sha256':data['source_sha256'],
                  'train_shape':list(data['train'].shape),'concepts':int(data['q'].shape[1]),
                  'split_lengths':[len(x) for x in data['rows']], 'seeds':[2028],
                  'selection':'highest validation ACC, earliest epoch; display seed chosen by validation ACC, lowest seed tie',
                  'retention':'user requests final Markdown reports and figures only; temporary originals removed after verification',
                  'command':subprocess.list2cmdline([sys.executable,'-m','Q_GNCDM.experiments.run_student_cl_all_datasets','--dataset',name]),
                  'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
                  'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  'protocol':'R0 reference ACC selector; R1 CL off; R2 shared by control and noise studies; R3-R5 change noise only'}
        prior.legacy.write_json(folder/'manifest.json',manifest)
        checks={}
        if not (folder/'preflight.json').exists():
            for mode in ('A0','A1'):
                for condition in noise.CONFIGS:
                    cfg=noise.config(vars(args),condition)
                    checks[mode+'_'+condition]=noise.preflight(data,cfg,mode)
            prior.legacy.write_json(folder/'preflight.json',checks)
        else:
            checks=json.loads((folder/'preflight.json').read_text(encoding='utf-8'))
            assert len(checks)==8
        print(f'{name}: all 8 preflights passed',flush=True)
        records=[];parity=[];parameters=[]
        for seed in (2028,):
            for mode in ('A0','A1'):
                for condition in ('R0','R1','R2','R3','R4','R5'):
                    cfg=noise.config(vars(args),'R2' if condition in ('R0','R1') else condition)
                    cfg.lambda_cl=0. if condition in ('R0','R1') else .1
                    out=folder/mode/f'seed_{seed}'/condition
                    if (out/'verified.json').exists():
                        result=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
                    else:
                        if out.exists():
                            raise RuntimeError(f'Incomplete run retained for review: {out}')
                        out.parent.mkdir(parents=True,exist_ok=True)
                        probes=[];noise_sums={}
                        def observed_branch(model,clean_np,anchor,a,s,e,t):
                            loss,st=noise.branch(model,clean_np,anchor,a,s,e,t)
                            count=int((clean_np!=0).sum())
                            for prefix in ('positive','negative'):
                                z=noise_sums.setdefault(prefix,{'count':0,'abs_sum':0.,'sq_sum':0.,'selected':0.,'changed':0.,'clipped':0.,'max':0.})
                                z['count']+=count;z['abs_sum']+=count*st[prefix+'_input_mae_observed']
                                z['sq_sum']+=count*st[prefix+'_input_mse_observed']
                                z['selected']+=count*st[prefix+'_selected_fraction'];z['changed']+=count*st[prefix+'_changed_fraction']
                                z['clipped']+=count*st[prefix+'_clipped_fraction_observed']
                                z['max']=max(z['max'],st[prefix+'_input_max_change'])
                            return loss,st
                        ids=np.flatnonzero((data['train']!=0).sum(1)>=2)[:128]
                        ep=prior.make_student_episode(data['train'],ids,cfg.query_fraction,np.random.default_rng([seed,1,0,1]))
                        clean=ep.support[ids].copy()
                        if cfg.lambda_cl:
                            model=prior.build_model(data,cfg,mode,seed)
                            probes=[{'epoch':0,**noise.probe(model,clean,cfg,seed)}]
                        def observed_eval(model,evidence,rows):
                            result=prior.core.evaluate(model,evidence,rows)
                            probes.append({'epoch':len(probes),**noise.probe(model,clean,cfg,seed)})
                            return result
                        log=out.parent/(condition+'_console.log')
                        with log.open('w',encoding='utf-8') as handle,contextlib.redirect_stdout(handle):
                            if condition=='R0':
                                result=prior.reference.train_one(data,cfg,mode,seed,out)
                            elif condition=='R1':
                                with patch.object(prior,'contrastive_branch',side_effect=AssertionError('CL called with zero coefficient')):
                                    result=prior.controlled_train_one(data,cfg,mode,seed,out)
                            else:
                                with patch.object(prior,'contrastive_branch',observed_branch),patch.object(prior,'evaluate',observed_eval):
                                    result=prior.controlled_train_one(data,cfg,mode,seed,out)
                        log.rename(out/'console.log')
                        result.update({'condition':condition,'lambda_cl':cfg.lambda_cl,'report_dataset':name})
                        if cfg.lambda_cl:
                            assert len(probes)==result['epochs_run']+3
                            prior.save_csv(out/'fixed_probe_curve.csv',probes[:result['epochs_run']+1])
                            result['input_actual']={p:{'mae_observed':z['abs_sum']/z['count'],'mse_observed':z['sq_sum']/z['count'],
                                'selected_fraction':z['selected']/z['count'],'changed_fraction':z['changed']/z['count'],
                                'clipped_fraction_selected':z['clipped']/max(z['selected'],1),'max_actual_change':z['max']} for p,z in noise_sums.items()}
                        prior.legacy.write_json(out/'metrics.json',result)
                        model=prior.build_model(data,cfg,mode,seed)
                        model.load_state_dict(prior.checkpoint_state(out))
                        scores,_,_=prior.core.evaluate(model,data['train'],data['rows'][2])
                        assert all(scores[k]==result['test_'+k] for k in scores)
                        with (out/'train_log.csv').open(encoding='utf-8-sig') as f: history=list(csv.DictReader(f))
                        assert int(max(history,key=lambda r:float(r['valid_acc']))['epoch'])==result['best_epoch']
                        if condition=='R1':
                            old=prior.checkpoint_state(out.parent/'R0');new=prior.checkpoint_state(out)
                            assert all(torch.equal(v,new[k]) for k,v in old.items())
                            with (out.parent/'R0/train_log.csv').open(encoding='utf-8-sig') as f: original=list(csv.DictReader(f))
                            assert len(original)==len(history)
                            assert all(all(a[k]==b[k] for k in a) for a,b in zip(original,history))
                        prior.legacy.write_json(out/'verified.json',{'checkpoint_replay':True,'ACC_selector':True,'zero_parity':condition=='R1'})
                    if condition=='R1':parity.append({'mode':mode,'seed':seed,'all_epoch_history_and_checkpoint_exact':True})
                    if condition=='R2':
                        base=prior.checkpoint_state(out.parent/'R1');treatment=prior.checkpoint_state(out)
                        for k,v in base.items():
                            if not (k.endswith(('.weight','.bias')) or k=='q_logits'):continue
                            delta=(treatment[k]-v).double()
                            parameters.append({'mode':mode,'seed':seed,'parameter':k,'shape':list(v.shape),
                                'difference_l2':float(delta.norm()),'difference_max':float(delta.abs().max()),
                                'baseline_mean':float(v.float().mean()),'cl_mean':float(treatment[k].float().mean()),
                                'effective_difference_l2':float((treatment[k].abs()-v.abs()).double().norm()),
                                'baseline_preview':v[:8,:8].tolist() if seed==2026 and k.startswith('f_nn.') and v.ndim==2 else None,
                                'cl_preview':treatment[k][:8,:8].tolist() if seed==2026 and k.startswith('f_nn.') and v.ndim==2 else None})
                    records.append({k:v for k,v in result.items() if not isinstance(v,dict)})
                    prior.save_csv(folder/'summary.csv',records)
                    prior.legacy.write_json(folder/'parity.json',parity)
                    prior.legacy.write_json(folder/'parameters.json',parameters)
                    print(f'{name} {mode} {seed} {condition} ACC={result["test_acc"]:.8f} AUC={result["test_auc"]:.8f} ({len(records)}/12)',flush=True)
                    if condition=='R5':
                        publish(name,mode,seed,folder)
        for path,digest in sources.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
        for path,digest in data['source_sha256'].items():
            source=Path(path) if Path(path).is_absolute() else Path(args.data_dir)/path
            assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        prior.legacy.write_json(folder/'completed.json',{'runs':12,'preflight_settings':8,'baseline_exact_pairs':2,
                               'checkpoint_replays':12,'protected_code_and_data_unchanged':True,'status':'passed'})
    except BaseException:
        (folder/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['assist2012','junyi2015','eedi_task34'],required=True)
    worker(parser.parse_args().dataset)
