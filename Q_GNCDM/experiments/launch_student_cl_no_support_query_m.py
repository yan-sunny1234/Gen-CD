"""Run all preflights, then paired record-wise experiments in isolated processes."""
import argparse
import hashlib
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from Q_GNCDM.experiments import run_student_cl_no_support_query_m as r
from Q_GNCDM.experiments import report_student_cl_no_support_query_m as reports

lock=threading.Lock()


def group(name,mode,seed):
    folder=r.WORK/name/mode/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    for condition in ['R0','R1',*r.CONFIGS]:
        if (folder/condition/'complete.json').exists():continue
        cmd=[sys.executable,'-m','Q_GNCDM.experiments.run_student_cl_no_support_query_m','--dataset',name,'--mode',mode,'--seed',str(seed),'--condition',condition]
        with (folder/(condition+'_run.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(cmd,cwd=r.ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} {mode} {seed} {condition} failed; see {folder}')
        print('DONE',name,mode,seed,condition,flush=True)
    with lock:
        reports.report(name);reports.index()
    return name,mode,seed


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--datasets',nargs='+',choices=r.DATASETS,default=r.DATASETS);a=p.parse_args()
    r.WORK.mkdir(parents=True,exist_ok=True)
    sources=[r.ROOT/'Q_GNCDM/models/weak_q_gncdm.py',r.ROOT/'GNCDM/model.py',r.ROOT/'GNCDM/train.py',
             *[r.ROOT/'Q_GNCDM/experiments'/f for f in ['run_student_cl_no_support_query.py','run_student_cl_no_support_query_m.py','verify_student_cl_no_support_query_m.py','run_simple_contrastive.py','controlled_student_cl.py']]]
    hashes={str(p.relative_to(r.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    r.write(r.WORK/'manifest.json',{'id':r.ID,'seeds':[2026,2027,2028],'datasets':a.datasets,'conditions':['R0','R1',*r.CONFIGS],
                                 'source_hashes':hashes,'selection':'validation ACC, earliest tie','batch_records':128,'cl_object':'student mastery m only; clean item input; no support/query'})
    # All checks pass before formal training on any dataset.
    for name in a.datasets:
        cmd=[sys.executable,'-m','Q_GNCDM.experiments.verify_student_cl_no_support_query_m','--dataset',name]
        result=subprocess.run(cmd,cwd=r.ROOT)
        if result.returncode:raise RuntimeError('Preflight failure: '+name)
        data,args=r.load(name)
        r.write(r.WORK/name/'manifest.json',{'args':vars(args),'source_hashes':hashes,'data_hashes':data['source_sha256'],
                                          'noise':r.CONFIGS,'train_shape':list(data['train'].shape),'split_lengths':[len(x) for x in data['rows']]})
    reports.index()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(group,name,mode,seed) for name in a.datasets for seed in [2026,2027,2028] for mode in ['A0','A1']]
        for job in as_completed(jobs):print('GROUP COMPLETE',job.result(),flush=True)
    for p,h in hashes.items():assert hashlib.sha256((r.ROOT/p).read_bytes()).hexdigest()==h,p
    r.write(r.WORK/'completed.json',{'status':'passed','runs':36*len(a.datasets),'sources_unchanged':True})
    print('ALL DATASETS COMPLETE',flush=True)


if __name__=='__main__':main()
