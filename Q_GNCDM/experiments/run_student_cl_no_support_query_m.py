"""Record-wise mastery-only InfoNCE. Item inputs remain clean.

Reuse the record-wise training loop in this isolated process. No changes to
the original runner on disk or any concurrently running interaction experiment.
"""
import numpy as np
import torch
from Q_GNCDM.experiments import run_student_cl_no_support_query as base

ID='student_cl_no_support_query_M_20261124'
WORK=base.ROOT/'Q_GNCDM/scratch'/ID
REPORT=base.ROOT/'Q_GNCDM/docs/experiments/student_cl_no_support_query_M'
zero_forward=base.forward_loss


def student_view(data,rows,student,cfg,seed,epoch,step,view):
    ratio,scale=cfg[:2] if view==0 else cfg[2:]
    key=np.random.SeedSequence([1123,seed,epoch,step,view]).generate_state(1,dtype=np.uint64)[0]
    n=data['train'].shape[1]
    coordinates=rows[:,0,None].astype(np.uint64)*np.uint64(n)+np.arange(n,dtype=np.uint64)[None,:]
    return base.noise_view(student,coordinates,ratio,scale,key)


def forward_loss(model,data,rows,coefficient,cfg,seed,epoch,step,reference=False):
    if coefficient==0:
        return zero_forward(model,data,rows,0,cfg,seed,epoch,step,reference)
    u,v=base.inputs(data,rows)
    theta=model.diagnose_theta(torch.from_numpy(u))
    psi=model.diagnose_psi(torch.from_numpy(v))
    prediction=model.predict_response(theta,psi,model.get_q_soft()[torch.tensor(rows[:,1])])
    target=torch.tensor(rows[:,2],dtype=torch.float32).reshape(-1,1)
    losses=model.compute_loss(prediction,target);original=losses['loss_total']
    stats={'loss_original':float(original.detach()),'loss_prediction':float(losses['loss_rec'].detach())}
    views=[]
    for view in (0,1):
        noisy,s=student_view(data,rows,u,cfg,seed,epoch,step,view)
        views.append(model.diagnose_theta(torch.from_numpy(noisy)))
        prefix='positive_' if view==0 else 'negative_'
        stats.update({prefix+'student_'+k:x for k,x in s.items()})
        # No item augmentation is executed; keep explicit zero diagnostic fields.
        stats.update({prefix+'item_'+k:0. for k in s})
    cl,diagnostics=base.same_student_infonce(theta,*views,.2)
    total=original+coefficient*cl
    stats.update(diagnostics)
    stats.update(loss_cl=float(cl.detach()),loss_total=float(total.detach()))
    return total,original,cl,stats,prediction


def grad_norm(grads):
    # Disconnected item/aggregation branches legitimately have no CL gradients.
    return sum(float(g.detach().double().square().sum()) for g in grads if g is not None)**.5


# Rebind only this process's imported module, retaining the exact common loop.
base.ID=ID;base.WORK=WORK;base.REPORT=REPORT;base.forward_loss=forward_loss
base.grad_norm=grad_norm


def __getattr__(name):return getattr(base,name)


if __name__=='__main__':base.main()
