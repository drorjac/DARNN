import copy
import numpy as np
import pytest
import torch
from recmo.data import synthetic, split_windows
from recmo.forecast import Forecaster, WindowLoss, next_context

@pytest.mark.parametrize('kind',['mlp','lstm','gru','darnn'])
@pytest.mark.parametrize('context',['rolling','expanding','separate'])
def test_partial_shapes_and_checkpoint(kind,context):
    model=Forecaster(kind,6,11,window=4,context=context,width=4)
    x=torch.randn(2,6,1)
    prediction,blocks=model(x,True)
    assert prediction.shape==(2,11,1)
    assert [b.shape[1] for b in blocks]==[4,4,3]
    clone=copy.deepcopy(model)
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone(x),prediction)
    blocks[0].retain_grad()
    blocks[-1].square().mean().backward()
    assert blocks[0].grad is not None
    assert blocks[0].grad.abs().sum()>0

@pytest.mark.parametrize('strategy',['direct','recursive','recmo','dirmo'])
def test_strategies(strategy):
    assert Forecaster('lstm',6,11,strategy,4,width=4)(torch.randn(2,6,1)).shape==(2,11,1)

def test_context():
    x=torch.arange(6.).reshape(1,6,1)
    b=torch.tensor([6.,7.]).reshape(1,2,1)
    assert next_context(x,b,6,'rolling').flatten().tolist()==[2,3,4,5,6,7]
    assert next_context(x,b,6,'expanding').shape[1]==8

def test_no_leakage_and_reproducibility():
    series=np.arange(300,dtype=float)
    splits,meta=split_windows(series,12,24)
    changed=series.copy(); changed[180:]+=10000
    other,new_meta=split_windows(changed,12,24)
    assert meta['mean']==new_meta['mean'] and meta['scale']==new_meta['scale']
    torch.testing.assert_close(splits['train'][0],other['train'][0])
    assert max(meta['origins']['train'])+24<=180
    assert min(meta['origins']['val'])>=180 and max(meta['origins']['val'])+24<=240
    assert meta==split_windows(series,12,24)[1]

@pytest.mark.parametrize('mode',['uniform','increasing','decreasing','softmax','uncertainty'])
def test_weights(mode):
    loss=WindowLoss(3,mode)
    assert (loss.weights()>0).all()
    if mode!='uncertainty':
        torch.testing.assert_close(loss.weights().sum(),torch.tensor(1.))
    target=torch.randn(2,11,1)
    blocks=[torch.randn(2,n,1,requires_grad=True) for n in [4,4,3]]
    loss(blocks,target).backward()
    assert all(b.grad is not None for b in blocks)

def test_synthetic_seed():
    for kind in ['ar','sine','trend','nonlinear','damped','regime','periodic']:
        np.testing.assert_array_equal(synthetic(kind,300),synthetic(kind,300))

def test_end_to_end_fit_selects_validation_checkpoint():
    import time
    from recmo.experiment import neural_fit
    splits,_=split_windows(synthetic('sine',300),6,11,train_limit=16,eval_limit=8)
    model=Forecaster('lstm',6,11,window=4,width=4)
    loss=WindowLoss(3)
    log=neural_fit(model,loss,splits,{'updates':2,'batch_size':4,'validate_every':1},7,time.monotonic()+30)
    with torch.no_grad():
        x,y=splits['val']
        restored_score=float((model(x)-y).square().mean())
    assert restored_score==pytest.approx(min(r['validation_mse'] for r in log))

def test_direct_equals_single_block_recmo():
    a=Forecaster('gru',6,11,'direct',window=4,width=4)
    b=Forecaster('gru',6,11,'recmo',window=11,width=4)
    b.load_state_dict(a.state_dict())
    x=torch.randn(2,6,1)
    torch.testing.assert_close(a(x),b(x))

@pytest.mark.parametrize('context',['rolling','expanding','separate'])
def test_intervention_affects_only_later_windows(context):
    model=Forecaster('gru',6,11,window=4,context=context,width=4)
    x=torch.randn(2,6,1)
    base=model(x)
    changed=model(x,intervention=(0,'bias',0.5))
    torch.testing.assert_close(base[:,:4],changed[:,:4])
    assert not torch.allclose(base[:,4:],changed[:,4:])
