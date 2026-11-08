"""Finite two-stage stochastic program. Costs must share units/objective.
No optimality claim beyond the supplied finite set of candidate programs.
"""
from itertools import product
import numpy as np


def risk(loss, weights, alpha=.9, tail_weight=.25):
    """Mean/CVaR mixture; weighted discrete CVaR with fractional tail atom."""
    x, p = np.asarray(loss, float), np.asarray(weights, float)
    if x.ndim != 1 or p.shape != x.shape or not len(x):
        raise ValueError('loss and weights must be nonempty vectors')
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('strictly positive finite weights required')
    if not 0 <= alpha < 1 or not 0 <= tail_weight <= 1:
        raise ValueError('invalid risk settings')
    if not np.isfinite(x).all():
        return dict(mean=float('inf'), cvar=float('inf'), objective=float('inf'))
    p = p / p.sum()
    remaining, total = 1-alpha, 0.
    for i in np.argsort(x)[::-1]:
        mass = min(remaining, p[i]); total += mass*x[i]; remaining -= mass
        if remaining <= 1e-14: break
    mean, cvar = float(p@x), float(total/(1-alpha))
    return dict(mean=mean, cvar=cvar, objective=(1-tail_weight)*mean+tail_weight*cvar)


def solve(costs, weights, observations, alpha=.9, tail_weight=.25, max_policies=100000):
    """costs[night, scenario, morning_action].

    A common night program is chosen now. One morning action per observation
    group is chosen for 10:00. observations[night,scenario] MUST encode only
    information actually available then, never a hidden ensemble-member ID.
    Groups conservatively identical (all zero) are valid if no observation
    model has been calibrated. This is a precommitted two-stage risk objective.
    Invalid land/zone trajectories must have +inf cost.
    """
    c=np.asarray(costs,float); o=np.asarray(observations)
    if c.ndim != 3 or any(n==0 for n in c.shape) or o.shape != c.shape[:2]:
        raise ValueError('bad cost/observation dimensions')
    if len(weights)!=c.shape[1]: raise ValueError('bad weights')
    risk(np.zeros(c.shape[1]),weights,alpha,tail_weight)
    results=[]
    for night in range(c.shape[0]):
        groups, inverse=np.unique(o[night],return_inverse=True)
        count=c.shape[2]**len(groups)
        if count>max_policies: raise ValueError('too many contingent policies; merge observation groups')
        best=None
        for actions in product(range(c.shape[2]),repeat=len(groups)):
            indices=np.asarray(actions)[inverse]
            losses=c[night,np.arange(c.shape[1]),indices]
            metrics=risk(losses,weights,alpha,tail_weight)
            if best is None or metrics['objective']<best['objective']:
                best=dict(night=night,**metrics,actions_by_observation={str(g):int(a) for g,a in zip(groups,actions)},losses=losses.tolist())
        results.append(best)
    valid=[r for r in results if np.isfinite(r['objective'])]
    return dict(best=min(valid,key=lambda r:r['objective']) if valid else None,
                alternatives=results, scope='finite candidate set; no guarantee of race victory')
