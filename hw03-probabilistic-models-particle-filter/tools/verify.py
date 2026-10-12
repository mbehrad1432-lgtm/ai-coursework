"""Check exact inference against joint enumeration and simulation invariants."""
from pathlib import Path
import sys,json,itertools,copy,random
import numpy as np
import networkx as nx
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from algorithms import *
checks=[]
def check(name,ok):
    assert ok,name
    checks.append(name)
check('CPDs are valid',model.check_model())
check('collider closed without evidence',not model.is_dconnected('Difficulty','Intelligence'))
check('collider opened by descendant evidence',model.is_dconnected('Difficulty','Intelligence',observed=['Letter']))
variables=list(model.nodes());cards=model.get_cardinality();joint=[]
for values in itertools.product(*(range(cards[v]) for v in variables)):
    assignment=dict(zip(variables,values));prob=1.
    for variable in variables:
        cpd=model.get_cpds(variable);prob*=cpd.get_value(**{v:assignment[v] for v in cpd.variables})
    joint.append((assignment,prob))
check('joint distribution normalizes',np.isclose(sum(p for _,p in joint),1))
grade=np.array([sum(p for s,p in joint if s['Grade']==k) for k in range(3)])
letter_mass=sum(p for s,p in joint if s['Letter']==0)
conditional=np.array([sum(p for s,p in joint if s['Letter']==0 and s['Intelligence']==k)/letter_mass for k in range(2)])
ve=VariableElimination(model)
check('grade inference matches exhaustive enumeration',np.allclose(grade,ve.query(['Grade'],show_progress=False).values))
check('conditional inference matches enumeration',np.allclose(conditional,ve.query(['Intelligence'],evidence={'Letter':0},show_progress=False).values))
order=weighted_min_fill_order(model);graph=induced_graph(model,order)
canonical=lambda g:{frozenset(e) for e in g.edges}
check('induced graph retains original moral edges',canonical(nx.Graph(model.moralize()))<=canonical(graph))
check('induced graph agrees with pgmpy',canonical(graph)==canonical(ve.induced_graph(order)))
world=Maze(maze_layout)
random.seed(42)
check('initial particles occupy free cells',all(world.is_free(*p.xy) for p in Particle.initialize_particles(300,world)))
robot=Robot(world)
check('robot sensor returns scalar',isinstance(robot.sense(world),float))
check('Gaussian weight maximized at matching measurement',gaussian_weight(2,2)==1 and gaussian_weight(2,3)<1)
p=Particle(2.5,.5,heading=0);check('movement rejects outside maze',not p.attempt_move(100,move_check=lambda p,dx,dy:world.is_free(p.x+dx,p.y+dy)))
try:WeightedSampler([Particle(0,0,w=0)])
except ValueError:valid=True
else:valid=False
check('zero-weight sampler fails explicitly',valid)
random.seed(42);states=[Particle(0,0,w=.1),Particle(1,0,w=.9)];sampler=WeightedSampler(states)
proportion=sum(sampler.sample().x==1 for _ in range(10000))/10000
check('weighted sampler tracks mass',abs(proportion-.9)<.02)
h1=run_particle_filter_simulation(steps=8,count=200,seed=123);h2=run_particle_filter_simulation(steps=8,count=200,seed=123)
check('seed reproduces simulation history',h1==h2)
check('simulation errors and ESS are finite',all(np.isfinite(p['position_error']) and 1<=p['effective_sample_size']<=200.000001 for p in h1))
result={'checks_passed':len(checks),'checks':checks,'joint_mass':sum(p for _,p in joint),'weighted_sampler_high_mass_frequency':proportion}
(root/'results').mkdir(exist_ok=True);(root/'results'/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
