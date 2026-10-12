import sys,pathlib,json,copy
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from algorithms import *
checks=[]
def check(name,condition):
 assert condition,name
 checks.append(name)
game=Onitama();state=game.init_game();before=copy.deepcopy(state)
a=Minimax(game,True,3);b=AlphaBeta(game,True,3)
check('minimax and alpha-beta agree at depth 3',a.minimax(state,3,True)==b._alphabeta(state,3,-math.inf,math.inf,True))
check('search preserves input',state==before)
blocked=copy.deepcopy(state);blocked['my_card']='Your Custom Card'
check('blocked agent returns pass',Minimax(game,True,2).make_move(blocked) is None)
p=game.pass_turn(blocked)
check('pass rotates card and preserves board',p['turn']==-1 and p['my_card']==blocked['center'] and p['board']==blocked['board'])
won=copy.deepcopy(state);won['board']=[['.' if c=='x' else c for c in row] for row in state['board']]
check('terminal win has dominant score',a.minimax(won,3,True)[0]==10000)
env=ScotlandYard(move_probs=[.7,.8,.9],win_reward=50,living_reward=-10,moving_rewards=[1,2,3])
check('all transition rows sum to one',np.allclose(env.T.sum(axis=2),1))
V,policy=value_iteration(env)
Q=np.sum(env.T*(env.R+.95*V[None,None,:]),axis=2)
check('Bellman residual within tolerance',np.max(np.abs(V-Q.max(axis=1)))<1e-6)
terminal=[i for i,(p,x) in enumerate(env.states) if p==x]
check('capture states have zero continuation value',np.allclose(V[terminal],0))
random.seed(SEED);T_hat,R_hat=validate(env)
mask=T_hat>0
nonterm=np.array([p!=x for p,x in env.states])
transition_rmse=float(np.sqrt(np.mean((T_hat[nonterm]-env.T[nonterm])**2)))
check('sampled rewards agree with analytic rewards',np.allclose(R_hat[mask],env.R[mask]))
check('sampled transition RMSE below 0.01',transition_rmse<.01)
# One-step environments test terminal and truncated updates independently.
class OneStep:
 observation_space=spaces.Box(np.full(4,-1,dtype=np.float32),np.full(4,1,dtype=np.float32))
 action_space=spaces.Discrete(2)
 def __init__(self,terminal):self.terminal=terminal
 def reset(self,seed=None):return np.zeros(4),{}
 def step(self,action):return np.zeros(4),1.,self.terminal,not self.terminal,{}
for terminal in [True,False]:
 e=OneStep(terminal);agent=QLearningAgent(e,(2,2,2,2),.9,1.,1.,1.,0.,0.,1.)
 s=agent.obs_to_state(np.zeros(4));agent.Q_table[s,0]=10.
 agent.train(1)
 check('terminal update' if terminal else 'truncation ends episode and bootstraps',np.isclose(agent.Q_table[s,0],1. if terminal else 10.))
results={'checks_passed':len(checks),'checks':checks,'transition_rmse':transition_rmse,'transition_frobenius_error':float(norm((T_hat-env.T)[nonterm])),'observed_reward_error':float(norm((R_hat-env.R)[mask])),'bellman_residual':float(np.max(np.abs(V-Q.max(axis=1))))}
(root/'results').mkdir(exist_ok=True)
(root/'results'/'verification.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
