import numpy as np
from dgl.data import DGLDataset
import dgl

class Dataset(DGLDataset):
    def __init__(self, dglist, problems):
        super().__init__(name='dataset')
        self.dglist = dglist
        self.problems = problems
    
    def __len__(self):
        return len(self.dglist)
    
    def __getitem__(self, idx):
        graph = self.dglist[idx]
        problem = self.problems[idx]
        return graph, idx
    

def validate(model, val_loader, dataset):
    val_objvalue = []
    for batch, idx in val_loader:
        batch_objvalue = []
        for i, graph in enumerate(dgl.unbatch(batch)):
            problem = dataset[45+idx[i]]
            with graph.local_scope():
                pred_gamma = model.predict(graph, problem)
                pred_gamma *= np.broadcast_to(problem['operation'][:, :, np.newaxis], pred_gamma.shape) 
            batch_objvalue.append(
                objvalue(problem, pred_gamma, construct_delta(problem, pred_gamma))
            )

        val_objvalue.append(sum(batch_objvalue) / len(batch_objvalue))
    return sum(val_objvalue)/len(val_objvalue)


def objvalue(problem, gamma, delta):
    time_cost = problem['time_cost'] 
    op_cost = problem['op_cost'] 
    productivity = problem['productivity'] 
    transportation_cost = problem['transportation_cost'] 
    dist = problem['dist'] 

    total_op_cost = np.sum(
        (time_cost * op_cost / productivity[None, :])[:, None, :] * gamma
    )
    total_logistic_cost = np.sum(
        (transportation_cost[:, None, None] * dist[None, ...])[..., None, None] * delta
    )
    return total_op_cost + total_logistic_cost

def construct_delta(problem, gamma):
    n_cities = problem['n_cities']
    n_operations = problem['n_operations']
    n_tasks = problem['n_tasks']

    delta = np.zeros((1, n_cities, n_cities, n_operations - 1, n_tasks))
    for t in range(n_tasks):
        o_iter, c_iter = np.where(gamma[:, t] == 1)
        for i in range(len(o_iter)-1):
            o = o_iter[i]
            c_u, c_v = c_iter[i], c_iter[i+1]
            delta[0, c_u, c_v, o, t] = 1
    return delta
