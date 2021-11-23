import itertools
import copy
import numpy as np
import gurobipy
import matplotlib.pyplot as plt
import networkx as nx
from gurobipy import GRB
from itertools import combinations,product
import json
import pandas as pd

#[[1,0],[5,5],[8,8]],
#job_list = [[[0,0],[4,7],[2,8],[0,8]],[[1,0],[0,8]],[[0,0],[7,4],[8,8]],[[3,0],[0,2],[1,5]],[[0,0],[3,8],[0,8]],[[7,0],[4,6]],[[2,0],[2,3],[5,8]],[[6,0],[2,3],[6,1]]]*2
#job_list=[[[0,0],[2,3],[6,1]],[[1,6],[8,8]],[[5,6],[0,8]],[[4,5],[0,1],[8,0]]]

WAREHOUSE_DIM = 10  # size of warehouse
N_FORKLIFTS = 5  #number of forklifts available
RECEIVING = [0, 0]  # location of receiving
SHIPPING = [WAREHOUSE_DIM-1, WAREHOUSE_DIM-1]  # location of shipping
LAB = [0, WAREHOUSE_DIM-1]  # location of lab
N_JOBS = 16 # number of jobs


############################################# Random Jobs ##############################################

def generate_random_job(warehouse_dim = WAREHOUSE_DIM, 
                        receiving = RECEIVING, 
                        shipping = SHIPPING, 
                        lab = LAB):
    job_length = 1 + np.random.randint(3) #1 + np.random.randint(3) # number of tasks
    job = np.random.randint(warehouse_dim, size=job_length*2)
    for i in range(0,job_length-1):
        while (job[2*i]==RECEIVING[0] and job[2*i+1]==RECEIVING[1]) or (job[2*i]==SHIPPING[0] and job[2*i+1]==SHIPPING[1]) or (job[2*i]==LAB[0] and job[2*i+1]==LAB[1]):
            job = np.random.randint(warehouse_dim, size=job_length*2)
    destination = [receiving, shipping, lab][np.random.choice([0,1,2])]
    if destination == receiving:
        job = np.insert(job, 0, destination)
    else:
        job = np.append(job, destination)
    job = job.reshape(job_length + 1, 2)
    return job


############################################# Order task ###############################################


def order_tasks(job_list, WAREHOUSE_DIM):
    #import copy
    """
    This orders the tasks within a certain job to minimize distance
    This does not take into account the original location of the forklift
    NOTE: distance \neq time since time had an error
    """
    
    for i in range(0,len(job_list)): #optimize each job
        job = copy.copy(job_list[i]) #consider a single job
        N_TASKS = len(job)-1 #number of tasks
        
        distance = WAREHOUSE_DIM*len(job)+1 #make this large, this is what we need to beat
        if all(job[0] == [0,0]): #change this to RECEIVING
            for perm in itertools.permutations(job[1:],N_TASKS): #potential permutations
                this_dist = 0 #the distance for this permutation
                for j in range(1,N_TASKS-1):
                    this_dist += abs(perm[j][0]-perm[j+1][0]) + abs(perm[j][1]-perm[j+1][1]) #add distance between tasks
                this_dist += abs(perm[0][0] - job[0][0]) + abs(perm[0][1] - job[0][1]) #add distance to delivery point
                
                if this_dist < distance: #better permutation
                    distance = this_dist
                    job_list[i][1:N_TASKS+1] = list(perm)
        else:
            for perm in itertools.permutations(job[:N_TASKS],N_TASKS): #potential permutations
                this_dist = 0 #the distance for this permutation
                for j in range(0,N_TASKS-1):
                    this_dist += abs(perm[j][0]-perm[j+1][0]) + abs(perm[j][1]-perm[j+1][1]) #add distance between tasks
                this_dist += abs(perm[N_TASKS-1][0] - job[N_TASKS][0]) + abs(perm[N_TASKS-1][1] - job[N_TASKS][1]) #add distance to delivery point
                    
                if this_dist < distance: #better permutation
                    distance = this_dist
                    job_list[i][0:N_TASKS] = list(perm)
    return job_list

############################################# Removing cycle: callback function ########################

def subtourelim(model, where): # cycle elimination
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._vars)
        selected = gurobipy.tuplelist((k, i, j) for k, i, j in model._vars.keys() if vals[k, i, j] >0.5)
                        
        Edges=[]
        for k,i,j in selected.select('*','*','*'):
            Edges.append([i,j])
        #print(selected.select('*','*'))
            
        #print(Edges)
        G=nx.DiGraph()
        G.add_edges_from(Edges)
        Strongly_connected_comp=sorted(nx.strongly_connected_components(G),key=len)
        N_components = len(sorted(nx.weakly_connected_components(G),key=len))
        #print(sorted(nx.strongly_connected_components(G),key=len))
        #print(N_components)
        if N_components > N_FORKLIFTS:
            for cmp in range(0,len(Strongly_connected_comp)):
                if len(Strongly_connected_comp[cmp])>1:
                    model.cbLazy(gurobipy.quicksum(model._vars[k,m, l] for k in range(N_FORKLIFTS) for m in list(Strongly_connected_comp[cmp])
                                                   for l in list(Strongly_connected_comp[cmp]))
                                 <= len(Strongly_connected_comp[cmp])-1)

###########################################################################################################
def total_distance_minimization(job_list, N_FORKLIFTS): # Optimization function
    
    N_JOBS = len(job_list)
    
    s=[job_list[i][0][0]+job_list[i][0][1] for i in range(N_JOBS)]
    
    d=[0 for i in range(N_JOBS)]
    
    
    for i in range(N_JOBS):
        job_dist=0
        for j in range(0, len(job_list[i])-1):
            job_dist+=abs(job_list[i][j][0]-job_list[i][j+1][0]) + abs(job_list[i][j][1]-job_list[i][j+1][1])
        d[i]=job_dist
        
    
    M=[[0 for i in range(N_JOBS)] for j in range(N_JOBS)]
    
    for i in range(N_JOBS):
        for j in range(N_JOBS):
            if i!=j:
                M[i][j]=abs(job_list[i][-1][0]-job_list[j][0][0])+abs(job_list[i][-1][1]-job_list[j][0][1])
    
    p=[0 for i in range(N_JOBS)]        
    for i in range(N_JOBS):
        pickup_time=0  
        for j in range(0, len(job_list[i])):
            pickup_time+=np.random.uniform(low=3.0,high=20.0)
        p[i]=pickup_time
    
    ######################################## Optimization #################################################                   
                           
    
    mymodel = gurobipy.Model()
    
    forklift_job = {(k, i, j): M[i][j]+d[j]+p[j] for k in range(N_FORKLIFTS) for i in range(N_JOBS) for j in range(N_JOBS)}
    forklift_initial={(k,i): s[i]+d[i]+p[i] for k in range(N_FORKLIFTS) for i in range(N_JOBS)}
    
    var1=mymodel.addVars(forklift_job.keys(), vtype=GRB.BINARY, name='var1')
    var2=mymodel.addVars(forklift_initial.keys(), vtype=GRB.BINARY, name='var2')
    
    
    mymodel.addConstrs(var1.sum('*',i, i) == 0 for i in range(N_JOBS) for k in range(N_FORKLIFTS))
    mymodel.addConstrs(var1.sum('*',i, '*') <= 1 for i in range(N_JOBS))
    mymodel.addConstrs(var1.sum('*','*',i) <= 1 for i in range(N_JOBS))
    mymodel.addConstrs(var1.sum('*','*',i)+var1.sum('*',i, '*') <= 2 for i in range(N_JOBS))
    mymodel.addConstrs(var1.sum('*','*',i)+var1.sum('*',i, '*') >= 1 for i in range(N_JOBS))
    mymodel.addConstr(var1.sum('*', '*','*') == N_JOBS-N_FORKLIFTS )
    mymodel.addConstrs(var1.sum('*','*',j)+var2.sum('*',j) == 1 for j in range(N_JOBS))
    mymodel.addConstrs(var1.sum(k,'*','*') >= np.floor(N_JOBS/N_FORKLIFTS)-1 for k in range(N_FORKLIFTS))
    mymodel.addConstrs(var1.sum(k,'*','*') <= np.ceil(N_JOBS/N_FORKLIFTS)-1 for k in range(N_FORKLIFTS))
    mymodel.addConstrs(var1.sum('*','*',i)-var1.sum(k,'*',i) <= 1-var1[k,i,j] for i in range(N_JOBS) for j in range(N_JOBS) for k in range(N_FORKLIFTS))
    mymodel.addConstrs(var1.sum('*',j,'*')-var1.sum(k,j,'*') <= 1-var1[k,i,j] for i in range(N_JOBS) for j in range(N_JOBS) for k in range(N_FORKLIFTS))

    
    
    mymodel.setObjective(gurobipy.quicksum((M[i][j]+d[j]+p[j])*var1[k,i,j] for j in range(0,N_JOBS) for i in range(0,N_JOBS) for k in range(N_FORKLIFTS))
                         +gurobipy.quicksum((s[l]+d[l]+p[l])*var2[k,l] for l in range(N_JOBS) for k in range(N_FORKLIFTS)), 
                         sense=gurobipy.GRB.MINIMIZE)
    
    
   # vars={(i,j): var1.sum(['*',i,j]) for i in range(N_JOBS) for j in range(N_JOBS)}

    
    mymodel._vars = var1
    mymodel.Params.lazyConstraints = 1
    mymodel.optimize(subtourelim)
    
    
    ############################################### Print and Plot #####################################
    Edges=[]
    
    TotalDistance=0
    Adj_matrix=[[0 for i in range(N_JOBS)] for j in range(N_JOBS)]
    
    for k in range(0,N_FORKLIFTS):
        for i in range(0,N_JOBS):
            for j in range(0,N_JOBS):
                if (var1[k,i,j].getAttr("X")) > 0.5:
                    Edges.append([i,j])
                    TotalDistance+=M[i][j]+d[j]+p[j]
                    Adj_matrix[i][j]=1
                    #print(var1[k,i,j].getAttr("X"))
                    #print([k,i,j])
    #print((Edges))
                    
    for k in range(0,N_FORKLIFTS):                       
        for i in range(N_JOBS):
            if (var2[k,i].getAttr("X")) != 0:
                TotalDistance+=s[i]+d[i]+p[i]
    #print(Edges)
    
   # print('Total distance is :', TotalDistance)
    #print('Adjacency matrix:', Adj_matrix)
    #print('Distance from [0,0]:', s)
    #print('Distance from i to j:', M)
    #print('Travel distance for job i:', d)
    
    G=nx.DiGraph()
    G.add_edges_from(Edges)
    nx.draw_circular(G, with_labels=True)
    
    
    roots = (v for v, d in G.in_degree() if d == 0)
    leaves = [v for v, d in G.out_degree() if d == 0]
    all_paths = []
    for root in roots:
        paths = nx.all_simple_paths(G, root, leaves)
        all_paths.extend(paths)

    return all_paths

#########################################################################################

raw_job_list = [generate_random_job(warehouse_dim = WAREHOUSE_DIM) for k in range(N_JOBS)]

job_list_new=order_tasks(raw_job_list, WAREHOUSE_DIM) # job list after sorting the task

job_order=total_distance_minimization(job_list_new, N_FORKLIFTS) # set of jobs for each forklift    

print(job_order)

job_number_each=[0 for i in range(N_FORKLIFTS)]
        
for i in range(N_FORKLIFTS):
    job_number_each[i]=len(job_order[i])
print(job_number_each)
