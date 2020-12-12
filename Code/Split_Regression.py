from IPython.display import display, clear_output
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import *
from sklearn import linear_model
import matplotlib.pyplot as plt
import ipywidgets as widgets
from random import shuffle
from uuid import uuid4
import pandas as pd
import numpy as np
import pycasso
import pickle
import torch
import sys
import os

np.set_printoptions(suppress=True)


def train_test_split(r, data):
    #takes the data from site r and splits it into a train and test set
    data = data[data['site'] == r]
    train = data.sample(n=int(len(data)*0.7))
    test = data[~data.isin(train)].dropna()
    return train, test


def dist_regression(train, regress):
    '''performs the distributed regression on the train dataset. Regress indicates which method you want to use'''
    
    if regress == 'lasso':
        penalty_string = 'l1'

        def penalty(lams, beta):
            return lams * np.sum(np.abs(beta),1)

        final_regression = distributed_lasso

    if regress == 'mcp':
        penalty_string = 'mcp'

        def penalty(lams, beta):
            #the MCP penalty
            gamma = 2.7
            beta = np.abs(beta)

            mask = (beta <= gamma * lams[:,None])
            
            under = gamma * beta - np.square(beta) / (2 * gamma)
            over = 1/2 * gamma * np.square(lams[:,None])
            
            return np.sum(under * mask + over * (~mask),1)

        final_regression = distributed_mcp

    if regress == 'scad':
        penalty_string = 'scad'

        def penalty(lams, beta):
            #the scad penalty
            gamma = 3.7
            beta = np.abs(beta)

            result = lams[:,None] * beta * (beta <= lams[:,None])
            result += (gamma * lams[:,None] * beta - 0.5 * (np.square(beta) + np.square(lams[:,None])))/(gamma - 1) * (beta > lams[:,None]) * (beta <= gamma * lams[:,None] )
            result += np.square(lams[:,None]) * (gamma**2-1) / (2 * (gamma - 1)) * (beta > gamma * lams[:,None])
            
            return np.sum(result, 1)

        final_regression = distributed_scad

    #get the variables to regress on 
    X = [t[vars].values for t in train]
    Y = [t[outcome].values for t in train]

    #check shapes
    depth, length = X[0].shape
    for x in X:
        assert x.shape[0] == depth
        assert x.shape[1] == length
    for y in Y:
        assert y.shape[0] == depth


    #partition the data into cross validation folds
    partition = np.random.choice(folds, replace=True, size=depth)

    lambdas = [100,0.05]
    cv_loss = np.zeros(100)

    for i in range(folds):
        #rotate through the folds
        train = (partition != i)
        hold = (partition == i)

        for (inp, out) in zip(X,Y):
            #for each site, run picasso regression on all the lambdas
            
            sys.stdout = open(os.devnull, "w")
            solver_mcp = pycasso.Solver(inp[train], out[train], penalty=penalty_string, lambdas = lambdas, useintercept=False)
            solver_mcp.train()
            coefs = solver_mcp.coef()
            lambdas = solver_mcp.lambdas
            sys.stdout = sys.__stdout__

            #calculate the hold out loss for each lambda
            cv_loss += np.mean(np.square(coefs['beta'] @ inp[hold].T - out[hold][None,:]),1) + penalty(lambdas, coefs['beta'])
    cv_loss /= folds * len(X)

    #get the best lambda from the list
    best_lam = lambdas[cv_loss.argmin()]    
    best_loss = cv_loss[cv_loss.argmin()]

    return final_regression(X,Y,best_lam)

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def distributed_lasso(X_data, Y_data, lam):
    #runs the distributed lasso with the lambda value
    def loss_func(inp, out, beta):
        return torch.mean(torch.square(inp @ beta - out)) + lam * torch.sum(torch.abs(beta))

    return distributed_regression(X_data, Y_data, loss_func, 0.1)


def distributed_mcp(X_data, Y_data, lam):
    #runs distributed mcp
    def penalty(beta):
        gamma = 2.7
        beta = torch.abs(beta)

        mask = (beta <= gamma * lam)
        
        under = gamma * beta - torch.square(beta) / (2 * gamma)
        over = 1/2 * gamma * np.square(lam)
        
        return torch.sum(under * mask + over * (~mask))
        
    def loss_func(inp, out, beta):
        return torch.mean(torch.square(inp @ beta - out)) + penalty(beta)

    return distributed_regression(X_data, Y_data, loss_func, 0.1)

def distributed_scad(X_data, Y_data, lam):
    #runs distributed scad
    def penalty(beta):
        gamma = 3.7
        beta = torch.abs(beta)

        result = lam * beta * (beta <= lam)
        result += (gamma * lam * beta - 0.5 * (torch.square(beta) + np.square(lam)))/(gamma - 1) * (beta > lam) * (beta <= gamma * lam )
        result += np.square(lam) * (gamma**2-1) / (2 * (gamma - 1)) * (beta > gamma * lam)
        
        return torch.sum(result)
    
    def loss_func(inp, out, beta):
        return torch.mean(torch.square(inp @ beta - out)) + penalty(beta)

    return distributed_regression(X_data, Y_data, loss_func, 0.1)

def distributed_regular(X_data, Y_data):
    '''runs a distributed regular regression'''
    def loss_func(inp, out, beta):
        return torch.mean(torch.square(inp @ beta - out)) 

    return distributed_regression(X_data, Y_data, loss_func, 0)
    
def logistic_regression(X_data, Y_data):
    #runs a distributed logistic regression
    def loss_func(inp, out, beta):
        pred = torch.sigmoid(inp @ beta)
        return -torch.sum(out * torch.log(pred))
    
    return distributed_regression(X_data, Y_data, loss_func, 0)


def distributed_regression(X_data, Y_data, loss_func, threshold):
    '''Runs a generic distributed regression. X_data is a list of X data per site. Y_data is a list of Y data per site. 
    loss_func is whatever loss function you want to minimize. Threshold is what fraction of the max element is considered too small.'''
    
    X = [torch.tensor(x, device=device) for x in X_data]
    Y = [torch.tensor(y, device=device) for y in Y_data]


    depth, length = X[0].shape
    losses = [[] for t in range(len(X))]

    #initial guess
    beta = torch.randn([length], dtype=float, device=device, requires_grad=True)
    opt = torch.optim.Adamax([beta], lr=0.01)

    avg_loss = []

    for rep_out in range(100000):
        #lots of cycles. There's a convergence check and break out.
        
        for (inp, out) in zip(X,Y):
            # for each site, we update beta 10 times
            for rep_in in range(10):
                loss = loss_func(inp, out, beta)

                opt.zero_grad()
                loss.backward()
                opt.step()

        #after one cycle, we zero out beta at the threshold, and evaluate the losses. 
        beta_thresh = beta.detach().clone().cpu()
        cutoff = (torch.abs(beta_thresh) > max(beta_thresh) * threshold)
        beta_thresh[~cutoff] = 0

        #the losses at each site with the zeroed out beta, adn teh average loss. 
        thresh_loss = [loss_func(x, y, beta_thresh).detach().cpu().numpy() for (x,y) in zip(X, Y)]
        avg_loss.append(np.mean(thresh_loss))
        for l,t in zip(losses, thresh_loss):
            l.append(t)

        #break at convergence
        if len(avg_loss) > 1:
            if np.abs(avg_loss[-2] - avg_loss[-1]) < tolerance:
                return beta_thresh.detach().cpu().numpy()

def measure_metrics(beta, test, all_data):
    '''takes the penalized regressed beta and the data and evaluates metrics. Note that in this code I have a lot of additional estimators that I ended up not writing about in the report. The report only uses the beta[0] estimator'''

    def narrow_threshold(x):
        #for dividing by probabilities
        a = 0.2
        return np.maximum(np.minimum(x, 1-a),a)

    #selects the significant coefficients
    select_vars = [f"X{x+1}" for x in range(len(beta)) if np.abs(beta[x]) >= 0.1 * np.max(beta)]

    #adds the treatment indicator
    X = [t[['a'] + select_vars].values for t in test]
    Y = [t[outcome].values for t in test]

    #regresses again
    rebeta = distributed_regular(X,Y)

    #calculates various other metrics. Check the paper for more details. 
    pred_y = [x[['a'] + select_vars] @ rebeta for x in all_data]
    m1 = [pred_y[i][all_data[i]['a'] == 1] for i in range(len(all_data))]
    m0 = [pred_y[i][all_data[i]['a'] == 0] for i in range(len(all_data))]

    X = [t[select_vars].values for t in test]
    Y = [t['a'].values for t in test]

    binbeta = logistic_regression(X,Y)

    binpred = [narrow_threshold(sigmoid(x[select_vars].values @ binbeta)) for x in all_data]
    pi1 = [binpred[i][all_data[i]['a'] == 1] for i in range(len(all_data))]
    pi0 = [1-binpred[i][all_data[i]['a'] == 0] for i in range(len(all_data))]

    aug1 = [(1-p)*m/(p) for (m,p) in zip(m1, pi1)]
    aug0 = [(1-p)*m/(p) for (m,p) in zip(m0, pi0)]

    ipw1 = [data[data['a']==1]['yob'].values/bp[data['a']==1] for (data,bp) in zip(all_data, binpred)]
    ipw0 = [data[data['a']==0]['yob'].values/(1-bp[data['a']==0]) for (data,bp) in zip(all_data, binpred)]

    #returns a set of summary metrics
    
    summary_metrics ={
        "or":  np.mean([np.mean(x) - np.mean(y) for (x,y) in zip(m1,m0)]),
        "ipw": np.mean([(sum(x) - sum(y))/(len(x) + len(y)) for (x,y) in zip(ipw1, ipw0)]),
        "dr":  np.mean([np.mean(i1-a1) - np.mean(i0 - a0) for (a0,a1,i0,i1) in zip(aug0, aug1, ipw0, ipw1)]),
        "beta": rebeta[0]
    }
    
    return summary_metrics

device='cpu'
folds = 5
tolerance = 1e-3

for rep in range(20):
    #unique id name
    name = str(uuid4()).split('-')[0]
    
    df = #put your data in here
    
    #formats the data and get the indices we want. Change this as necessary for new data.
    df['site'] = df['site'].astype(int)
    vars = df.columns.tolist()
    vars = [x for x in vars if x[0] == 'X']
    outcome = 'yob'
    
    #partitions the data in train and test sets as well as by site
    sites = set(df['site'].tolist())
    sites = [train_test_split(i, df) for i in sites]
    train = [x[0] for x in sites]
    test = [x[1] for x in sites]
    
    #partitions the data by site but not by train and test
    all_data = [pd.concat([x, y]) for (x,y) in zip(train, test)]
    
    for method in 'lasso scad mcp'.split():
        print(method)
        beta = dist_regression(train, method)
        results = measure_metrics(beta, test, all_data)

        print(results['beta'])