library(tidyverse)
library(dplyr)
library(nleqslv)
library(ncvreg)
library(glmnet)
library(ggplot2)
library(reshape2)
library(lemon)
library(foreach)
library(doParallel)
library(coefplot)
library(xtable)

#########################
#### Data Generation ####
#########################

############
## Inputs ##
############

# n = sample size per site (scalar, assuming equal in all sites)
# K = number of sites (scalar)
# p = number of covariates (X is n*K x p)
# p.star = number of significant covariates (this will give X.star matrix n*K x p.star)
# b = number of binary variables
# v = between-site variance (scalar)
# v.b = between.site variance for binary variables (scalar, gives range of uniform)
# sigma.sq = population error
# tx = treatment effect
# beta.p = vector of true propensity score coefficients
# beta.y = vector of true outcome coefficients

#############
## Outputs ##
#############

# df = data frame of person-level covariates, outcomes, treatment, counterfactuals, etc.
# X = covariate matrix with site as first column
# X.star = matrix of significant covariates
# A = true observed treatment
# Y = observed outcome
# Y1 = counterfactual outcome forcing treatment = 1
# Y0 = counterfactual outcome forcing treatment = 0
# pr = true propensity score
# true.ate = overall ATE
# site.ate = site-specific ATE

#p.star = 3

dataGen = function(n = 200, K = 50, p = 6, p.star = 3, 
                   v = 2, sigma.sq = 2, tx = 3,
                   beta.p, beta.y, target, target.delta){
  
  #########################
  ## Generate Covariates ## # generate them "smartly" so that regularization works well to select X.star
  #########################
  # first column: site
  # next p.star columns are significant variables
  # there are b binary covariates
  
  N = n*K # total sample size
  
  
  # get contiunuous covariates
  c.m = matrix(rnorm(p*K, 0, v), nrow = K, ncol = p)
  c.x.t = apply(c.m, c(1, 2), function(x) rnorm(n, x, sigma.sq))
  c.x = apply(c.x.t, 3L, c)
  
  x.t = cbind(c.x[,1:(p.star)],c.x[,(p.star + 1):p])
  site = as.factor(sort(rep(seq(1:K), n)))
  x = data.frame(cbind(site, x.t))
  colnames(x) = c("site", paste0("X", seq(1:p)))
  x[x$site %in% target,2:p] = x[x$site %in% target,2:floor(p.star/2)] + target.delta
  x.star = x[,2:(p.star + 1)]
  
  ######################
  ## Propensity Score ##
  ######################
  #here, we make simplifying assumption that true propensity score depends on same variables as outcome
  
  thing = apply(x.star, 1, function(x) t(beta.p) %*% x)
  pr = exp(thing)/(1 + exp(thing)) # propensity score
  a = rbinom(N, 1, pr) # actual treatment received 
  
  #############
  ## Outcome ##
  #############
  
  #err = rnorm(N, 0, 1)
  
  y1 = apply(x.star, 1, function(x) t(beta.y) %*% x) + tx + 50*x[,2]/sum(x[,2])*tx + rnorm(N, 0, 1)
  y0 = apply(x.star, 1, function(x) t(beta.y) %*% x) +  rnorm(N, 0, 1)
  
  a1 = a
  a0 = 1 - a1
  yob = a0*y0 + a1*y1  
  aob = a1 
  
  df = data.frame(site, y1, y0)
  df$site = as.factor(df$site)
  true.ate = mean(y1 - y0)
  site.ate = sapply(by(df, df$site, function(x) mean(x$y1 - x$y0)), mean)
  
  #############
  ## Return! ##
  #############
  
  lt = list(df = data.frame(x, a, yob, y1, y0), X = x, X.star = data.frame(site, x.star), A = data.frame(site, a), 
            Y = data.frame(site, yob), Y1 = data.frame(site, y1), Y0 = data.frame(site, y0), 
            pr = data.frame(site, pr), true.ate = true.ate, site.ate = site.ate)
  
  return(lt)
}

##############

# p.star = 3
# K = 50
# d = dataGen()
# df = d$df
# target = seq(1:K)
# central = 1

####################################################################################################################

#######################
#### Model Fitting ####
#######################

############
## Inputs ##
############

# Target Site
# Models to use for M, Pi, W, and er.tate
# Data
# Central site

#############
## Outputs ##
#############

# M1 = estimated site-specific outcome forcing treatment = 1 (K vector)
# M0 = estimated site-specific outcome forcing treatment = 0 (K vector)
# Pi = estimated propensity score (N=n*K vector)
# W = estimated density ratio weights (N=n*K vector)
# tate = Targeted ATE, one estimate from each site (K vector)
# er.tate = "Efficient and robust TATE" (smart combination, perhaps weighted median?)
# default target site is all of the sites (ATE)

D.TATE = function(df, target=unique(df$site), central = 1, X.star.names, or.bias){
  # get covariate matrix and outcomes
  X = df[, grepl( "X" , names(df))]
  # assuming we know significant variables
  X.star = X[,X.star.names]
  X.star.wsite = cbind(site = df$site, X.star)
  # outcome variable
  Y = df[, grepl("yob" , names(df))]
  
  ################################
  ## Step 1: Central site means ##
  ################################
  
  central.mean = colMeans(df[df$site == central,X.star.names])
  
  ############################################################
  ## Step 2a: Compute gamma for each covariate at each site ##
  ############################################################
  
  K = length(unique(df$site))
  
  # This is the function to be optimized
  fn = function(gamma, r){
    site.df = X.star.wsite[X.star.wsite$site == r,]
    site2 = as.matrix(site.df[,-1])
    n = nrow(site2)
    y = (1/n)*(t(site2) %*% (exp(site2 %*% as.matrix(gamma))) - central.mean)
    return(as.numeric(y))
  }
  
  oneSite = function(r){
    return(nleqslv(numeric(length(X.star.names)), fn, jac=NULL, r)$x)
  }
  
  # Here we store the gamma values in a p.star by K matrix
  gamma = sapply(1:K, oneSite)
  
  ################
  ## Get Omegas ##
  ################
  
  # Next we want to compute omega and w
  # I'm using the formula that omega at site r, 
  # individual i is 
  # omega_ri = exp(sum_p {gamma_rp * X_rpi})
  # Note that X_rpi is the p-th covariate for the i-th 
  # individual in site r
  df.gamma = data.frame(site = 1:K, t(gamma))
  names(df.gamma)[-1] = paste("gamma",X.star.names,sep="")
  df = merge(df, df.gamma, all=TRUE)
  df$omega = exp(rowSums(df[,paste("gamma",X.star.names,sep="")]*
                           X.star))
  df$omega = ifelse(df$omega >= quantile(df$omega, 0.99), quantile(df$omega, 0.99), df$omega)
  
  # Here w_ri = omega_ri/(sum_r sum_i rho_r omega_ri)
  df$w = df$omega/sum(df[df$site %in% target,]$omega)
  df$w = df$w/sum(df$w)
  
  #######################
  ## Propensity Scores (IPW) ## # can add regularization
  #######################
  
  predict.ps = function(r) {
    ps.model = glm(a ~ ., data=df[df$site==r,
                                  c("a",X.star.names)],
                   family=binomial)
    return(predict(ps.model,type="response"))
  }
  
  ########################
  ## Outcome Regression (OR) ## # can add regularization
  ########################
  
  # predict.outcome = function(r) {
  #   outcome.model = lm(yob ~ ., data=df[df$site==r,
  #                                       c("yob", "a", X.star.names)])
  #   lt = list(pred = predict(outcome.model), fitted.a = coef(outcome.model)[2])
  #   return(lt)
  # }
  
  ################################
  ## New Regression Interaction ##
  ################################
  
  predict.outcome = function(r) {
    outcome.model = lm(as.formula(paste("yob ~ a", paste(X.star.names, collapse = " + "), 
                                        sep = " + ")), data=df[df$site==r,])
    lt = list(pred = predict(outcome.model), fitted.a = coef(outcome.model)[2])
    return(lt)
  }
  
  
  #######################################
  ## Apply PS and OR Models over sites ##
  #######################################
  
  # Note: this only works if data is sorted by site, which it is
  # Compute site specific fitted values
  df$fitted.ps = c(sapply(1:K, predict.ps))
  
  temp = sapply(1:K, predict.outcome)
  df$fitted.outcome = unlist(temp[1,])
  
  ####################################################
  ## Common Functions used for different estimators ##
  ####################################################
  
  ipw = function(site){
    return(site$a*site$yob/site$fitted.ps - (1 - site$a)*site$yob/(1-site$fitted.ps))
  }
  
  mu.simple1 = function(r) {
    m = lm(as.formula(paste("yob ~ a", paste(X.star.names, collapse = " + "), 
                            sep = " + ")), data=df[df$site==r,])
    return(predict(m, newdata = data.frame(yob = df[df$site==r,]$yob, a=1, df[df$site==r,X.star.names])))
  }
  
  mu.simple0 = function(r) {
    m = lm(as.formula(paste("yob ~ a", paste(X.star.names, collapse = " + "), 
                            sep = " + ")), data=df[df$site==r,])
    return(predict(m, newdata = data.frame(yob = df[df$site==r,]$yob, a=0, df[df$site==r,X.star.names])))
  }
  
  #################################################################
  ## Very simple IPW and OR estimators using only target site(s) ##
  #################################################################
  
  ## IPW ##
  
  mu.simple.ipw = function(r) {
    site = df[df$site==r, ]
    return(mean(ipw(site)))
  }
  
  # simple(mu) per target site
  mu.simple.ipw.site = sapply(target, mu.simple.ipw)
  simple.ipw.delta.site = mu.simple.ipw.site
  
  # overall simple(mu)
  simple.ipw.delta.overall = mean(simple.ipw.delta.site)
  
  ## OR ##
  
  or.est = function(r) {
    site = site = df[df$site==r, ]
    m1 = mu.simple1(r)
    m0 = mu.simple0(r)
    
    return(mean(m1 - m0))
  }
  
  # simple(mu) per target site
  simple.or.delta.site = unlist(temp[2,])
  
  # overall simple(mu)
  simple.or.delta.overall = mean(simple.or.delta.site)
  
  ################################################
  ## Simple doubly robust w/out density ratio weight estimator using only target site(s) ##
  ################################################
  
  dr.est = function(r) {
    site = df[df$site==r, ]
    m1 = mu.simple1(r)
    m0 = mu.simple0(r)
    
    return(mean(site$a*site$yob/site$fitted.ps - m1*(site$a - site$fitted.ps)/site$fitted.ps, na.rm = TRUE) -
             mean((1-site$a)*site$yob/(1-site$fitted.ps) + m0*(site$a - site$fitted.ps)/(1-site$fitted.ps), na.rm = TRUE))
  }
  
  # simple(mu) per target site
  simple.delta.site = sapply(1:K, dr.est)
  
  # overall simple(mu)
  simple.delta.overall = mean(simple.delta.site)
  
  # simple(mu) everywhere
  naive.simple.delta.site = sapply(target, dr.est)
  
  # overall simple(mu)
  naive.simple.delta.overall = mean(naive.simple.delta.site)
  
  ####################################
  ## Use all sites with DR weights  ##
  ####################################
  
  # This function computes the doubly robust site-specific mu 
  # using w rather than 1/w and no 1/n term in the return
  
  wdr.est = function(r) {
    site = df[df$site==r, ]
    m1 = mu.simple1(r)
    m0 = mu.simple0(r)
    
    lt = list(est = sum((site$w)/sum(site$w) * ((site$a*site$yob/site$fitted.ps - m1*(site$a - site$fitted.ps)/site$fitted.ps) -
                                                ((1-site$a)*site$yob/(1-site$fitted.ps) + m0*(site$a - site$fitted.ps)/(1-site$fitted.ps)))),
              wt = sum(site$w))
    
    return(lt)
  }
  
  # tilde(mu) per site
  delta.site = sapply(1:K, wdr.est)
  
  # Doubly robust estimator
  est = unlist(delta.site[1,])
  wt = unlist(delta.site[2,])
  top.half.wt = wt[wt>median(wt)]
  top.half.est = est[wt>quantile(wt, (K-length(target))/K)]
  delta.overall = median(top.half.est)
  
  #############
  ## Return! ##
  #############
  
  # True ATE
  target.df = df[df$site %in% target, ]
  target.ate = mean(target.df$y1-target.df$y0)
  
  overall.ate = c(target.ate, simple.or.delta.overall, simple.ipw.delta.overall, 
                  simple.delta.overall, naive.simple.delta.overall, delta.overall)
  names(overall.ate) = c("Target.ATE","OR","IPW","DR","Naive.DR","Weighted.DR")
  overall.ate
  
  lt = list(overall.ate = overall.ate,
            simple.ate.sd = sd(simple.delta.site),
            weighted.ate.sd = sd(delta.site),
            target.ate.site = sapply(by(target.df, target.df$site, function(x) mean(x$y1-x$y0)), mean),
            simple.or.delta.site = simple.or.delta.site,
            simple.ipw.delta.site = simple.ipw.delta.site,
            simple.delta.site = simple.delta.site,
            weighted.delta.site = delta.site,
            ps = df$fitted.ps
  )
  
  return(lt)
  
}

###################################################################
#### Test ####
###################################################################

# n = 200 
# K = 30
# p = 750 
# p.star = 5 
# v = 2
# sigma.sq = 2
# tx = 10
# beta.p = rnorm(p.star, 0, 1)
# beta.y = rnorm(p.star, 0, 1)

n = 100
K = 50
p = 100
p.star = 5
v = 1 ### between-site variance (cannot be bigger than within-site variance)
sigma.sq = 2 ### within-site variance - the smaller this is, the less bias there is
tx = 10 ### treatment effect (0 is null)
beta.p = c(0.05, 0.1, 0.1, -0.05, -0.1)
beta.y = c(0.5, 1, 0.5, -0.5, -1)
dg.target = seq((K-4),K)
target = list(dg.target, seq(1:K))
target.d = 5
X.star.names = paste0("X", seq(1:p.star))

simreps=30
count = 1
sim.output = list()
for (i in 1:simreps){
  for(psim in p){
    for(nsim in n){
    for(t in tx){
    for(tar in target){
      test.df1 = dataGen(nsim, K, psim, p.star, 
                         v, sigma.sq, t,
                         beta.p, beta.y, dg.target, target.d) 
      
      central = 1
      df = test.df1$df
      
      test.fit1 = D.TATE(df, tar, central, X.star.names, or.bias)
      sim.output[[count]] = test.fit1
      
      print(count)
      count = count + 1
  }
  }
  }
  }
}

saveRDS(sim.output, "SimOutput.KnownXStar.NewEst.50.onesetting.EM.RDS")

out.overall.ate = unlist(lapply(sim.output, `[`, c('overall.ate')))
out.overall.ate2 = matrix(out.overall.ate, ncol=count-1)
rownames(out.overall.ate2) = c("Target.ATE","OR","IPW","DR","Naive.DR","Weighted.DR")
ate.output = t(out.overall.ate2)
ate.output = data.frame(ate.output)

ate.output$p = rep(sort(rep(p, nrow(ate.output)/(simreps*length(p)))), simreps)
ate.output$n = rep(sort((rep(n, length(tx)*length(target)))), nrow(ate.output)/(length(n)*length(tx)*length(target)))
ate.output$tx = rep(sort(rep(tx, length(target))), nrow(ate.output)/(length(tx)*length(target)))
ate.output$tar = rep(c("five", "all"), nrow(ate.output)/length(target))
ate.output$setting = rep(seq(1:(length(p)*length(n)*length(tx)*length(target))), simreps)

ate.output$bias.or = ate.output$OR - ate.output$Target.ATE
ate.output$bias.ipw = ate.output$IPW - ate.output$Target.ATE
ate.output$bias.dr = ate.output$DR - ate.output$Target.ATE
ate.output$bias.ndr = ate.output$Naive.DR - ate.output$Target.ATE
ate.output$bias.wdr = ate.output$Weighted.DR - ate.output$Target.ATE

write.csv(ate.output, "ate.output.KnownXStar.NewEst.onesetting.EM.csv")

###############################################

ate.output.n = ate.output[ate.output$n == 100,]

data.plot.n = melt(ate.output.n, id.vars=c('setting', 'p', 'n', 'tx', 'tar'), 
                   measure.vars=c('bias.or','bias.ipw','bias.dr','bias.ndr', 'bias.wdr'))

ggplot(data.plot.n) +
  geom_boxplot(aes(y=value, fill=variable), alpha = 0.3) +
  facet_wrap(vars(setting)) +
  geom_abline(intercept = 0, slope = 0) +
  theme_bw()

##############

ate.output = read.csv("ate.output.KnownXStar.NewEst.csv")[,-1]

count = 1
for(i in seq(12, 16)){
  if(count == 1){
    bias.df = aggregate(ate.output[,i] ~ p + n + tx + tar, FUN=mean, data=ate.output)
    sd.df = aggregate(ate.output[,i] ~ p + n + tx + tar, FUN=sd, data=ate.output)
    count = count + 1
    } else{
 bias.df = cbind(bias.df, aggregate(ate.output[,i] ~ p + n + tx + tar, FUN=mean, data=ate.output)[,5])
 sd.df = cbind(sd.df, aggregate(ate.output[,i] ~ p + n + tx + tar, FUN=sd, data=ate.output)[,5])
    }
}

colnames(bias.df) = c(colnames(bias.df)[1:4], colnames(ate.output)[12:16])
colnames(sd.df) = c(colnames(sd.df)[1:4], colnames(ate.output)[12:16])

bias.df[,5:9] = round(bias.df[,5:9], 3)
sd.df[,5:9] = round(sd.df[,5:9], 3)

write.csv(bias.df, "bias.df.csv")
write.csv(sd.df, "sd.df.csv")
