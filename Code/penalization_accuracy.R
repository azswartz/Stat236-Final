library(simsalapar)
library(parallel)
library(dplyr)
library(tidyr)

doOne <- function(n, tx) {
  library(ncvreg)
  library(glmnet)
  library(coefplot)
  
  dataGen = function(n, tx, p){
    
    beta.p = c(0.05, 0.1, 0.1, -0.05, -0.1)
    beta.y = c(0.5, 1, 0.5, -0.5, -1)
    K = 50
    p.star = 5
    v = 1 ### between-site variance (cannot be bigger than within-site variance)
    sigma.sq = 2 ### within-site variance - the smaller this is, the less bias there is
    
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
    
    err = rnorm(N, 0, 1)
    
    y1 = apply(x.star, 1, function(x) t(beta.y) %*% x) + tx + err
    y0 = apply(x.star, 1, function(x) t(beta.y) %*% x) + err
    
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
  p = 100
  test.df1 = dataGen(n,tx,p) 
  df = test.df1$df
  
  df.site = df[df$site==1,]
  index = sample(1:nrow(df.site), 0.7*nrow(df.site)) 
  train = df.site[index,]
  test = df.site[-index,]
  x <- model.matrix(yob ~., df.site[,
                               c("yob","a",paste("X",1:p,sep=""))])
  x.train = x[index,]
  x.test = x[-index,]
  y.train = train$yob
  y.test = test$yob
  X.true = names(df.site)[2:6]
  lasso <- cv.glmnet(x.train, y.train, alpha = 1, standardize = TRUE, nfolds = 5)
  X.lasso = extract.coef(lasso)$Coefficient[3:length(extract.coef(lasso)$Coefficient)]
  scad <- cv.ncvreg(x.train, y.train, penalty="SCAD")
  X.scad = names(coef(scad)[coef(scad)!=0])[c(-1,-2)]
  mcp <- cv.ncvreg(x.train, y.train, penalty="MCP")
  X.mcp = names(coef(mcp)[coef(mcp)!=0])[c(-1,-2)]

  return(c(sum(X.lasso %in% X.true)/5,
           (p-5-sum(!(X.lasso %in% X.true)))/(p-5),
           sum(X.scad %in% X.true)/5,
           (p-5-sum(!(X.scad %in% X.true)))/(p-5),
           sum(X.mcp %in% X.true)/5,
           (p-5-sum(!(X.mcp %in% X.true)))/(p-5)
           ))
}

run <- function(iter, n, tx) {

  vList <- varlist(
    n.sim = list(value = iter, expr = quote(N[sim])), # type = N
    n = list(type="grid", value=n),
    tx = list(type="grid", value=tx)
  )
  
  # When this gives an error use cluster = makeCluster(1)
  # cluster = parallel::makeCluster(2, setup_strategy = "sequential"),
  # cluster=makeCluster(detectCores(), type="PSOCK")
  raw.results <- doForeach(vList, 
                           #cluster=makeCluster(detectCores(), type="PSOCK"),
                           cluster = parallel::makeCluster(4, setup_strategy = "sequential"),
                           cores=NULL, block.size = 1, seed="seq", repFirst=TRUE,
                           sfile=NULL, check=TRUE, doAL=TRUE, subjob.=subjob, monitor=FALSE,
                           doOne, extraPkgs=character(), exports=character())
  val <- getArray(raw.results)
  # D.1 = 1 => Model.1 beta, etc.
  df <- array2df(val)
  parsed.output = df %>% mutate(type = rep(c("Sensitivity","Specificity"),nrow(df)/2),
                     penalty = rep(rep(
                       c("LASSO","SCAD","MCP"),each=2)
                     ,nrow(df)/(2*3))) %>%
    select(-1) %>%
    spread(type,value)%>%
    group_by(penalty,tx,n) %>%
    summarize(Sensitivity=mean(Sensitivity),
              Specificity=mean(Specificity)) %>%
    dplyr::arrange(tx,n,penalty) %>%
    select(tx,n,penalty,Sensitivity,Specificity) 
  
  print(xtable(parsed.output), include.rownames=FALSE)
  return(list(parsed = parsed.output,
              raw = raw.results,
              vList = vList))
}

args = commandArgs(trailingOnly=TRUE)

results <- run(iter = as.numeric(args[1]),  
            n = c(100, 200, 500),
            tx = c(0, 10)
            )

results$parsed

save(file=paste("~/results/perm", Sys.Date(),".Rda" ,sep=""), results)


