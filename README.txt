We included the codes, but did not include the data due to the lack of space. Only small datasets like Iris and Wine datasets have been included. In case any other dataset is downloaded, please assign the data to 'X' and the labels to 'trueLabel' to emulate the experiments. Also, please make each row of X represent each data instance.

The codes will require Tensor Toolbox version 2.5 from Sandia National Library. Please do not use any other version. 


The main modules are as follows:

KMRand(X,K,tolerance) := K-Means with Random Initialization
KMPP(X,K,tolerance):= K-Means++
KMPL(X,K,tolerance):= K-Means||
MoM(X,K,tolerance):= K-Means with MoM initialization
