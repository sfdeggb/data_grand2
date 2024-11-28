import numpy as np 

def linear_kernel(x, y):
    return np.dot(x, y)

def rbf_kernel(x, y, gamma=0.1):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * gamma**2))

def pegasos_kernel(data,labels,kernel='linear',lambda_val=0.01,iterations=1000,sigma=1.0):
    n_samples,n_features = data.shape
    alphas = np.zeros(n_samples)
    b=0

    for t in range(1,iterations+1):
        for i in range(n_samples):
            eta = 1/(lambda_val*t)
            if kernel == 'linear':
                kernel_func = linear_kernel
            elif kernel == 'rbf':
                kernel_func = lambda x,y: rbf_kernel(x,y,sigma)
            else:
                raise ValueError(f"Unsupported kernel: {kernel}")
            decision = sum(alphas[j]*labels[j]*kernel_func(data[j],data[i]) for j in range(n_samples)+b)
            if labels[i]*decision<1:
                labels[i]+= eta*(labels[i]-lambda_val*alphas[i])
                b+=eta*labels[i]
    return np.round(alphas,4).tolist(),np.round(b,4)