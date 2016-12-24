import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.sparse import *
from scipy.sparse.linalg import norm
import random
from scipy import linalg
import timeit
from numpy import linalg as LA
import matplotlib.pyplot as plt

# Power Iteration 
# vects will be a list of eigen vectors (it is equivalent to V.T)
# eigvals will be list of eigen values
# number of eigen values to be retrieved using power iteration
def power_iteration(sq_matrix, k):
    vects = []
    eigvals = []
    for i in range(0,k):
        x = np.random.rand(sq_matrix.shape[0],1)
        for j in range(0, 1000):           # run it for 1000 iterations
            temp = sq_matrix.dot(x)
            denom = LA.norm(temp)
            temp /= denom
            x = temp
        lmbda = (x.T).dot(sq_matrix)
        lmbda = lmbda.dot(x)
        eigvals.append(lmbda)
        vects.append(x)
        sq_matrix = sq_matrix - (lmbda * (x.dot(x.T)))

    return (eigvals, vects)
    


def SVD(A,dim):   

    A_T = np.transpose(A)      # downcasting the data type of the matrix to float16
    A = sp.csr_matrix(A)       # converting into a Compressed Sparse Row matrix
    A_T = sp.csr_matrix(A_T)   # converting the tranpose into a Compressed Sparse Row matrix

    # V matrix calculation
    M = np.dot(A_T, A)
    sigma_sq, V = sp.linalg.eigs(M, k = dim)
    sigma_sq = sigma_sq.real
    V = V.real
    idx = sigma_sq.argsort()[::-1]
    sigma_sq = sigma_sq[idx]
    V = V[:,idx]
    # sigma_sq_pi, V = power_iteration(M, dim) # calculating the eigen vectors and eigen values using the power iteration algorithm
    # V = np.array(V)
    # V = np.array(V.T)
    # V = np.squeeze(V, axis=(0,))
    
    #calculating the sigma matrix
    Eig_mat= np.zeros(shape=(dim,dim))
    np.fill_diagonal(Eig_mat, sigma_sq, wrap=True)
    Eig_mat = np.sqrt(Eig_mat)
    
    # U matrix calculation
    N = np.dot(A, A_T)
    sigma_sq2, U = sp.linalg.eigs(N, k = dim)
    sigma_sq2 = sigma_sq2.real
    U = U.real
    idx = sigma_sq2.argsort()[::-1]
    sigma_sq2 = sigma_sq2[idx]
    U = U[:,idx]

    for i in range(0,dim-1):
    	temp = (A*(V[:,i]))/U[:,i]
    	if temp[0] < 0:
    		U[:,i] = U[:,i] * -1.0

    return (U ,Eig_mat, V)

"""Function computes the CUR decomposition of matrix A, sampling numr rows and numc columns. A is of size nrows X ncols."""
def CUR(mat,numr,numc,nrows,ncols):
	A_T = np.transpose(mat)
	A = sp.csr_matrix(mat)
	A_T = sp.csr_matrix(A_T)
	rows = np.zeros(nrows, dtype=float)
	r = []
	rowsprob = np.zeros(nrows, dtype=float)
	total = 0
	tempr = sp.coo_matrix(A)
	#calculating the probability of each row being selected
	for i,j,v in zip(tempr.row, tempr.col, tempr.data):
		rowsprob[i] = rowsprob[i] + v*v
		total = total + v*v

	rowsprob = rowsprob/total
	#calculating the cumulative probabilities
	cumrowsprob = np.zeros(nrows, dtype=float)
	cumrowsprob[0] = rowsprob[0]
	for i in range(1,rowsprob.size):
		cumrowsprob[i] = cumrowsprob[i-1] + rowsprob[i]
		
	#generating random rows and building r matrix
	for i in range(0,numr):
		rand = random.random()
		entry = np.searchsorted(cumrowsprob,rand)
		rows[entry] = rows[entry] + 1	

	#handling duplicates by multiplying duplicate rows with square root of number of duplications and removing duplicates 
	selectedrows = []
	rows = np.sqrt(rows)
	for i in range(0,nrows):
		if rows[i]>0:
			r.append((A[i].toarray()/((numr*rowsprob[i])**0.5))*rows[i])
			selectedrows.append(i)

	cols = np.zeros(ncols, dtype=float)
	c = []
	colsprob = np.zeros(ncols, dtype=float)
	total = 0
	tempc = sp.coo_matrix(A_T)
	#calculating the probability of each column being selected
	for i,j,v in zip(tempc.row, tempc.col, tempc.data):
		colsprob[i] = colsprob[i] + v*v
		total = total + v*v

	colsprob = colsprob/total
	#calculating the cumulative probabilities
	cumcolsprob = np.zeros(ncols, dtype=float)
	cumcolsprob[0] = colsprob[0]
	for i in range(1,colsprob.size):
		cumcolsprob[i] = cumcolsprob[i-1] + colsprob[i]
		
	#generating random cols and building r matrix
	for i in range(0,numc):
		rand = random.random()
		entry = np.searchsorted(cumcolsprob,rand)
		cols[entry] = cols[entry] + 1	

	#handling duplicates by multiplying duplicate columns with square root of number of duplications and removing duplicates
	selectedcols = []
	cols = np.sqrt(cols)
	for i in range(0,ncols):
		if cols[i]>0:
			c.append((A_T[i].toarray()/((numc*colsprob[i])**0.5))*cols[i])
			selectedcols.append(i)

	c = np.vstack(c)
	r = np.vstack(r)
	#finding the intersection of c and r = w
	w = np.zeros(shape=(len(selectedrows),len(selectedcols)))
	for i in range(0,len(selectedrows)):
		for j in range(0,len(selectedcols)):
			w[i][j] = mat[selectedrows[i]][selectedcols[j]]

	c = sp.csr_matrix(c)
	c = c.transpose()
	r = sp.csr_matrix(r)

	#computing the SVD decomposition of the w matrix
	x,z,y_T = linalg.svd(w)
	z = linalg.diagsvd(z, x.shape[1], y_T.shape[0])
	y = np.transpose(y_T)

	#computing the u matrix
	zplus = linalg.pinv(np.matrix(z))
	zplussquare = zplus*zplus
	u = np.matmul(y,np.matmul(zplussquare,np.transpose(x)))

	#computing the reconstructed matrix and error
	reconstructedmatrix = c*(u*r)
	errormatrix = sp.csr_matrix(A-reconstructedmatrix)
	reconstructionerror = norm(errormatrix)
	return (c,u,r,reconstructionerror)

random.seed(97)
df = pd.read_excel("jester-data-3.xls", header=None)
df = df.replace(99.00, 0)
A = df.as_matrix()
nrows = 5000
ncols = 100
A[A==99]=0
A=A[:nrows,1:ncols+1]
A = np.array(A)

curtime = []
curerror = []
cursize = []
samplesize = []
svdsize = []
svdtime = []
svderror = []
for i in xrange(1000,2500,100):
	temp = A[:i,:]
	start = timeit.default_timer()
	U, sigma, V = SVD(temp,98)
	svdsize.append(sp.csr_matrix(U).data.nbytes+sp.csr_matrix(sigma).data.nbytes+sp.csr_matrix(V).data.nbytes)
	H = np.dot(U, sigma)
	H = H.dot(V.T)
	reconstructionError = temp - H
	reconstructionError = LA.norm(reconstructionError)
	svderror.append(reconstructionError)
	stop = timeit.default_timer()
	svdtime.append(float(stop-start))
	numr = 1000
	numc = 75
	start = timeit.default_timer()
	c,u,r,reconstructionerror = CUR(temp,numr,numc,nrows,ncols)
	#np.array([[1,1,1,0,0],[3,3, 3, 0, 0],[4 ,4, 4, 0, 0],[5,5 ,5, 0, 0],[0, 0, 0, 4, 4],[0, 0, 0, 5, 5],[0, 0, 0, 2, 2]]),3,3,7,5)
	stop = timeit.default_timer()
	curtime.append(float(stop-start))
	curerror.append(reconstructionerror)
	samplesize.append(i)
	cursize.append(c.data.nbytes+sp.csr_matrix(u).data.nbytes+r.data.nbytes)



plt.plot(np.array(samplesize),np.array(curerror), 'ro', linewidth=2.0)
plt.plot(np.array(samplesize),np.array(svderror), 'bo', linewidth=2.0)
plt.xlabel('Number of rows in the original matrix')
plt.ylabel('Reconstruction error')
#fit = np.polyfit(np.array(x), np.array(recon), deg=1)
#plt.plot(np.array(x), fit[0] * np.array(x) + fit[1], color='red')
#plt.scatter(np.array(x), np.array(recon))
plt.show()

plt.plot(np.array(samplesize),np.array(curtime), 'ro', linewidth=2.0)
plt.plot(np.array(samplesize),np.array(svdtime), 'bo', linewidth=2.0)
plt.xlabel('Number of rows in the original matrix')
plt.ylabel('Time taken')
#fit = np.polyfit(np.array(x), np.array(recon), deg=1)
#plt.plot(np.array(x), fit[0] * np.array(x) + fit[1], color='red')
#plt.scatter(np.array(x), np.array(recon))
plt.show()

plt.plot(np.array(samplesize),np.array(cursize), 'ro', linewidth=2.0)
plt.plot(np.array(samplesize),np.array(svdsize), 'bo', linewidth=2.0)
plt.xlabel('Number of rows in the original matrix')
plt.ylabel('Space consumed')
#fit = np.polyfit(np.array(x), np.array(recon), deg=1)
#plt.plot(np.array(x), fit[0] * np.array(x) + fit[1], color='red')
#plt.scatter(np.array(x), np.array(recon))
plt.show()