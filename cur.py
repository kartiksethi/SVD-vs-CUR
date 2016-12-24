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
import matplotlib.pyplot as plt

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

print A.shape

print "Specify the number of rows to be sampled randomly"
numr = int(raw_input())

print "Specify the number of columns to be sampled randomly"
numc = int(raw_input())
start = timeit.default_timer()
c,u,r,reconstructionerror = CUR(A,numr,numc,nrows,ncols)
#np.array([[1,1,1,0,0],[3,3, 3, 0, 0],[4 ,4, 4, 0, 0],[5,5 ,5, 0, 0],[0, 0, 0, 4, 4],[0, 0, 0, 5, 5],[0, 0, 0, 2, 2]]),3,3,7,5)
print "The C matrix is : \n" 
print c
print "The U matrix is : \n"
print u
print "The R matrix is : \n"
print r
print "The reconstruction error is " 
print reconstructionerror
stop = timeit.default_timer()
print "The time taken is "
print str(stop-start)