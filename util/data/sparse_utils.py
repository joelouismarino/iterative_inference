from scipy.sparse import csr_matrix,csc_matrix,coo_matrix
import time
import os,sys,h5py
import numpy as np
"""
Utilities for sparse matrices from https://github.com/clinicalml/theanomodels
"""
def saveSparseHDF5(matrix, prefix, fname):
    """ matrix: sparse matrix
    prefix: prefix of dataset
    fname : name of h5py file where matrix will be saved
    """
    assert matrix.__class__==csr_matrix or matrix.__class__==csc_matrix,'Expecting csc/csr'
    with h5py.File(fname,mode='a') as f:
        for info in ['data','indices','indptr','shape']:
            key = '%s_%s'%(prefix,info)
            try:
                data = getattr(matrix, info)
            except:
                assert False,'Expecting attribute '+info+' in matrix'
            """
            For empty arrays, data, indicies and indptr will be []
            To deal w/ this use np.nan in its place
            """
            if len(data)==0:
                f.create_dataset(key,data=np.array([np.nan]))
            else:
                f.create_dataset(key,data= data)
        key = prefix+'_type'
        val = matrix.__class__.__name__
        f.attrs[key] = np.string_(val)

def loadSparseHDF5(prefix, fname, changeval = None):
    """ Load from matrix """
    params= []
    data  = None
    with h5py.File(fname, mode='r') as f:
        for info in ['data','indices','indptr','shape']:
            key = '%s_%s'%(prefix,info)
            params.append(f[key].value)
        key = prefix+'_type'
        dtype=f.attrs[key]
        params = [np.array([]) if np.isnan(np.array(k).sum()) else k for k in params]
        if len(params[0])==0: #Empty data matrix
            if dtype  =='csc_matrix':
                data = csc_matrix(tuple(params[3]))
            elif dtype=='csr_matrix':
                data = csr_matrix(tuple(params[3]))
            else:
                raise TypeError('dtype not supported: '+dtype)
        else:                 #Reconstruct sparse matrix while changing data values if necessary
            if changeval is not None:
                params[0] = params[0]*0.+changeval
            if dtype=='csc_matrix':
                data = csc_matrix(tuple(params[:3]),shape=params[3])
            elif dtype=='csr_matrix':
                data = csr_matrix(tuple(params[:3]),shape=params[3])
            else:
                raise TypeError('dtype not supported: '+dtype)
    return data

def _testSparse():
    m1    = np.random.randn(4,12)
    m1[:,3:8] = 0
    m1[0:2,:] = 0
    csr_1 = csr_matrix(m1)
    csc_1 = csc_matrix(m1)
    m2    = np.random.randn(4,12)
    csr_2 = csr_matrix(m2)
    csc_2 = csc_matrix(m2)
    fname = 'tmp.h5'
    saveSparseHDF5(csc_1, 'c1', fname)
    saveSparseHDF5(csr_1, 'r1', fname)
    saveSparseHDF5(csc_2, 'c2', fname)
    saveSparseHDF5(csr_2, 'r2', fname)
    l_csc_1 = loadSparseHDF5('c1', fname)
    l_csr_1 = loadSparseHDF5('r1', fname)
    l_csc_2 = loadSparseHDF5('c2', fname)
    l_csr_2 = loadSparseHDF5('r2', fname)
    result = 0.
    for init,final in zip([csc_1,csr_1,csc_2,csr_2],[l_csc_1,l_csr_1,l_csc_2,l_csr_2]):
        result += (init-final).sum() +(init.toarray()-final.toarray()).sum()
    print 'Diff b/w saved vs loaded matrices',result
    os.unlink(fname)

def readSparseFile(fname, MAXDIM, zeroIndexed=True):
    """
    Sparse format :
    l1: #non-zero elements idx:val idx2:val2 idx3:val3
    """
    from tqdm import tqdm
    start = time.time()
    row, col, val  = [],[],[]
    for idx, line in tqdm(enumerate(open(fname,'r'))):
        words= line.strip().split(' ')
        for sp_w in words[1:]:
            dd = sp_w.split(':')
            cval = int(dd[0])
            if not zeroIndexed:
                cval-=1
            col.append(cval)
            val.append(int(dd[1]))
        row += [idx]*len(words[1:])
        if idx%5000==0:
            assert len(row)==len(col),'Failure.1'+str(len(row))+' vs '+str(len(col))
            assert len(val)==len(col),'Failure.2'+str(len(val))+' vs '+str(len(col))
    matrix = coo_matrix((val,(row,col)),shape=(idx+1,MAXDIM))
    val,row,col=None,None,None
    cmat   = matrix.tocsr()
    print 'Time Taken: ',(time.time()-start)/60.,' minutes'
    return cmat

if __name__=='__main__':
    _testSparse()
