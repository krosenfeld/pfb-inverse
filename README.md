pfb-inverse
---

##Description:
Invert PFB back into a timestream using CUDA acceleration.

##requirements:
* cvxopt
* lapack

##local install of dgbmv.f
'''
f2py -m dgbmv -h dgbmv.pyf dgbmv.f
f2py -c dgbmv.pyf dgbmv.f -llapack
'''
