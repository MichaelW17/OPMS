"""Exact-rational certificates for frozen numerical coefficients.

Source embedding uses the saved complex factor entries exactly as binary
rationals. SNR certificate concerns the archived finite response polynomial,
not an uncountable atmospheric population or exact propagation arithmetic.
"""
from pathlib import Path
from fractions import Fraction as F
from math import comb
import json
import numpy as np
import scipy.linalg as la
P=Path(__file__).resolve().parent
def rat(x):return F(float(x))
def add(a,b):
 c=dict(a)
 for k,v in b.items():c[k]=c.get(k,F(0))+v
 return {k:v for k,v in c.items() if v}
def scale(a,v):return {k:x*v for k,x in a.items() if x*v}
def mul(a,b):
 c={}
 for (i,j),x in a.items():
  for (k,l),y in b.items():c[i+k,j+l]=c.get((i+k,j+l),F(0))+x*y
 return {k:v for k,v in c.items() if v}
def diff(a,axis):
 c={}
 for k,v in a.items():
  if k[axis]:z=list(k);z[axis]-=1;c[tuple(z)]=v*k[axis]
 return c
def minimum_bernstein(poly,box):
 # Exact conversion from monomials on [lo,hi]^2 to Bernstein coefficients.
 n=max([k[0] for k in poly]+[0]);m=max([k[1] for k in poly]+[0]);x0,x1,y0,y1=box;p={}
 for (i,j),v in poly.items():
  for a in range(i+1):
   for b in range(j+1):p[a,b]=p.get((a,b),F(0))+v*comb(i,a)*x0**(i-a)*(x1-x0)**a*comb(j,b)*y0**(j-b)*(y1-y0)**b
 vals=[]
 for k in range(n+1):
  for l in range(m+1):vals.append(sum((p.get((i,j),F(0))*F(comb(k,i),comb(n,i))*F(comb(l,j),comb(m,j)) for i in range(k+1) for j in range(l+1)),F(0)))
 return min(vals)
def positive_on_square(polys,radius=F(1,10),cells=8):
 bound=None
 for i in range(cells):
  for j in range(cells):
   box=(-radius+2*radius*i/cells,-radius+2*radius*(i+1)/cells,-radius+2*radius*j/cells,-radius+2*radius*(j+1)/cells)
   for q in polys:
    lo=minimum_bernstein(q,box)
    if bound is None or lo<bound:bound=lo
    if lo<=0:return False,float(lo)
 return True,float(bound)
def complex_product_real(a,b):return rat(a.real)*rat(b.real)+rat(a.imag)*rat(b.imag)
def source_poly(Factors,row=None,col=None,imag=False):
 # Real/imaginary selected CSD numerator, or total Frobenius norm.
 modes=[(0,0),(1,0),(0,1)];p={}
 for a in range(3):
  for b in range(3):
   k=tuple(modes[a][i]+modes[b][i] for i in range(2));v=F(0)
   if row is None:
    for x,y in zip(Factors[a].ravel(),Factors[b].ravel()):v+=complex_product_real(x,y)
   else:
    for x,y in zip(Factors[a,row],Factors[b,col]):
     if imag:v+=rat(x.imag)*rat(y.real)-rat(x.real)*rat(y.imag)
     else:v+=complex_product_real(x,y)
   p[k]=p.get(k,F(0))+v
 return {k:v for k,v in p.items() if v}
def response_poly(Q):
 return {(0,0):rat(Q[0,0]),(1,0):2*rat(Q[0,1]),(0,1):2*rat(Q[0,2]),(2,0):rat(Q[1,1]),(1,1):2*rat(Q[1,2]),(0,2):rat(Q[2,2])}