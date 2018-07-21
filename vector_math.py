import math
import numpy as np

def dotproduct(u, v):
  return sum((a*b) for a, b in zip(u, v))

def length(u):
  return math.sqrt(dotproduct(u, u))

def angle(u, v):
  return (dotproduct(u, v) / (length(u) * length(v)))

def crossproduct(u, v):
	n=np.array([0,0,0])
	n[0] = u[1]*v[2] - u[2]*v[1]
	n[1] = u[2]*v[0] - u[0]*v[2]
	n[2] = u[0]*v[1] - u[1]*v[0]
	return n