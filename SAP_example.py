import math
import numpy as np
import numpy.random
import scipy
from secant_functions import get_secants, SAP

# Number of points
numb_points = 100
# Dimension of points (should be a multiple of 2)
input_dim = 6
# Dimension of desired projection
proj_dim = 3
# Number of iterations
iterations = 20
# Step size
step_size = .01

# Generate a synthetic data set

# Initialize array
data_points = np.zeros((input_dim,numb_points))

# Choose some points on the circle
for i in range(numb_points):
    a = np.random.uniform(0,2*3.14)
    for j in range(int(input_dim/2)):
        data_points[0,i] = scipy.sin((j+1)*a)
        data_points[1,i] = scipy.cos((j+1)*a)

# Generate initial projection
proj = np.random.rand(input_dim,proj_dim)

# Get orthonormalization
proj, r = np.linalg.qr(proj)

# Get secants
secants = get_secants(data_points)

# Run SAP
proj = SAP(secants,proj,iterations,step_size)

