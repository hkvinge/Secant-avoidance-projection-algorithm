import math
import numpy as np
import numpy.linalg
import numpy.random
import matplotlib.pyplot as plt
import scipy

def get_secants(data_points):
    """Calculate the normalized secant set for an array
    of data point.
    
    Args:
        data_points (NumPy float array): A NumPy float
            array. The data points are assumed to be
            given by the columns of the array.
            
    Yields:
        NumPy array: The normalized secants set of data_points
            stored as a NumPy array with the secants taking the form
            of columns.
            
    """
    
    # Calculate dimension of data array
    dims = np.shape(data_points)
    # Ambient dimension of the data
    input_dim = dims[0]
    # Number of data points
    numb_points = dims[1]

    # Initialize array to hold secants
    secants = np.zeros((input_dim,int((numb_points-1)*numb_points/2)))

    # Initialize counter to count secants
    count = 0

    # Calculate secants for set of points
    for i in range(numb_points):
        for j in range(numb_points):
            if i < j:
                secants[:,count] = data_points[:,i] - data_points[:,j]
                norm = np.linalg.norm(secants[:,count])
                secants[:,count] = (1/norm)*secants[:,count]
                count = count + 1

    # Return array of secants
    return secants

def SAP(secants,proj,iterations,step_size):
    """Runs the SAP algorithm on a data set
        
    Args:
        secants (NumPy float array): A NumPy float array containing
            the secants of the data set as columns.
        proj (NumPy float array): A NumPy float array whose columns are
            a set of orthonormal vectors that span the projection subspace.
        iterations (int): The number of iterations that the algorithm will
            run.
        step_size (float): The step size for each shift of the projection
            subspace (usually between .01-.1).
        
    Yields:
        (NumPy float array): A NumPy float array whose columns give a set
            of orthonormal vectors that span the projection subspace.
        
    """
    
    # Initialize array to record shortest secants for kappa-profile
    worst_proj_secants_record = np.zeros((iterations,1))

    # Run the SAP algorithm for some number of iterations
    for i in range(iterations):

        # The projection of each secant
        secant_projections = np.matmul(np.transpose(proj),secants)

        # Calculate norm of each column
        secant_norms = np.linalg.norm(secant_projections,axis=0)

        # Calculate minimum index
        index_min = np.argmin(secant_norms)
    
        # Print smallest projected secant norm
        print("The smallest projected secant norm is:  " + str(secant_norms[index_min]))
        worst_proj_secants_record[i] = secant_norms[index_min]
    
        # Grab secant that is most diminished under projection
        most_diminished_secant = secants[:,index_min]

        # Calculate projection of most diminished
        proj_most_diminished_secant = np.matmul(proj,secant_projections[:,index_min])

        # Find largest coefficient for projected secant
        largest_coefficient = np.argmax(np.absolute(secant_projections[:,index_min]))
    
        # Switch columns
        proj[:,largest_coefficient] = proj[:,0]
        proj[:,0] = proj_most_diminished_secant

        # Apply modified Graham-Schmidt QR decomposition
        proj, r = np.linalg.qr(proj)

        # Shift projection
        proj[:,0] = (1-step_size)*proj_most_diminished_secant + step_size*(most_diminished_secant - proj_most_diminished_secant)
 
        # Normalize shift
        norm = np.linalg.norm(proj[:,0])
        proj[:,0] = (1/norm)*proj[:,0]

    # X-values for plotting of kappa-profile
    t = range(iterations)
    
    # Plot length of shortest projected secant as a function of iteration (to check for convergence).
    plt.plot(t,worst_proj_secants_record)
    plt.show()

    return proj


        
