## -- Code to calculate the Cluster Index of an array given the binary defect array and voxel dimensions
## -- RPT, 4/25/2024
import numpy as np
import time
from tqdm import tqdm # ------ for progress bar
import concurrent.futures # -- for parallel processing
import logging
import os

def multi_which(A):
    '''Given a binary 3D array, returns all row/column/slice indices corresponding to 1's'''
    if np.isscalar(A):
        return np.where(A)[0]

    d = np.shape(A)
    T = np.where(A.flatten())[0]
    nd = len(d)

    def calculate_indices(t):
        I = np.zeros(nd, dtype=int)

        for j in range(nd-1, 0, -1):
            I[j] = t % d[j]
            t = t // d[j]

        I[0] = t
        return I

    result = np.array([calculate_indices(t)  for t in T])
    return result


def getSpherePix(vox, radius):
    '''This is the sphereGrowing algorithm. It will grow a sphere around [0,0,0] for voxels scaled 
        to voxel dimensions specified in 'vox' and keep track of all voxel intersections up to a  
        specified maximum radius 'radius'.
        Inputs: vox - 1D vector specifying voxel dimensions in [row, col, slice]
                radius - the radius at which the code will stop growing (essentially, the maximum
                radius expected from growing spheres in the CI calculation
        Outputs: pxls - a 2D Nx4 array listing all radii (column 0) and pixel indices (cols 1-3) 
    
    '''
    pxListFileName = f'{vox[0]}x{vox[1]}x{vox[2]}_{radius}.npy'
    try:
        pxls = np.load(os.path.join(os.getcwd(),pxListFileName))
        print(f'\n spherePx {pxListFileName} exists and is being loaded...')
    except:
        print(f"\n Could not find {os.path.join(os.getcwd(),pxListFileName)}. Building List of Pixels in sphere of radius {radius}...")
        radius = int(radius)
        starttime = time.time()
        vox = vox/np.min(vox)
        X,Z,Y = np.meshgrid(range(-radius,radius+1,1),range(-radius,radius+1,1),range(-radius,radius+1,1))
        pxls = np.zeros((1,4))

        for r in tqdm(np.arange(0,radius,0.01), desc="Processing",unit="iteration"):
            circle = ((X*vox[0])**2 + (Y*vox[1])**2 + (Z*vox[2])**2 <= r**2)*((X*vox[0])**2 + (Y*vox[1])**2 + (Z*vox[2])**2 > (r-0.01)**2)
            x = X[circle]
            y = Y[circle]
            z = Z[circle]
            pxls = np.vstack((pxls,np.column_stack((np.repeat(r,len(x)),x,y,z))))
        np.save(os.path.join(os.getcwd(),pxListFileName),pxls)
        print(f"Sphere Pixel List of length {pxls.shape[0]} calculation time: {np.round(time.time()-starttime,2)} seconds")
    return pxls

def px2vec(i,j,k,arrayShape):
    '''given a row/col/slice and 3D array shape, returns an integer corresponding to that index location in the array
        (the reverse of vec2px)'''
    return i + (j-1)*arrayShape[0] + (k-1)*arrayShape[0]*arrayShape[1]

def vec2px(n,arrayShape):
    '''given an integer index location and 3D array shape, returns the row/col/slice corresponding to that location
        (the reverse of px2vec)'''
    s = np.ceil(n/(arrayShape[0]*arrayShape[1]))
    n = n - (s-1)*arrayShape[1]*arrayShape[0]
    c = np.ceil(n/arrayShape[0])
    r = n- (c-1)*arrayShape[0]
    return int(r),int(c),int(s)

def getRadiiIndices(data):
    '''Given the Nx4 spherePx array, it returns all the indices at which a new radius was found
        (useful for indexing the CV calculator for loop)'''
    diffs = np.diff(data[:, 0]) > 0
    sphere_rads = np.where(diffs)[0] + 2
    sphere_rads = sphere_rads[sphere_rads > 0] - 1
    return sphere_rads

def calculate_CV(defectArrayShape,activeVoxel,defVec,spherePx):
    '''Given the defectArray shape (length 3 vector), activeVoxel (length 3 vector specifying index row/col/slice), defVec 
        (vector of defect voxel indices cast to vector by px2vec), and spherePx (the Nx4 array from spherePix), it will
        calculate that voxel's CV by increasing the radius in spherePx until less than half of the voxels in the sphere
        are defect (as calculated by intersect1d)'''
    sphereRads = getRadiiIndices(spherePx)
    sphereVec = px2vec(spherePx[:,1]+activeVoxel[0],spherePx[:,2]+activeVoxel[1],spherePx[:,3]+activeVoxel[2],defectArrayShape)
    for ii in sphereRads:
        growBreak = 0
        C = len(np.intersect1d(sphereVec[:ii],defVec))/len(sphereVec[:ii])
        if C<0.5:
            growBreak = 1
            break
    
    if growBreak == 0:
        logging.critical('--MAX RADUIS of {spherePx[sphereRads[ii],0]} REACHED--')
        raise ValueError

    return np.append(activeVoxel,spherePx[ii-1,0])

def calculate_CI(defectArray,vox=[1,1,1],Rmax=50,type='fast'):
    '''Calculates CVs for the entire defectArray.
        This is a separate function from calculate_CV so that it can use the 
        concurrent futures module for rapid calculation of CI
        Inputs: the 3D binary defectArray and the voxel dimensions
        Outputs: the CI array'''
    ## -- Create the list of sphere pixels -- ##
    spherePx = getSpherePix(vox,Rmax)

    ## -- Create the defect voxel list and vectorize it -- ##
    defList = multi_which(defectArray*1)
    defVec = px2vec(defList[:,0],defList[:,1],defList[:,2],defectArray.shape)

    ## -- slow CI calculation (doesn't use concurrent futures)-- ##
    if type == 'slow':
        start_time = time.time()
        CI = defectArray*0
        for k in tqdm(range(len(defVec)), desc="Processing",unit="iteration"):
            CI[defList[k,0],defList[k,1],defList[k,2]] = calculate_CV(defectArray.shape,defList[k,:],defVec,spherePx)[3] ## -- NEEDS FIXING -- ##
        print(f"Time to calculate slow CI array: {np.round((time.time()-start_time)/60,2)} min")

    ## -- fast CI calculation (uses concurrent futures)-- ##
    if type == 'fast':
        CIlist = []
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print('\n Generating futures list...')
            CI_futures = [executor.submit(calculate_CV,defectArray.shape,defList[k,:],defVec,spherePx) for k in tqdm(range(defList.shape[0]))]
            print('\n Calculating CI map...')
            concurrent.futures.as_completed(CI_futures)
            for f1 in tqdm(CI_futures):
                CIlist.append(f1.result())
        CIlist = np.vstack(CIlist)
        CI = np.double(defectArray*0)
        for k in range(CIlist.shape[0]):
            CI[int(CIlist[k,0]),int(CIlist[k,1]),int(CIlist[k,2])] = CIlist[k,3]*np.min(vox)
        print(f"Time to calculate fast CI array: {np.round((time.time()-start_time)/60,2)} min")

    return CI


