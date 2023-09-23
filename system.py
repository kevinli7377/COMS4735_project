import cv2 as cv
import numpy as np
import os

#-------Display image method----------
def show(img):
    '''Show image using cv'''
    cv.imshow("Img",img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

#-------Helper methods for main--------
def remove_small_areas(image):
    '''Return largest area for a given binary image.'''
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(image)

    largest_label = 1+np.argmax(stats[1:,cv.CC_STAT_AREA])
    img_largest = np.zeros_like(image)
    img_largest[labels==largest_label] = 255

    return img_largest

def liquid_segmentation(img):
    '''Use k-means clustering to segment colored liquid from surroundings,
        assuming that the color of the liquid is dark and that the 
        vessel is clear'''

    lab_image = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    pixels = lab_image.reshape((-1,3)) # Reshape to 2D array of pixel
    pixels = np.float32(pixels) # Convert pixel values to floats

    # Criteria for kmeans clustering algorithm
    # (Criteria, iterations, epsilon convergence accuracy)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)

    # Run k-means clustering algorithm
    # Form 2 clusters -> assuming that the container is transparent 
    # and the background is white

    downsampled_img = downsample_image(lab_image)
    downsampled_pixels = downsampled_img.reshape((-1,3)) # Reshape to 2D array of pixel
    downsampled_pixels = np.float32(downsampled_pixels) # Convert pixel values to floats

    _,labels,centers = cv.kmeans(downsampled_pixels,2,None,criteria,10,cv.KMEANS_PP_CENTERS)
    
    centers = np.uint8(centers)

    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(downsampled_img.shape)

    # Get the colors - there should be a total of 2
    unique_colors = np.unique(segmented_image.reshape(-1, segmented_image.shape[2]), axis=0)

    # Find liquid color, assuming that the color with the higher value (darker) is the liquid
    liquid_color = unique_colors[np.argmax(np.mean(unique_colors, axis=1))]


    binary_image = np.zeros(segmented_image.shape[:2], dtype=np.uint8)
    binary_image[(segmented_image == liquid_color).all(axis=2)] = 255 # determine pixels with the liquid color
    
    liquid_image = remove_small_areas(cv.bitwise_not(binary_image))

    return liquid_image

def cfactor(pixelV,realV):
    '''Returns conversion factor between pixel and real liquid volumes'''
    return pixelV/realV

def downsample_image(img, max_dimension=1000):
    '''
    Downsample image. Used in k-means sampling based image segmentation.
    '''
    max_original_dimension = max(img.shape[:2])
    scale_factor = max_dimension / max_original_dimension
    scale_factor = min(1.0, scale_factor)  # Ensure scale_factor is not greater than 1

    return cv.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)

#-------Volume estimation functions-------
def vol_calc_disc(img):
    '''Determine volume of liquid using a binary silloheutte
    of the liquid (elevation view). 
    
    img should be binary.

    The disk method is used to determine
    the overall volume. Each disk width is equivalent 1 pixel. Each row
    of pixels is thus treated as a cylindrical disk. '''

    total_volume = 0

    for y in range(img.shape[0]-1,-1,-1): # Iterate through each row of the image
        row = img[y,:]
        ones_indices = np.where(row==255)[0] # Find the pixels that are white - i.e. represent liquid

        if ones_indices.size > 0: # If there are white pixels in a given row
            # Determine the leftmost and rightmost pixel indices and determine the radius
            # Then use pi*r**2*h to determine the volume of each disk
            height = 1  # Since one pixel height rows
            leftmost = ones_indices[0] 
            rightmost = ones_indices[-1]
            radius_pixels = (rightmost-leftmost)/2
            radius = radius_pixels
            disk_volume = np.pi * (radius**2)*height
            total_volume += disk_volume
    return total_volume
    
def find_closest_references(ref_pvol_cfs, query_pvol):
    '''
    Find the conversion factors for reference images with liquid 
    volumes below and above the liquid's cubic pixel volume

    ref_pvol_cfs = List of ref. cubic volumes and cfs
    query_pvol = Query pixel volume
    '''

    ref_volumes_pixel = [volume_pixel for volume_pixel, cf in ref_pvol_cfs]
    ref_volumes_pixel.sort()

    below = None
    above = None

    if query_pvol <= ref_volumes_pixel[0]:
        above = ref_pvol_cfs[0]
    elif query_pvol >= ref_volumes_pixel[-1]:
        below = ref_pvol_cfs[-1]
    else:
        for idx, volume_pixel in enumerate(ref_volumes_pixel):
            if query_pvol >= volume_pixel:
                below_idx = idx
                below = ref_pvol_cfs[below_idx]
            if query_pvol <= volume_pixel:
                above_idx = idx
                above = ref_pvol_cfs[above_idx]
                break

    return below, above

def ref_factors(ref_imgs,ref_volumes):
    '''
    Compute conversion factors for a list of ref images and their 
    corresponding ground truth volumes
    '''

    conversion_factors = []

    for ref_img, ref_volume in zip(ref_imgs,ref_volumes):
        ref_volume_pixel = vol_calc_disc(ref_img)
        cf = cfactor(ref_volume_pixel,ref_volume)
        conversion_factors.append((ref_volume_pixel,cf))

    return conversion_factors

def interpolate_cf(query_pvol,above,below):
    '''
    Interpolate the cf for the query pixel volume based
    on the conversion factors for reference bolumes
    below and above.
    '''
    if below is None: 
        # If there is no reference below the query volume, use the closts reference's cf
        return above[1]
    else:
        weight = (query_pvol-below[0])/(above[0]-below[0])
        interpolated_cf = (1-weight)*below[1] + weight*(above[1])
    return interpolated_cf

#-----------------------------Main method----------------------------
#--------------------------------------------------------------------
def main(ref_img_paths,ref_volumes,q_fpath,q_volume,interpolation=True):
    '''Function takes in the paths to reference and query images
        to compute the volume of the liquid being held within in the query image.
        Reference image volumes and needed for computation of conversion factors.
        Query volume is alsoo used to assess system accuracy.
        
        ref_img_paths = list of filepaths to references iamges
        ref_volumes = list of volumes corresponding to each image in ref_imgs
        q_fpath = filepath to query image as a string
        q_volume = volume of query as a numeric value
        interpolation = whether interpolation should be used or not
        '''
    
    #-----------------------1. Reference image processing---------------------
    #-------------------------------------------------------------------------

    # Read in all images and filter them
    ref_imgs = [cv.bilateralFilter(cv.imread(os.path.abspath(path)),9,75,75)for path in ref_img_paths]
    ref_liquid_imgs = [liquid_segmentation(img) for img in ref_imgs] # Segment out liquid
    conversion_factors = ref_factors(ref_liquid_imgs,ref_volumes) # Compute cubic pixel volumes and conversion factors
    conversion_factors = sorted(conversion_factors,key=lambda x:x[0]) # Sort the list by cubic-pixel volume

    #-----------------------2. Query image processing--------------------------
    #--------------------------------------------------------------------------
    q_img = cv.imread(os.path.abspath(q_fpath))
    q_filtered_img = cv.bilateralFilter(q_img,9,75,75)
    q_liquid_img = liquid_segmentation(q_filtered_img)
    q_vol = vol_calc_disc(q_liquid_img)
    vol_cf_below, vol_cf_above = find_closest_references(conversion_factors,q_vol)

    if interpolation and vol_cf_below is not None and vol_cf_above is not None:
        f = interpolate_cf(q_vol, vol_cf_above, vol_cf_below)
    else:
        if vol_cf_below is None:
            f = vol_cf_above[1]
        else:
            f = vol_cf_below[1]

    q_vol_estimate_pixel = vol_calc_disc(q_liquid_img)
    estimation = q_vol_estimate_pixel/f

    #-----------------------3.Display results----------------------------------
    #--------------------------------------------------------------------------
    relative_diff = -(1-(estimation/q_volume))
    abs_diff = abs(estimation-q_volume)
    print('Query true volume: ',q_volume)
    print('Estimated volume: ',estimation)
    print('Relative Difference (%): ', relative_diff)
    print('Absolute Difference (ml):', abs_diff)
    return estimation,relative_diff, abs_diff

#-----------------------------Main for app---------------------------
#--------------------------------------------------------------------
def app_main(ref_img_paths,ref_volumes,q_fpath,q_volume,interpolation=True):
    ''' Application version of main()

        Function takes in the paths to reference and query images
        to compute the volume of the liquid being held within in the query image.
        Reference image volumes and needed for computation of conversion factors.
        Query volume is alsoo used to assess system accuracy.
        
        ref_img_paths = list of filepaths to references iamges
        ref_volumes = list of volumes corresponding to each image in ref_imgs
        q_fpath = filepath to query image as a string
        q_volume = volume of query as a numeric value
        interpolation = whether interpolation should be used or not
        '''
    
    #-----------------------1. Reference image processing---------------------
    #-------------------------------------------------------------------------

    # Read in all images and filter them
    ref_imgs = [cv.bilateralFilter(cv.imread(os.path.abspath(path)),9,75,75)for path in ref_img_paths]
    ref_liquid_imgs = [liquid_segmentation(img) for img in ref_imgs] # Segment out liquid

    conversion_factors = ref_factors(ref_liquid_imgs,ref_volumes) # Compute cubic pixel volumes and conversion factors
    conversion_factors = sorted(conversion_factors,key=lambda x:x[0])# Sort the list by cubic-pixel volume

    #-----------------------2. Query image processing--------------------------
    #--------------------------------------------------------------------------
    q_img = cv.imread(os.path.abspath(q_fpath))
    q_filtered_img = cv.bilateralFilter(q_img,9,75,75)
    q_liquid_img = liquid_segmentation(q_filtered_img)
    q_vol = vol_calc_disc(q_liquid_img)
    vol_cf_below, vol_cf_above = find_closest_references(conversion_factors,q_vol)

    if interpolation and vol_cf_below is not None and vol_cf_above is not None:
        f = interpolate_cf(q_vol, vol_cf_above, vol_cf_below)
    else:
        if vol_cf_below is None:
            f = vol_cf_above[1]
        else:
            f = vol_cf_below[1]

    q_vol_estimate_pixel = vol_calc_disc(q_liquid_img)
    estimation = q_vol_estimate_pixel/f

    #-----------------------3.Display results----------------------------------
    #--------------------------------------------------------------------------
    relative_diff = -(1-(estimation/q_volume))
    abs_diff = abs(estimation-q_volume)

    return ref_liquid_imgs,q_liquid_img,estimation,relative_diff,abs_diff


#--------------------------Simulation method-------------------------
#--------------------------------------------------------------------
def reference_processing(ref_img_paths,ref_volumes):
    '''
    Processing reference images and computing conversion factors
    '''
    ref_imgs = [cv.imread(os.path.abspath(path)) for path in ref_img_paths]
    ref_liquid_imgs = [liquid_segmentation(img) for img in ref_imgs] # Segment out liquid
    conversion_factors = ref_factors(ref_liquid_imgs,ref_volumes) # Compute cubic pixel volumes and conversion factors
    return conversion_factors

def query_processing(q_fpath,q_volume,conversion_factors,interpolation=True):
    '''
    Processing query
    '''
    q_img = cv.imread(os.path.abspath(q_fpath))
    q_filtered_img = cv.bilateralFilter(q_img,9,75,75)
    q_liquid_img = liquid_segmentation(q_filtered_img)
    q_vol = vol_calc_disc(q_liquid_img)

    vol_cf_below, vol_cf_above = find_closest_references(conversion_factors,q_vol)

    if interpolation and vol_cf_below is not None and vol_cf_above is not None:
        f = interpolate_cf(q_vol, vol_cf_above, vol_cf_below)
    else:
        if vol_cf_below is None:
            f = vol_cf_above[1]
        else:
            f = vol_cf_below[1]

    q_vol = vol_calc_disc(q_liquid_img)
    estimation = q_vol/f
    return estimation

def run_simulation(containers):
    '''
    Function to run simulation. Results outputted to
    simulation_results.txt in the same directory.

    Function takes in a dictionary as its input:

    For each k:v pair
        k = container type
        v = Tuple(ref_img_paths,ref_volumes,q_img_paths,q_volumes)

    '''
    with open("simulation_results.txt", "w") as outfile:
        for container_type, (ref_img_paths, ref_volumes, query_img_paths, query_volumes) in containers.items():
            
            references = reference_processing(ref_img_paths,ref_volumes) 
            # Compute reference conversion factors once for each container type for efficiency

            for q_fpath, q_volume in zip(query_img_paths, query_volumes):
                outfile.write(f"Container type: {container_type}\n")
                outfile.write(f"Query: {q_fpath}\n")
                outfile.write(f"Ground truth volume: {q_volume}\n")

                for interpolation in [True, False]: # Run iteration on both systems: interpolation and closest method
                    method = 'Interpolation' if interpolation else 'Closest reference'
                    outfile.write(f"{method} method\n")
                    estimation = query_processing(q_fpath, q_volume, references,interpolation)
                    error = abs(estimation - q_volume) / q_volume

                    outfile.write(f"Estimated volume: {estimation}\n")
                    outfile.write(f"Difference: {estimation-q_volume}\n")
                    outfile.write(f"Relative error: {error:.2%}\n")
                outfile.write("\n")

            outfile.write("---------------------------------------------------\n")

if __name__ == '__main__':

    # Image paths for simulation
    containers = {
        "juice":(['./final_images/juice/50.jpg',
                  './final_images/juice/200.jpg',
                  './final_images/juice/350.jpg'
                  ],
                  [50,200,350],
                  ['./final_images/juice/30.jpg',
                   './final_images/juice/100.jpg',
                   './final_images/juice/150.jpg',
                   './final_images/juice/250.jpg',
                   './final_images/juice/300.jpg',
                   './final_images/juice/400.jpg',
                   './final_images/juice/450.jpg'
                   ],
                   [30,100,150,250,300,400,450]
                   ),
        "neck":(['./final_images/neck/100.jpg',
                 './final_images/neck/300.jpg',
                 './final_images/neck/500.jpg',
                 ],
                 [100,300,500],
                 ['./final_images/neck/50.jpg',
                  './final_images/neck/150.jpg',
                  './final_images/neck/180.jpg',
                  './final_images/neck/200.jpg',
                  './final_images/neck/400.jpg',
                  './final_images/neck/520.jpg'
                 ],
                 [50,150,180,200,400,520]
                 ),
        "wine":(['./final_images/wine/50.jpg',
                 './final_images/wine/150.jpg',
                 './final_images/wine/300.jpg'
                 ],
                 [50,150,300],
                 ['./final_images/wine/30.jpg',
                  './final_images/wine/100.jpg',
                  './final_images/wine/200.jpg',
                  './final_images/wine/250.jpg'
                 ],
                 [30,100,200,250]
                )
    }

    run_simulation(containers) # Running simulation
