import rasterio
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def read_sentinel_bands(raster_path):
    """
    Reads Sentinel-2 raster bands from a provided raster file path.

    Args:
        raster_path (str): The file path to the raster file containing Sentinel-2 bands.

    Returns:
        tuple: A dictionary with band names as keys and corresponding band data as values, 
               and a profile dictionary containing metadata of the raster file.
    """
    with rasterio.open(raster_path) as src:
        bands = {}
        for i, band_name in enumerate(['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 
                                       'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']):
            bands[band_name] = src.read(i + 1)
        profile = src.profile
    return bands, profile

def calculate_indices(pre_bands, post_bands):
    """
    Calculates dNBR (Normalized Burn Ratio), NDWI (Normalized Difference Water Index), and NDVI (Normalized Difference Vegetation Index) from Sentinel-2 bands.

    Args:
        pre_bands (dict): A dictionary with Sentinel-2 band names as keys and corresponding band data as values from the pre-fire image.
        post_bands (dict): A dictionary with Sentinel-2 band names as keys and corresponding band data as values from the post-fire image.

    Returns:
        tuple: A tuple containing dNBR, NDWI, and NDVI as 2D numpy arrays.
    """
    nbr_pre = (pre_bands['B8A'] - pre_bands['B12']) / (pre_bands['B8A'] + pre_bands['B12'])
    nbr_post = (post_bands['B8A'] - post_bands['B12']) / (post_bands['B8A'] + post_bands['B12'])
    dnbr = nbr_pre - nbr_post
    ndwi = (post_bands['B03'] - post_bands['B08']) / (post_bands['B03'] + post_bands['B08'])
    ndvi = (post_bands['B08'] - post_bands['B04']) / (post_bands['B08'] + post_bands['B04'])
    return dnbr, ndwi, ndvi

def create_burn_label(dnbr, ndwi, ndvi, b08):
    """
    Creates a burn label mask for a given set of conditions based on 
    dNBR (Normalized Burn Ratio), NDWI (Normalized Difference Water Index), 
    NDVI (Normalized Difference Vegetation Index), and the B08 band.

    Args:
        dnbr (numpy.ndarray): 2D array of dNBR values.
        ndwi (numpy.ndarray): 2D array of NDWI values.
        ndvi (numpy.ndarray): 2D array of NDVI values.
        b08 (numpy.ndarray): 2D array of B08 band values.

    Returns:
        numpy.ndarray: A binary mask where 1 indicates a burn area that meets 
        the specified thresholds, and 0 otherwise.
    """
    burn_label = np.where(
        (dnbr > 0.27) & 
        (ndwi < 0) & 
        (ndvi < 0.14) & 
        (b08 < 2500),
        1, 0
    )
    return burn_label

def setup_directories(root_dir):
    """
    Sets up the directory structure for the input and output files. Creates a folder
    named 'input' and 'output' in the given root directory. If the folders already exist,
    does nothing.

    Args:
        root_dir (str or Path): The root directory where the input and output folders will be created.

    Returns:
        tuple: A tuple containing the paths to the input and output folders.
    """
    input_dir = os.path.join(root_dir, 'input')
    output_dir = os.path.join(root_dir, 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return input_dir, output_dir

def get_tile_pairs(input_dir):
    """
    Recursively reads all .tif files in the input directory and its subdirectories.
    Groups them into pairs of pre- and post-fire images based on tile IDs.
    """
    tile_pairs = {}
    for file_path in Path(input_dir).rglob('*.tif'):
        tile_id = file_path.stem.split('_')[0]
        if tile_id not in tile_pairs:
            tile_pairs[tile_id] = {'pre': None, 'post': None}
        
        if 'pre' in file_path.stem.lower():
            tile_pairs[tile_id]['pre'] = str(file_path)
        elif 'post' in file_path.stem.lower():
            tile_pairs[tile_id]['post'] = str(file_path)
    return tile_pairs

def process_tile_pair(tile_id, paths, output_dir):
    """
    Processes a single tile pair of pre- and post-fire Sentinel-2 images.

    Args:
        tile_id (str): The ID of the tile pair.
        paths (dict): A dictionary with 'pre' and 'post' keys containing the file paths to the pre- and post-fire images.
        output_dir (str or Path): The directory where the output files will be saved.

    Returns:
        None
    """
    if paths['pre'] is None or paths['post'] is None:
        print(f"Missing pre or post image for tile {tile_id}")
        return
    
    pre_bands, pre_profile = read_sentinel_bands(paths['pre'])
    post_bands, post_profile = read_sentinel_bands(paths['post'])
    
    dnbr, ndwi, ndvi = calculate_indices(pre_bands, post_bands)
    burn_label = create_burn_label(dnbr, ndwi, ndvi, post_bands['B08'])
    
    output_profile = post_profile.copy()
    output_profile.update({
        'count': 16,
        'dtype': 'float32',
        'compress': 'zstd'  # Apply Zstandard compression
    })
    
    post_filename = os.path.basename(paths['post'])
    tile_date = post_filename.split('_')[1]
    output_filename = f"{tile_id}_{tile_date}_Train.tif"
    output_path = os.path.join(output_dir, output_filename)
    
    with rasterio.open(output_path, 'w', **output_profile) as dst:
        band_idx = 1
        for band_name in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']:
            dst.write(post_bands[band_name].astype('float32'), band_idx)
            dst.set_band_description(band_idx, band_name)
            band_idx += 1
        
        dst.write(dnbr.astype('float32'), band_idx)
        dst.set_band_description(band_idx, 'dNBR')
        band_idx += 1
        
        dst.write(ndwi.astype('float32'), band_idx)
        dst.set_band_description(band_idx, 'NDWI')
        band_idx += 1
        
        dst.write(ndvi.astype('float32'), band_idx)
        dst.set_band_description(band_idx, 'NDVI')
        band_idx += 1
        
        dst.write(burn_label.astype('float32'), band_idx)
        dst.set_band_description(band_idx, 'Burn_Label')
    
    print(f"Processed {tile_id}: Output saved as {output_filename}")

def main(root_dir):
    """
    Main entry point for processing Sentinel-2 images.

    This function sets up the directory structure, retrieves the tile pairs, and processes each tile pair
    in parallel using a ProcessPoolExecutor.

    Args:
        root_dir (str or Path): The root directory where the input and output folders will be created.

    Returns:
        None
    """
    input_dir, output_dir = setup_directories(root_dir)
    tile_pairs = get_tile_pairs(input_dir)
    
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_tile_pair, tile_id, paths, output_dir)
            for tile_id, paths in tile_pairs.items()
        ]
        
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    root_dir = r"Raster"
    print("Processing Sentinel-2 images...")
    main(root_dir)
