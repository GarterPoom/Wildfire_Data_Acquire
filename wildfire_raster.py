import rasterio
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def read_sentinel_bands(raster_path):
    """
    Reads a Sentinel-2 raster file and returns a dictionary of its bands and its profile.

    Parameters
    ----------
    raster_path : str
        The path to the Sentinel-2 raster file to read.

    Returns
    -------
    bands : dict
        A dictionary of the bands in the raster file, with keys being the band names ('B01', 'B02', etc.)
        and values being the band data.
    profile : dict
        The rasterio profile of the raster file, which contains its geotransform, crs, and other metadata.
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
    Calculates the dNBR, NDWI, and NDVI indices from two dictionaries of Sentinel-2 bands.

    Parameters
    ----------
    pre_bands : dict
        A dictionary of Sentinel-2 bands from a pre-fire image.
    post_bands : dict
        A dictionary of Sentinel-2 bands from a post-fire image.

    Returns
    -------
    dnbr : numpy.ndarray
        The dNBR index, calculated as the difference between the pre-fire and post-fire Normalized Burn Ratios.
    ndwi : numpy.ndarray
        The NDWI index, calculated as the difference between the post-fire band 3 and band 8 values, normalized by their sum.
    ndvi : numpy.ndarray
        The NDVI index, calculated as the difference between the post-fire band 8 and band 4 values, normalized by their sum.
    """
    
    nbr_pre = (pre_bands['B12'] - pre_bands['B8A']) / (pre_bands['B12'] + pre_bands['B8A'])
    nbr_post = (post_bands['B12'] - post_bands['B8A']) / (post_bands['B12'] + post_bands['B8A'])
    dnbr = nbr_pre - nbr_post
    ndwi = (post_bands['B03'] - post_bands['B08']) / (post_bands['B03'] + post_bands['B08'])
    ndvi = (post_bands['B08'] - post_bands['B04']) / (post_bands['B08'] + post_bands['B04'])
    return dnbr, ndwi, ndvi

def create_burn_label(dnbr, ndwi, ndvi, b08):
    """
    Creates a burn label array from the dNBR, NDWI, NDVI, and band 8 values.

    Parameters
    ----------
    dnbr : numpy.ndarray
        The dNBR index, calculated as the difference between the pre-fire and post-fire Normalized Burn Ratios.
    ndwi : numpy.ndarray
        The NDWI index, calculated as the difference between the post-fire band 3 and band 8 values, normalized by their sum.
    ndvi : numpy.ndarray
        The NDVI index, calculated as the difference between the post-fire band 8 and band 4 values, normalized by their sum.
    b08 : numpy.ndarray
        The post-fire band 8 values.

    Returns
    -------
    burn_label : numpy.ndarray
        A binary array where 1 indicates a burned pixel and 0 indicates an unburned pixel.
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
    Sets up the input and output directories in the given root directory.

    Parameters
    ----------
    root_dir : str
        The root directory where the input and output directories will be created.

    Returns
    -------
    input_dir : str
        The path to the created input directory.
    output_dir : str
        The path to the created output directory.
    """

    input_dir = os.path.join(root_dir, 'input')
    output_dir = os.path.join(root_dir, 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return input_dir, output_dir

def get_tile_pairs(input_dir):
    """
    Reads all .tif files in the given input directory and groups them into pairs of pre- and post-fire images
    based on their tile IDs.

    Parameters
    ----------
    input_dir : str
        The path to the directory containing the pre- and post-fire Sentinel-2 images.

    Returns
    -------
    tile_pairs : dict
        A dictionary where the keys are the tile IDs and the values are dictionaries containing the paths to the
        pre- and post-fire images for that tile ID.
    """
    files = os.listdir(input_dir)
    tile_pairs = {}
    for file in files:
        if file.endswith('.tif'):
            tile_id = file.split('_')[0]
            if tile_id not in tile_pairs:
                tile_pairs[tile_id] = {'pre': None, 'post': None}
            file_path = os.path.join(input_dir, file)
            if 'pre' in file.lower():
                tile_pairs[tile_id]['pre'] = file_path
            elif 'post' in file.lower():
                tile_pairs[tile_id]['post'] = file_path
    return tile_pairs

def process_tile_pair(tile_id, paths, output_dir):
    """
    Processes a pair of pre- and post-fire Sentinel-2 images for a given tile ID, calculates various indices,
    and writes the output to a GeoTIFF file.

    Parameters
    ----------
    tile_id : str
        The identifier for the tile being processed.
    paths : dict
        A dictionary containing the file paths for the 'pre' and 'post' fire images.
    output_dir : str
        The directory where the output file will be saved.

    Returns
    -------
    None
    This function writes the processed data to a file and prints a message upon completion.
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
        'compress': 'lzw'
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
    Main function to process Sentinel-2 images by setting up directories, identifying tile pairs, and processing them
    in parallel using a process pool.

    Parameters
    ----------
    root_dir : str
        The root directory where the input and output directories are located.

    Returns
    -------
    None
    This function orchestrates the setup and processing of Sentinel-2 image tiles, utilizing parallel processing
    to improve efficiency.
    """
    input_dir, output_dir = setup_directories(root_dir)
    tile_pairs = get_tile_pairs(input_dir)
    
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_tile_pair, tile_id, paths, output_dir)
            for tile_id, paths in tile_pairs.items()
        ]
        
        for future in as_completed(futures):
            future.result()  # Wait for each task to complete

if __name__ == "__main__":
    root_dir = r"Raster"
    print("Processing Sentinel-2 images...")
    main(root_dir)
