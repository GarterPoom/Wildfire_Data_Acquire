import rasterio
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from rasterio.windows import Window
from shapely.geometry import box
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentinelProcessor:
    def __init__(self, root_dir, tile_size=1024):
        """
        Initialize the Sentinel processor with root directory and tile size.

        The root directory should contain the following subdirectories:

        - input: containing the Sentinel-2 images to process
        - output: containing the processed images

        Args:
            root_dir (str): Root directory for processing
            tile_size (int): Size of tiles to chop images into (default: 1024)
        """
        self.root_dir = Path(root_dir).resolve()  # Get absolute path
        self.tile_size = tile_size
        self.input_dir, self.output_dir = self._setup_directories()

    def _setup_directories(self):
        """
        Set up input and output directories.

        The input directory should contain the Sentinel-2 images to process, and
        the output directory will contain the processed images.

        Returns:
            input_dir (Path): Path to the input directory
            output_dir (Path): Path to the output directory
        """
        input_dir = self.root_dir / 'input'
        output_dir = self.root_dir / 'output'
        
        # Create directories if they don't exist
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return input_dir, output_dir


    @staticmethod
    def read_sentinel_bands(raster_path):
        """
        Read Sentinel-2 bands from raster file.

        Args:
            raster_path (Path): Path to the raster file

        Returns:
            bands (dict): Dictionary with band names as keys and band data as values
            profile (dict): Rasterio profile for the raster file
        """
        with rasterio.open(raster_path) as src:
            # Get band names from the raster file
            band_names = [
                'B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'
            ]
            # Read the bands and convert to float32
            bands = {
                band_name: src.read(i + 1).astype(np.float32)
                for i, band_name in enumerate(band_names)
            }
            # Update the profile to float32
            profile = src.profile.copy()
            profile.update(dtype='float32')
        return bands, profile


    @staticmethod
    def calculate_indices(pre_bands, post_bands):
        """
        Calculate various spectral indices, ensuring divisions by zero are handled safely.

        Args:
            pre_bands (dict): Dictionary of pre-fire bands.
            post_bands (dict): Dictionary of post-fire bands.

        Returns:
            tuple: dNBR, NDWI, and NDVI indices.
        """
        def safe_index(band1, band2):
            """
            Calculate index safely, handling division by zero.

            This function is used to calculate various spectral indices
            (e.g. NBR, NDWI, NDVI) by performing a division between two
            bands. To handle division by zero, the function returns NaN
            for any pixels where the denominator is zero.

            Args:
                band1 (np.ndarray): First band data.
                band2 (np.ndarray): Second band data.

            Returns:
                np.ndarray: Calculated index with NaN for division by zero results.
            """
            # Calculate the denominator
            denominator = band1 + band2
            # Create a mask for pixels where the denominator is non-zero
            mask = denominator != 0
            # Initialize the result array with zeros
            result = np.zeros_like(denominator, dtype=np.float32)
            # Calculate the index for pixels where the denominator is non-zero
            result[mask] = (band1[mask] - band2[mask]) / denominator[mask]
            # Set pixels where the denominator is zero to NaN
            result[~mask] = np.nan
            return result

        # Calculate pre and post-fire Normalized Burn Ratio (NBR)
        nbr_pre = safe_index(pre_bands['B8A'], pre_bands['B12'])
        nbr_post = safe_index(post_bands['B8A'], post_bands['B12'])
        # Calculate dNBR (difference in NBR)
        dnbr = nbr_pre - nbr_post
        # Calculate Normalized Difference Water Index (NDWI)
        ndwi = safe_index(post_bands['B03'], post_bands['B08'])
        # Calculate Normalized Difference Vegetation Index (NDVI)
        ndvi = safe_index(post_bands['B08'], post_bands['B04'])
        
        return dnbr, ndwi, ndvi

    @staticmethod
    def create_burn_label(dnbr, ndwi, ndvi, b08):
        """
        Create burn label mask based on spectral indices.

        Parameters
        ----------
        dnbr : np.ndarray
            Difference in Normalized Burn Ratio (dNBR)
        ndwi : np.ndarray
            Normalized Difference Water Index (NDWI)
        ndvi : np.ndarray
            Normalized Difference Vegetation Index (NDVI)
        b08 : np.ndarray
            Band 8 (nir) data

        Returns
        -------
        np.ndarray
            Burn label mask with values of 0 (no burn) or 1 (burn)
        """
        # Handle NaN values in the input arrays
        valid_mask = ~(np.isnan(dnbr) | np.isnan(ndwi) | np.isnan(ndvi) | np.isnan(b08))
        
        # Initialize the result array with zeros
        result = np.zeros_like(dnbr, dtype=np.float32)
        # Create burn label mask based on spectral indices
        result[valid_mask] = np.where(
            # dNBR > 0.27: high burn probability
            (dnbr[valid_mask] > 0.27) & 
            # ndwi < 0: low water content
            (ndwi[valid_mask] < 0) & 
            # ndvi < 0.14: low vegetation
            (ndvi[valid_mask] < 0.14) & 
            # b08 < 2500: low nir reflectance
            (b08[valid_mask] < 2500),
            1, 0
        )
        return result

    def chop_and_save_tiles(self, raster_data, meta, output_path, original_filename):
        """
        Chop raster data into smaller tiles and save them.

        This function handles both small and large raster images. If the image
        is smaller than or equal to the tile size, the entire image is saved
        as a single tile. Otherwise, the image is divided into smaller tiles
        and each tile is saved individually.

        Args:
            raster_data (dict): Dictionary containing band data.
            meta (dict): Metadata for the raster, including dimensions.
            output_path (Path): Directory path to save the tiles.
            original_filename (str): Base filename to use for the saved tiles.
        """
        try:
            # Check if the image is small enough to be saved as a single tile
            if meta['width'] <= self.tile_size and meta['height'] <= self.tile_size:
                # Save the entire image as a single tile
                self._save_single_tile(raster_data, meta, output_path, original_filename)
                return

            # For larger images, process and save multiple tiles
            self._process_tiles(raster_data, meta, output_path, original_filename)

        except Exception as e:
            # Log an error message if tile saving fails
            logger.error(f"Failed to save tiles for {original_filename}: {str(e)}")
            raise

    def _save_single_tile(self, raster_data, meta, output_path, filename):
        """
        Save a single tile when image is smaller than tile size.

        This function is used when the image is small enough to be saved
        as a single tile. It creates the output directory if it does not
        exist and saves the tile as a GeoTIFF file.

        Args:
            raster_data (dict): Dictionary containing band data.
            meta (dict): Metadata for the raster, including dimensions.
            output_path (Path): Directory path to save the tile.
            filename (str): Base filename to use for the saved tile.
        """
        try:
            # Create the output directory if it does not exist
            output_path.mkdir(parents=True, exist_ok=True)
            tile_path = output_path / f"{filename}.tif"
            
            # Save the tile as a GeoTIFF file
            with rasterio.open(tile_path, "w", **meta) as dst:
                # Loop through each band and write it to the file
                for band_idx, (band_name, band_data) in enumerate(raster_data.items(), 1):
                    dst.write(band_data.astype(meta['dtype']), band_idx)
                    # Set the band description to the band name
                    dst.set_band_description(band_idx, band_name)
            
            # Log a message indicating that the tile was saved
            logger.info(f"Saved entire image {filename} to {output_path}")
            
        except Exception as e:
            # Log an error message if saving the tile fails
            logger.error(f"Error saving single tile {filename}: {str(e)}")
            raise

    def _process_tiles(self, raster_data, meta, output_path, filename):
        """
        Process and save multiple tiles for larger images.

        This function is used to divide a larger image into smaller tiles
        and save each tile as a separate GeoTIFF file. The tiles are stored
        in subdirectories named "AreaXY", where X and Y are zero-padded numbers
        indicating the row and column of the tile, respectively.

        Args:
            raster_data (dict): Dictionary containing band data.
            meta (dict): Metadata for the raster, including dimensions.
            output_path (Path): Directory path to save the tiles.
            filename (str): Base filename to use for the saved tiles.
        """
        tile_width = (meta['width'] + self.tile_size - 1) // self.tile_size
        tile_height = (meta['height'] + self.tile_size - 1) // self.tile_size
        
        for row in range(tile_height):
            for col in range(tile_width):
                window = Window(
                    col * self.tile_size,
                    row * self.tile_size,
                    min(self.tile_size, meta['width'] - col * self.tile_size),
                    min(self.tile_size, meta['height'] - row * self.tile_size)
                )
                
                # Extract tile data for all bands
                tile_data = {
                    band_name: band_data[
                        window.row_off:window.row_off + window.height,
                        window.col_off:window.col_off + window.width
                    ]
                    for band_name, band_data in raster_data.items()
                }
                
                if any(np.any(data) for data in tile_data.values()):
                    self._save_tile(
                        tile_data, meta, output_path, filename,
                        row, col, window
                    )

    def _save_tile(self, tile_data, meta, output_path, filename, row, col, window):
        """
        Save individual tile with proper metadata.

        This function takes the tile data and metadata, and saves the tile as a GeoTIFF file
        in the proper area folder. The file name is based on the original filename, and the
        area folder is named as "AreaXY", where X and Y are zero-padded numbers indicating
        the row and column of the tile, respectively.

        Args:
            tile_data (dict): Dictionary containing band data for the tile.
            meta (dict): Metadata for the raster, including dimensions.
            output_path (Path): Directory path to save the tile.
            filename (str): Base filename to use for the saved tile.
            row (int): Row number of the tile.
            col (int): Column number of the tile.
            window (Window): Window object defining the area of the tile.
        """
        try:
            # Create the complete path for the area folder with zero-padded numbers
            area_folder = output_path / f"Area{row:02d}{col:02d}"
            area_folder.mkdir(parents=True, exist_ok=True)
            
            tile_path = area_folder / f"{filename}.tif"
            
            # Update metadata for tile
            tile_meta = meta.copy()
            tile_meta.update({
                "height": window.height,
                "width": window.width,
                # Calculate the new transform for the window
                "transform": rasterio.transform.from_origin(
                    meta['transform'].c + window.col_off * meta['transform'].a,
                    meta['transform'].f + window.row_off * meta['transform'].e,
                    meta['transform'].a,
                    meta['transform'].e
                )
            })
            
            with rasterio.open(tile_path, "w", **tile_meta) as dst:
                for band_idx, (band_name, band_data) in enumerate(tile_data.items(), 1):
                    dst.write(band_data.astype(tile_meta['dtype']), band_idx)
                    # Set the band description to the band name
                    dst.set_band_description(band_idx, band_name)
            
            logger.info(f"Saved tile {filename} to {area_folder}")
            
        except Exception as e:
            logger.error(f"Error saving tile to {area_folder}: {str(e)}")
            raise

    def process_tile_pair(self, tile_id, paths):
        """
        Process a pair of pre/post-fire images.

        Args:
            tile_id (str): Unique identifier for the tile.
            paths (dict): Paths to the pre and post fire images.

        Returns:
            None
        """
        if not paths['pre'] or not paths['post']:
            logger.warning(f"Missing pre or post image for tile {tile_id}")
            return

        try:
            # Read bands and calculate indices
            pre_bands, pre_profile = self.read_sentinel_bands(paths['pre'])
            post_bands, post_profile = self.read_sentinel_bands(paths['post'])
            
            # Calculate indices
            dnbr, ndwi, ndvi = self.calculate_indices(pre_bands, post_bands)
            
            # Create burn label
            burn_label = self.create_burn_label(dnbr, ndwi, ndvi, post_bands['B08'])

            # Combine all bands and indices
            combined_data = {
                **post_bands,
                'dNBR': dnbr,
                'NDWI': ndwi,
                'NDVI': ndvi,
                'Burn_Label': burn_label
            }

            # Update profile for output
            output_profile = post_profile.copy()
            output_profile.update({
                'count': len(combined_data),
                'dtype': 'float32',
                'nodata': np.nan
            })

            # Generate output path and save tiles
            post_filename = Path(paths['post']).stem
            tile_date = post_filename.split('_')[1]
            output_filename = f"{tile_id}_{tile_date}_Train"
            output_path = self.output_dir / tile_id
            
            self.chop_and_save_tiles(combined_data, output_profile, output_path, output_filename)

        except Exception as e:
            logger.error(f"Error processing tile pair {tile_id}: {str(e)}")
            raise

    def get_tile_pairs(self):
        """Get pairs of pre/post-fire images from input directory.

        Returns:
            dict: Dictionary with tile IDs as keys and dictionaries with 'pre' and 'post' keys
                  as values. The values are the file paths to the pre and post fire images.
        """
        tile_pairs = {}

        try:
            # Iterate over all files in the input directory
            for file_path in self.input_dir.rglob('*.tif'):
                tile_id = file_path.stem.split('_')[0]
                if tile_id not in tile_pairs:
                    # Initialize the tile pair dictionary if it does not exist
                    tile_pairs[tile_id] = {'pre': None, 'post': None}
                
                # Check if the file is a pre or post fire image
                if 'pre' in file_path.stem.lower():
                    tile_pairs[tile_id]['pre'] = str(file_path)
                elif 'post' in file_path.stem.lower():
                    tile_pairs[tile_id]['post'] = str(file_path)
            
            return tile_pairs
            
        except Exception as e:
            logger.error(f"Error getting tile pairs: {str(e)}")
            raise

    def process_all(self, max_workers=None):
        """
        Process all tile pairs in parallel.

        :param max_workers: The maximum number of worker processes to use.
        """
        try:
            # Get all tile pairs from the input directory
            tile_pairs = self.get_tile_pairs()

            if not tile_pairs:
                logger.warning("No tile pairs found in input directory")
                return

            logger.info(f"Found {len(tile_pairs)} tile pairs to process")

            # Process each tile pair in parallel using a ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit each tile pair to a worker process
                futures = [
                    executor.submit(self.process_tile_pair, tile_id, paths)
                    for tile_id, paths in tile_pairs.items()
                ]

                # Wait for each worker process to complete and collect the results
                for future in as_completed(futures):
                    try:
                        # If a worker process raises an exception, log the error
                        future.result()
                    except Exception as e:
                        logger.error(f"Error in worker process: {str(e)}")

        except Exception as e:
            logger.error(f"Error in process_all: {str(e)}")
            raise

def main() -> None:
    """Main entry point.

    This function is the main entry point for the script. It sets up the
    SentinelProcessor object with the root directory and tile size, and
    then calls the process_all method to start processing all tile pairs
    in parallel.

    """
    try:
        # Use absolute path for root directory
        root_dir = Path("Raster").resolve()
        # Set up the SentinelProcessor object
        processor = SentinelProcessor(root_dir, tile_size=2048)
        # Log a message to indicate that processing has started
        logger.info(f"Processing Sentinel-2 images in {root_dir}...")
        # Process all tile pairs in parallel
        processor.process_all()
        # Log a message to indicate that processing has completed
        logger.info("Processing completed successfully")
        
    except Exception as e:
        # Log a fatal error if an exception occurs
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()