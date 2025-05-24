"""
Point cloud generation and volume estimation utilities.

This module provides classes for generating point clouds from RGB+depth data
and calculating volumes from segmented point cloud regions.
"""

import numpy as np
# import open3d as o3d
from typing import Tuple, List

class PointCloudGenerator:
    """
    Generates point clouds from RGB images and depth maps, and calculates volumes
    from segmented regions.
    """
    
    def __init__(self, focal_length_x: float = 470.4, focal_length_y: float = 470.4, mode: str = "convex_hull"):
        """
        Initialize the point cloud generator.
        
        Args:
            focal_length_x: Camera focal length along x-axis
            focal_length_y: Camera focal length along y-axis
            mode: Volume calculation mode ("convex_hull" or "pca")
        """
        self.focal_length_x = focal_length_x
        self.focal_length_y = focal_length_y
        self.mode = mode
    
    def generate_point_cloud(self, rgb_image: np.ndarray, depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a point cloud from RGB image and depth map.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            depth_map: Depth map as numpy array (H, W)
            
        Returns:
            Tuple of (points, colors) where:
            - points: 3D coordinates as numpy array (N, 3)
            - colors: RGB colors as numpy array (N, 3) normalized to [0, 1]
        """
        height, width = depth_map.shape
        
        # Generate mesh grid and calculate point cloud coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / self.focal_length_x
        y = (y - height / 2) / self.focal_length_y
        z = depth_map
        
        # Calculate 3D points
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        
        # Get colors from RGB image
        colors = rgb_image.reshape(-1, 3) / 255.0
        
        return points, colors
    
    def apply_mask_to_pointcloud(self, points: np.ndarray, colors: np.ndarray, 
                                mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a binary mask to filter point cloud points.
        
        Args:
            points: 3D points array (N, 3)
            colors: Color array (N, 3)
            mask: Binary mask array (H, W) with values 0 or 255
            
        Returns:
            Tuple of filtered (points, colors)
        """
        # Flatten mask and convert to boolean
        mask_flat = (mask.flatten() > 0)
        
        # Filter points and colors
        filtered_points = points[mask_flat]
        filtered_colors = colors[mask_flat]
        
        return filtered_points, filtered_colors
    
    def calculate_volume_from_pointcloud(self, points: np.ndarray) -> float:
        """
        Calculate volume from a set of 3D points using the specified mode.
        
        Args:
            points: 3D points array (N, 3)
            
        Returns:
            Volume in cubic units
        """
        if len(points) < 4:  # Need at least 4 points for meaningful volume calculation
            return 0.0
        
        if self.mode == "pca":
            return self._calculate_volume_pca(points)
        elif self.mode == "convex_hull":
            return self._calculate_volume_convex_hull(points)
        else:
            # Default to convex hull
            return self._calculate_volume_convex_hull(points)
    
    def _calculate_volume_pca(self, points: np.ndarray) -> float:
        """
        Calculate volume using PCA-based ellipsoid estimation.
        
        This method treats the point cloud as an ellipsoid and estimates its volume
        using the principal components of the data distribution.
        
        Args:
            points: 3D points array (N, 3)
            
        Returns:
            Volume in cubic units
        """
        try:
            # Center the data around the mean
            X = points - points.mean(axis=0)
            
            # Get singular values from SVD (related to principal components)
            pcs = np.linalg.svdvals(X)
            
            # Calculate standard deviations along each principal component
            rs = pcs / np.sqrt(0.5*X.shape[0] - 1)
            
            # Ensure we have at least 3 components for 3D volume calculation
            if len(rs) >= 3:
                # Volume of ellipsoid: V = (4/3) * π * a * b * c
                volume = (4/3) * np.pi * rs[0] * rs[1] * rs[2]
            elif len(rs) == 2:
                # For 2D case, use the provided approximation formula
                volume = (4/3) * np.pi * rs[0] * rs[1] * (rs[0] + rs[1]) / 2
            elif len(rs) == 1:
                # For 1D case, treat as sphere with radius rs[0]
                volume = (4/3) * np.pi * rs[0]**3
            else:
                return 0.0
            
            return float(volume)
            
        except Exception as e:
            # Fallback to bounding box volume if PCA fails
            return self._calculate_volume_bounding_box(points)
    
    def _calculate_volume_convex_hull(self, points: np.ndarray) -> float:
        """
        Calculate volume using convex hull method.
        
        Args:
            points: 3D points array (N, 3)
            
        Returns:
            Volume in cubic units
        """
        try:
            # Create point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Compute convex hull
            hull, _ = pcd.compute_convex_hull()
            
            # Calculate volume
            volume = hull.get_volume()
            
            return float(volume)
            
        except Exception as e:
            # Fallback to bounding box volume if convex hull fails
            return self._calculate_volume_bounding_box(points)
    
    def _calculate_volume_bounding_box(self, points: np.ndarray) -> float:
        """
        Calculate volume using bounding box method (fallback).
        
        Args:
            points: 3D points array (N, 3)
            
        Returns:
            Volume in cubic units
        """
        if len(points) > 0:
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            dims = max_coords - min_coords
            return float(np.prod(dims))
        return 0.0
    
    def calculate_volumes_from_masks(self, rgb_image: np.ndarray, depth_map: np.ndarray, 
                                   masks: List[np.ndarray]) -> List[float]:
        """
        Calculate volumes for multiple segmentation masks.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            depth_map: Depth map as numpy array (H, W)
            masks: List of binary masks (each H, W with values 0 or 255)
            
        Returns:
            List of calculated volumes for each mask
        """
        # Generate full point cloud
        points, colors = self.generate_point_cloud(rgb_image, depth_map)
        
        volumes = []
        for mask in masks:
            # Apply mask to get segmented points
            filtered_points, _ = self.apply_mask_to_pointcloud(points, colors, mask)
            
            # Calculate volume for this segment
            volume = self.calculate_volume_from_pointcloud(filtered_points)
            volumes.append(volume)
        
        return volumes
    
    def set_mode(self, mode: str):
        """
        Set the volume calculation mode.
        
        Args:
            mode: Volume calculation mode ("convex_hull" or "pca")
        """
        if mode in ["convex_hull", "pca"]:
            self.mode = mode
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'convex_hull' or 'pca'")
    
    def get_mode(self) -> str:
        """
        Get the current volume calculation mode.
        
        Returns:
            Current mode string
        """
        return self.mode