import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


def find_image_in_larger(small_img, large_img, search_region=None):
    """
    Find a smaller image within a larger one using SIFT feature matching
    Args:
        small_img: The smaller image we're looking for (numpy array)
        large_img: The larger image to search in (numpy array)
        search_region: Optional (x, y, radius) to constrain the search area
    Returns:
        (x, y, w, h): Coordinates and dimensions of the found region, or None if not found
    """
    if search_region:
        x_center, y_center, radius = search_region
        x1 = max(0, x_center - radius)
        y1 = max(0, y_center - radius)
        x2 = min(large_img.shape[1], x_center + radius)
        y2 = min(large_img.shape[0], y_center + radius)
        large_img = large_img[y1:y2, x1:x2]
        if large_img.size == 0:
            return None

    small_gray = cv2.equalizeHist(cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY))
    large_gray = cv2.equalizeHist(cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY))

    sift = cv2.SIFT_create()
    kp_small, desc_small = sift.detectAndCompute(small_gray, None)
    kp_large, desc_large = sift.detectAndCompute(large_gray, None)
    
    if desc_small is None or desc_large is None:
        return None

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), 
        dict(checks=50))
    matches = flann.knnMatch(desc_small, desc_large, k=2)

    good_matches = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.75 * n.distance]

    src_pts = np.float32([kp_small[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_large[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = small_gray.shape
    
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    
    x_coords = [pt[0][0] for pt in dst]
    y_coords = [pt[0][1] for pt in dst]
    x, y = min(x_coords), min(y_coords)
    w, h = max(x_coords) - x, max(y_coords) - y
    
    if search_region:
        x += x1
        y += y1
        
    return (int(x), int(y), int(w), int(h))

def trace_uav_path(crop_sequence, satellite_img, search_radius=100):
    """
    Trace the UAV path by sequentially finding crop images in the satellite image
    Args:
        crop_sequence: List of crop images (numpy arrays) in sequence
        satellite_img: The full satellite image (numpy array)
        search_radius: Radius to search for the next crop after finding the previous one
    Returns:
        List of (x, y, w, h) for each found crop, representing the UAV path
    """
    if not crop_sequence:
        return []
        
    path = []
    
    initial_result = find_image_in_larger(crop_sequence[0], satellite_img)
    
    if initial_result is None:
        return []
    
    x, y, w, h = initial_result
    path.append(initial_result)
    center_x, center_y = x + w//2, y + h//2
    
    for crop in crop_sequence[1:]:
        search_region = (center_x, center_y, search_radius)
        
        result = find_image_in_larger(crop, satellite_img, search_region)
        
        if result is None:
            break
        
        x, y, w, h = result
        path.append(result)
        center_x, center_y = x + w//2, y + h//2
    
    return path

def visualize_path(satellite_img_path, crop_sequence, path):
    """
    Visualize the traced UAV path on the satellite image.
    
    Args:
        satellite_img_path: Path to the satellite image
        crop_sequence: List of paths to crop images
        path: List of (x, y, w, h, scale) tuples for each found crop
    """
    satellite_img = cv2.imread(satellite_img_path)
    
    vis_img = satellite_img.copy()
    
    colors = [
        (0, 255, 0),    
        (0, 0, 255),  
        (255, 0, 0),    
        (0, 255, 255),  
        (255, 0, 255), 
        (255, 255, 0),  
        (128, 0, 128), 
        (0, 128, 128),  
        (128, 128, 0), 
        (0, 0, 128)  
    ]
    
    path_points = []
    
    for i, (x, y, w, h) in enumerate(path):
        center_x = x + w//2
        center_y = y + h//2
        path_points.append((center_x, center_y))
        
        color = colors[i % len(colors)]
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
        
        cv2.putText(vis_img, f"Crop {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        crop_img = cv2.imread(crop_sequence[i])
        if crop_img is not None:
            thumb_size = 100
            thumb = cv2.resize(crop_img, (thumb_size, thumb_size))
            
            thumb_x = min(satellite_img.shape[1] - thumb_size, x + w + 10)
            thumb_y = min(satellite_img.shape[0] - thumb_size, y)
            
            vis_img[thumb_y:thumb_y+thumb_size, thumb_x:thumb_x+thumb_size] = thumb
    
    for i in range(1, len(path_points)):
        pt1 = path_points[i-1]
        pt2 = path_points[i]
        cv2.line(vis_img, pt1, pt2, (255, 255, 255), 2)
    
    for i in range(1, len(path_points)):
        pt1 = path_points[i-1]
        pt2 = path_points[i]
        
        angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        
        arrow_len = 20
        
        arrow_pt1 = (
            int(mid_x - arrow_len * math.cos(angle + math.pi/6)),
            int(mid_y - arrow_len * math.sin(angle + math.pi/6))
        )
        arrow_pt2 = (mid_x, mid_y)
        arrow_pt3 = (
            int(mid_x - arrow_len * math.cos(angle - math.pi/6)),
            int(mid_y - arrow_len * math.sin(angle - math.pi/6))
        )
        
        cv2.line(vis_img, arrow_pt1, arrow_pt2, (255, 255, 255), 2)
        cv2.line(vis_img, arrow_pt3, arrow_pt2, (255, 255, 255), 2)
    
    plt.figure(figsize=(15, 15))
    plt.title("UAV Path Traced on Satellite Image")
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def pixel_to_geo_coords(x, y, img_width, img_height, LT_lat, LT_lon, RB_lat, RB_lon):
    """
    Convert pixel coordinates to geographical coordinates
    
    Args:
        x, y: Pixel coordinates
        img_width, img_height: Image dimensions
        LT_lat, LT_lon: Top-left corner geographical coordinates (latitude, longitude)
        RB_lat, RB_lon: Bottom-right corner geographical coordinates (latitude, longitude)
    
    Returns:
        (latitude, longitude): Geographical coordinates
    """
    x_prop = x / img_width
    y_prop = y / img_height
    
    latitude = LT_lat - y_prop * (LT_lat - RB_lat)
    longitude = LT_lon + x_prop * (RB_lon - LT_lon)
    
    return (latitude, longitude)

if __name__ == "__main__":
    LT_lat_map =  24.666899 
    LT_lon_map = 102.340055 
    RB_lat_map = 24.650422  
    RB_lon_map = 102.365252 
    satellite_img = cv2.imread("05/satellite05.tif")
    crop_sequence = [cv2.imread(f"05/drone/05_000{i}.JPG") for i in range(1,9)]
    print(len(crop_sequence))
    new_width = int(satellite_img.shape[1] * 0.5)
    new_height = int(satellite_img.shape[0] * 0.5)
    satellite_img =  cv2.resize(satellite_img, (new_width, new_height))
    path = trace_uav_path(crop_sequence, satellite_img, search_radius=100)

    print("UAV path traced successfully")
        
    img_height, img_width = satellite_img.shape[:2]

    for i, (x, y, w, h) in enumerate(path):
        center_x = x + w//2
        center_y = y + h//2
        
        lat, lon = pixel_to_geo_coords(center_x, center_y, img_width, img_height, 
                                    LT_lat_map, LT_lon_map, RB_lat_map, RB_lon_map)
        
        print(f"Crop {i}: Pixel Position ({center_x}, {center_y}), "
            f"Geo Coordinates (Lat: {lat:.6f}, Lon: {lon:.6f})")
    
    if path:
        print("UAV path traced successfully")
        for i, (x, y, w, h) in enumerate(path):
            print(f"Crop {i}: Position ({x}, {y}), Size ({w}x{h})")
    else:
        print("Failed to trace UAV path")