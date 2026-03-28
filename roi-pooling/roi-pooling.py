import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    H = len(feature_map)
    W = len(feature_map[0])
    
    results = []
    
    for roi in rois:
        x1, y1, x2, y2 = roi
        
        roi_h = y2 - y1
        roi_w = x2 - x1
        
        pooled = []
        
        for i in range(output_size):
            row = []
            for j in range(output_size):
                
                # Compute bin boundaries
                h_start = y1 + math.floor(i * roi_h / output_size)
                h_end   = y1 + math.floor((i + 1) * roi_h / output_size)
                
                w_start = x1 + math.floor(j * roi_w / output_size)
                w_end   = x1 + math.floor((j + 1) * roi_w / output_size)
                
                # Ensure at least 1 pixel
                if h_end <= h_start:
                    h_end = h_start + 1
                if w_end <= w_start:
                    w_end = w_start + 1
                
                # Clip to feature map bounds
                h_end = min(h_end, H)
                w_end = min(w_end, W)
                
                # Max pooling
                max_val = float('-inf')
                for h in range(h_start, h_end):
                    for w in range(w_start, w_end):
                        max_val = max(max_val, feature_map[h][w])
                
                row.append(max_val)
            
            pooled.append(row)
        
        results.append(pooled)
    
    return results