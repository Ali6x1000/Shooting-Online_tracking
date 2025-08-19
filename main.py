import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import csv
from scipy.spatial.distance import pdist, squareform

# Load or generate data
def generate_synthetic_data(n_points=50):
    t = np.linspace(0, 10, n_points)
    x = 100 + 50 * t - 4.9 * t**2
    y = 500 - 4.9 * t**2
    noise = np.random.normal(0, 5, (n_points, 2))
    return [(x[i] + noise[i,0], y[i] + noise[i,1], t[i]) for i in range(n_points)]

try:
    raw = []
    with open('data.csv', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw.append((float(row['x']), float(row['y']), float(row['t'])))
    if not raw:
        raise FileNotFoundError
except:
    raw = generate_synthetic_data()

data = np.array(raw)  # shape (N, 3): x, y, t

# --- 3D DBSCAN Parameters ---
spatial_eps = 15      # max spatial distance in x-y
temporal_eps = 1.0    # max time difference in seconds
min_samples = 3       # minimum points per cluster
spatial_cluster_threshold = 5.0  # max distance to consider points "clustered together"

# Scale time axis
time_scale = spatial_eps / temporal_eps
features = np.column_stack([data[:,0], data[:,1], data[:,2] * time_scale])

# Run DBSCAN
db = DBSCAN(eps=spatial_eps, min_samples=min_samples).fit(features)
labels = db.labels_  # -1 = noise

# Quadratic curve fitting function
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

# Fit quadratic curve to a set of points
def fit_quadratic_curve(points):
    x = points[:, 0]
    y = points[:, 1]
    
    if len(x) < 3:  # Need at least 3 points to fit a quadratic
        return None, None, None
    
    try:
        popt, _ = curve_fit(quadratic, x, y)
        y_pred = quadratic(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt, r2, (x, y)
    except:
        return None, None, None

# Check if curve is "increasing" (positive slope at midpoint)
def is_increasing_curve(coefficients, x_vals):
    a, b, _ = coefficients
    x_mid = np.mean(x_vals)
    slope = 2 * a * x_mid + b
    return slope > 0

# Filter out points that are too close together
def filter_clustered_points(points, threshold=spatial_cluster_threshold):
    if len(points) < 3:
        return points
    
    # Compute pairwise distances in x-y space
    xy_points = points[:, :2]
    distances = squareform(pdist(xy_points))
    
    # Mark points to keep (not too close to others)
    keep = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if keep[i]:
            # Check if this point is too close to others
            close_points = np.sum((distances[i] < threshold) & (distances[i] > 0))
            if close_points > 1:  # More than 1 neighbor too close
                keep[i] = False
    
    filtered_points = points[keep]
    if len(filtered_points) < 3:
        return points  # Return original if filtering leaves too few points
    return filtered_points

# Split cluster into subsets based on residuals or time gaps
def split_cluster(points, residual_threshold=10.0, time_gap_threshold=0.5):
    if len(points) < 6:  # Need enough points to consider splitting
        return [filter_clustered_points(points)]
    
    # Sort points by time
    points = points[points[:, 2].argsort()]
    x, y = points[:, 0], points[:, 1]
    
    # Fit initial quadratic to entire cluster
    popt, r2, _ = fit_quadratic_curve(points)
    if popt is None:
        return [filter_clustered_points(points)]
    
    # Calculate residuals
    y_pred = quadratic(x, *popt)
    residuals = np.abs(y - y_pred)
    
    # Find split points based on high residuals or time gaps
    subsets = []
    current_subset = [points[0]]
    for i in range(1, len(points)):
        time_diff = points[i, 2] - points[i-1, 2]
        if residuals[i] > residual_threshold or time_diff > time_gap_threshold:
            filtered_subset = filter_clustered_points(np.array(current_subset))
            if len(filtered_subset) >= 3:
                subsets.append(filtered_subset)
            current_subset = [points[i]]
        else:
            current_subset.append(points[i])
    filtered_subset = filter_clustered_points(np.array(current_subset))
    if len(filtered_subset) >= 3:
        subsets.append(filtered_subset)
    
    # Filter subsets with at least 3 points
    return [subset for subset in subsets if len(subset) >= 3]




# Plot only valid trajectory lines
fig, ax = plt.subplots(figsize=(12,8))
unique_labels = sorted(set(labels))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for lbl, color in zip(unique_labels, colors):
    mask = labels == lbl
    pts = data[mask]
    
    if lbl != -1:  # Skip noise points
        # Split cluster into subsets
        subsets = split_cluster(pts)
        
        for i, subset in enumerate(subsets):
            popt, r2, fitted_data = fit_quadratic_curve(subset)
            
            if popt is not None and r2 > 0.9 and is_increasing_curve(popt, subset[:,0]):
                # Plot fitted curve only
                x_fit = np.linspace(min(subset[:,0]), max(subset[:,0]), 100)
                y_fit = quadratic(x_fit, *popt)
                ax.plot(x_fit, y_fit, color=color, linewidth=2, linestyle='--')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Valid Quadratic Trajectory Lines (R² > 0.9, Increasing)')
plt.tight_layout()
plt.show()