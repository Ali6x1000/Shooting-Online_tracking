import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import csv
from matplotlib.animation import FuncAnimation

# Generate synthetic data if data.csv is unavailable
def generate_synthetic_data(n_points=50):
    t = np.linspace(0, 10, n_points)
    x = 100 + 50 * t - 4.9 * t**2  # Parabolic trajectory in x
    y = 500 - 4.9 * t**2  # Parabolic trajectory in y
    w = np.ones(n_points) * 5  # Dummy width
    h = np.ones(n_points) * 5  # Dummy height
    noise = np.random.normal(0, 5, (n_points, 2))  # Add noise to x, y
    return [(x[i] + noise[i, 0], y[i] + noise[i, 1], w[i], h[i], t[i]) for i in range(n_points)]

# Load data from CSV or use synthetic data
try:
    raw_data = []
    with open('data.csv', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row['x'])
                y = float(row['y'])
                w = float(row['w'])
                h = float(row['h'])
                t = float(row['t'])
                raw_data.append((x, y, w, h, t))
            except (ValueError, KeyError) as e:
                print(f"Skipping invalid row: {row}, error: {e}")
    print(f"Loaded {len(raw_data)} points from data.csv")
except (FileNotFoundError, ValueError) as e:
    print(f"Error loading data.csv: {e}. Using synthetic data.")
    raw_data = generate_synthetic_data()

# Validate and sort data
if not raw_data:
    print("No valid data to process. Exiting.")
    exit()
raw_data = sorted(raw_data, key=lambda t: t[4])
print("First 5 points:", raw_data[:5])

# Helper function for quadratic fit
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def fit_quadratic_curve(cluster, t=None):
    x, y = cluster[:, 0], cluster[:, 1]
    if len(x) < 3:
        return None, None
    try:
        popt, _ = curve_fit(quadratic, x, y)
        y_pred = quadratic(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt, r2
    except RuntimeError:
        return None, None

# Trajectory class to manage individual trajectories
class Trajectory:
    def __init__(self, idx, points, coeffs, r2, last_time):
        self.idx = idx
        self.points = points  # List of (x, y, w, h, t)
        self.coeffs = coeffs  # Quadratic fit coefficients (a, b, c)
        self.r2 = r2
        self.last_time = last_time  # Timestamp of the last point

    def predict_next_point(self, t):
        """Predict the next point's position at time t based on the quadratic fit."""
        a, b, c = self.coeffs
        x = self.points[-1][0]  # Use last x for simplicity
        y_pred = quadratic(t, a, b, c)
        return np.array([x, y_pred])

    def check_point_fit(self, point, t, max_dist=10):
        """Check if a point fits this trajectory based on predicted location."""
        x, y, _, _, pt_t = point
        pred_pos = self.predict_next_point(pt_t)
        dist = np.sqrt((x - pred_pos[0])**2 + (y - pred_pos[1])**2)
        if dist > max_dist:
            return False, f"Distance too large ({dist:.2f} > {max_dist})"
        return True, "Fit"

# TrajectoryProcessor class to manage trajectory detection and updates
class TrajectoryProcessor:
    def __init__(self, data, eps=10, min_samples=3, max_gap=1.0, r2_threshold=0.7, outlier_time_window=1.0):
        self.data = data
        self.eps = eps
        self.min_samples = min_samples
        self.max_gap = max_gap
        self.r2_threshold = r2_threshold
        self.outlier_time_window = outlier_time_window  # Time window for outlier grouping (seconds)
        self.active_trajectories = []
        self.terminated_trajectories = []
        self.current_points = []  # Points for active trajectory evaluation
        self.outliers = []  # Time-stamped outliers for new trajectory formation
        self.shot_idx = 0

    def initialize_trajectories(self):
        """Initialize trajectories by finding clusters with 3+ points for quadratic fit."""
        if len(self.current_points) >= self.min_samples:
            window_pts = np.array([(p[0], p[1], p[4]) for p in self.current_points])
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(window_pts[:, :2])
            labels = db.labels_
            clusters = set(labels) - {-1}
            for cid in clusters:
                cluster_indices = np.where(labels == cid)[0]
                cluster_pts = window_pts[cluster_indices]
                if len(cluster_pts) >= self.min_samples:
                    xy_pts = cluster_pts[:, :2]
                    t_vals = cluster_pts[:, 2]
                    coeffs, r2 = fit_quadratic_curve(xy_pts)
                    if coeffs is not None and r2 >= self.r2_threshold:
                        cluster_points = [self.current_points[i] for i in cluster_indices]
                        last_t = max(p[4] for p in cluster_points)
                        new_traj = Trajectory(self.shot_idx, cluster_points, coeffs, r2, last_t)
                        self.active_trajectories.append(new_traj)
                        print(f"Initialized trajectory {self.shot_idx} with {len(cluster_points)} points (R²={r2:.2f})")
                        self.shot_idx += 1

    def process_point(self, point):
        x, y, _, _, t = point
        self.current_points.append(point)  # Add to main point list

        # Initialize trajectories if not yet done and enough points are available
        if not self.active_trajectories and len(self.current_points) >= self.min_samples:
            self.initialize_trajectories()

        point_added = False

        # Check against existing trajectories with sliding window
        if self.active_trajectories:
            window_pts = np.array([(p[0], p[1], p[4]) for p in self.current_points])
            if len(window_pts) >= 1:  # Check individual points
                db = DBSCAN(eps=self.eps, min_samples=1).fit(window_pts[:, :2])  # Lower min_samples for individual points
                labels = db.labels_
                clusters = set(labels) - {-1}
                print(f"Frame {t:.2f}s: {len(clusters)} clusters found in window")

                for cid in clusters:
                    cluster_indices = np.where(labels == cid)[0]
                    cluster_points = [self.current_points[i] for i in cluster_indices]
                    last_t = max(p[4] for p in cluster_points)

                    for traj in self.active_trajectories:
                        fits, _ = traj.check_point_fit(cluster_points[-1], last_t)
                        if fits:
                            traj.points.extend(cluster_points)
                            traj.last_time = last_t  # Reset last_time to keep trajectory alive
                            traj.coeffs, traj.r2 = fit_quadratic_curve(np.array([(p[0], p[1]) for p in traj.points]))
                            point_added = True
                            print(f"Added point to trajectory {traj.idx} at t={t:.2f}s (R²={traj.r2:.2f})")
                            break
                    if point_added:
                        break

        # If no trajectory fits, add as outlier
        if not point_added:
            self.outliers.append((x, y, 5, 5, t))  # Add with dummy width and height
            print(f"Added point at t={t:.2f}s as outlier")

        # Terminate trajectories inactive for max_gap seconds
        for traj in self.active_trajectories[:]:
            if t - traj.last_time > self.max_gap:
                self.terminated_trajectories.append(traj)
                self.active_trajectories.remove(traj)
                print(f"Terminated trajectory {traj.idx} (no update for {self.max_gap}s)")

        # Evaluate outliers for new trajectories within time window
        if self.outliers:
            current_t = t
            recent_outliers = [o for o in self.outliers if current_t - o[4] <= self.outlier_time_window]
            if len(recent_outliers) >= self.min_samples:
                outlier_pts = np.array([(p[0], p[1], p[4]) for p in recent_outliers])
                db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(outlier_pts[:, :2])
                labels = db.labels_
                clusters = set(labels) - {-1}
                for cid in clusters:
                    cluster_indices = np.where(labels == cid)[0]
                    cluster_pts = outlier_pts[cluster_indices]
                    if len(cluster_pts) >= self.min_samples:
                        xy_pts = cluster_pts[:, :2]
                        t_vals = cluster_pts[:, 2]
                        coeffs, r2 = fit_quadratic_curve(xy_pts)
                        if coeffs is not None and r2 >= self.r2_threshold:
                            cluster_points = [recent_outliers[i] for i in cluster_indices]
                            last_t = max(p[4] for p in cluster_points)
                            new_traj = Trajectory(self.shot_idx, cluster_points, coeffs, r2, last_t)
                            self.active_trajectories.append(new_traj)
                            self.outliers = [o for o in self.outliers if o not in cluster_points]
                            print(f"Created new trajectory {self.shot_idx} from outliers with {len(cluster_points)} points (R²={r2:.2f})")
                            self.shot_idx += 1

        return point_added

    def finalize(self):
        self.terminated_trajectories.extend(self.active_trajectories)
        self.active_trajectories = []
        print(f"Finalized {len(self.terminated_trajectories)} trajectories, {len(self.outliers)} outliers remaining")

# Animation function
def animate_trajectories(data):
    if not data:
        print("No data to animate")
        return
    processor = TrajectoryProcessor(data)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(data) // 10)))

    def init():
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Quadratic Trajectory Fits with Time-Dependent Outliers')
        ax.grid(True)
        x_vals = [x for x, _, _, _, _ in data]
        y_vals = [y for _, y, _, _, _ in data]
        ax.set_xlim(min(x_vals, default=0) - 10, max(x_vals, default=100) + 10)
        ax.set_ylim(min(y_vals, default=0) - 10, max(y_vals, default=100) + 10)
        return []

    def update(frame):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Quadratic Trajectory Fits with Outliers (t={data[frame][4]:.2f}s)')
        ax.grid(True)
        x_vals = [x for x, _, _, _, _ in data]
        y_vals = [y for _, y, _, _, _ in data]
        ax.set_xlim(min(x_vals, default=0) - 10, max(x_vals, default=100) + 10)
        ax.set_ylim(min(y_vals, default=0) - 10, max(y_vals, default=100) + 10)

        processor.process_point(data[frame])

        # Plot terminated trajectories
        for traj in processor.terminated_trajectories:
            if traj.points:
                pts = np.array([(x, y) for x, y, _, _, _ in traj.points])
                color = colors[traj.idx % len(colors)]
                ax.scatter(pts[:, 0], pts[:, 1], color=color, marker='o', alpha=0.5)
                if traj.coeffs is not None:
                    x_range = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 100)
                    y_fit = quadratic(x_range, *traj.coeffs)
                    ax.plot(x_range, y_fit, color=color, label=f'Trajectory {traj.idx+1} (R²={traj.r2:.2f})')

        # Plot active trajectories
        for traj in processor.active_trajectories:
            if traj.points:
                pts = np.array([(x, y) for x, y, _, _, _ in traj.points])
                color = colors[traj.idx % len(colors)]
                ax.scatter(pts[:, 0], pts[:, 1], color=color, marker='o', alpha=0.5)
                if traj.coeffs is not None:
                    x_range = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 100)
                    y_fit = quadratic(x_range, *traj.coeffs)
                    ax.plot(x_range, y_fit, color=color, linestyle='--', label=f'Active T{traj.idx+1} (R²={traj.r2:.2f})')
                    pred_t = data[frame][4] + 0.05
                    pred_pos = traj.predict_next_point(pred_t)
                    ax.scatter([pred_pos[0]], [pred_pos[1]], color=color, marker='x', s=100, label=f'Pred T{traj.idx+1}')

        # Plot outliers
        if processor.outliers:
            outlier_pts = np.array([(x, y) for x, y, _, _, _ in processor.outliers])
            ax.scatter(outlier_pts[:, 0], outlier_pts[:, 1], color='gray', marker='x', alpha=0.5, label='Outliers')

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        return []

    processor.finalize()
    ani = FuncAnimation(fig, update, frames=len(data), init_func=init, blit=False, interval=200)
    plt.tight_layout()
    # ani.save('trajectories.mp4', writer='ffmpeg', fps=5)  # Uncomment to save
    plt.show()

# Run animation
animate_trajectories(raw_data)