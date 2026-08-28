"""Custom QoS profiles for camera sensors with bandwidth optimization."""

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Sensor data with reduced history depth (1 instead of 5) to minimize buffering.
# Trades latency jitter for reduced memory and network load.
qos_profile_sensor_data_low_latency = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
