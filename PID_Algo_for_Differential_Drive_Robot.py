import numpy as np
import matplotlib.pyplot as plt

class PID_Controller:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    
    def pid_tune(self, error, last_error, dt):
        if abs(error) > 0.5:
            self.kp += 0.005
        if abs(error - last_error) > 0.5:
            self.kd += 1e-4
        if abs(error) < 0.5 and abs(self.integral) > 1:
            self.ki += 1e-5

wheel_radius = 0.05
distance_between_wheels = 0.2

class DifferentialDriveRobot:
    def __init__(self, wheel_radius, distance_between_wheels, x=0, y=0, theta=0):
        self.wheel_radius = wheel_radius
        self.distance_between_wheels = distance_between_wheels
        self.x = x
        self.y = y
        self.theta = theta

    def update_pose(self, v, omega, dt):
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt

def bot_simulation(start, waypoints, dt=0.1):
    x, y = start
    path = [(x, y)]
    theta = 0
    pid_bot = PID_Controller(kp=0, ki=0, kd=0)

    for point in waypoints:
        last_error = 0
        while np.sqrt((point[0] - x)**2 + (point[1] - y)**2) > 0.05:
            error = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)

            pid_bot.pid_tune(error, last_error, dt)
            control_signal = pid_bot.compute(error, dt)

            theta_deviation = np.arctan2((point[1] - y), (point[0] - x))
            theta_error = theta_deviation - theta
            theta += 0.5 * theta_error

            v = control_signal
            omega = 2.0 * theta_error

            omega_r = (2 * v + omega * distance_between_wheels) / (2 * wheel_radius)
            omega_l = (2 * v - omega * distance_between_wheels) / (2 * wheel_radius)

            v = wheel_radius * (omega_r + omega_l) / 2
            omega = wheel_radius * (omega_r - omega_l) / distance_between_wheels

            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            path.append((x, y))
            last_error = error

    return path, pid_bot

start = (0, 0)
waypoints = [(2, 1), (1, 3)]

real_path, tuned_pid = bot_simulation(start=start, waypoints=waypoints)
real_path = np.array(real_path)

desired_x = [start[0]] + [p[0] for p in waypoints]
desired_y = [start[1]] + [p[1] for p in waypoints]

plt.scatter(*zip(*waypoints), color='red', label="Waypoints")
plt.scatter(*start, color='green', label="Start")
plt.plot(real_path[:, 0], real_path[:, 1], label="PID Path", color="blue")
plt.plot(desired_x, desired_y, label="Desired Path", color="orange", linestyle="--")

pid_text = f"Final Tuned Values:\nKp = {round(tuned_pid.kp, 2)}\nKi = {round(tuned_pid.ki, 4)}\nKd = {round(tuned_pid.kd, 4)}"
plt.text(
    min(real_path[:, 0]), max(real_path[:, 1]),
    pid_text,
    fontsize=10,
    color="black",
    verticalalignment='top',
    bbox=dict(facecolor='gray', alpha=0.5)
)

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Differential Drive Robot with PID Tuning")
plt.grid()
plt.show()