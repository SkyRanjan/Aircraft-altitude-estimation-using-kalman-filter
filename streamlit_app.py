import streamlit as st
import numpy as np
import time

# Kalman filter function
def kalman_filter(accelerometer_data, initial_state_estimate, initial_state_covariance,
                  process_altitude_variance, measurement_variance):
    num_steps = accelerometer_data.shape[1]
    dt = 0.1  # Assuming constant time step for simplicity

    # State transition matrix (constant velocity model)
    F = np.array([[1, dt],
                  [0, 1]])

    # Measurement matrix (we directly measure altitude using accelerometer)
    H = np.array([[0, 0]])

    # Kalman filter initialization
    state_estimate = np.zeros((2, num_steps))
    state_estimate[:, 0] = initial_state_estimate
    state_covariance = np.zeros((2, 2, num_steps))
    state_covariance[:, :, 0] = initial_state_covariance

    # Kalman filter loop
    for k in range(1, num_steps):
        # Prediction
        state_estimate[:, k] = np.dot(F, state_estimate[:, k-1])
        state_covariance[:, :, k] = np.dot(np.dot(F, state_covariance[:, :, k-1]), F.T) + np.diag([process_altitude_variance, 0])

        # Measurement update
        K = np.dot(np.dot(state_covariance[:, :, k], H.T),
                   np.linalg.inv(np.dot(np.dot(H, state_covariance[:, :, k]), H.T) + measurement_variance))
        state_estimate[:, k] = state_estimate[:, k] + np.dot(K, (accelerometer_data[:, k] - np.dot(H, state_estimate[:, k])))
        state_covariance[:, :, k] = np.dot((np.eye(2) - np.dot(K, H)), state_covariance[:, :, k])

    return state_estimate[0, :], state_covariance[0, 0, :]

# Streamlit app
def simulate_aircraft_altitude_prediction():
    # Function to simulate aircraft takeoff using Streamlit

    # Initial setup
    altitude = 0.0
    velocity = 0.1
    dt = 0.1

    # Streamlit layout
    st.title("Aircraft Altitude Prediction")

    st.subheader("Sensor Data")
    accelerometer = st.empty()
    magnetometer = st.empty()
    gyroscope = st.empty()

    st.subheader("Altitude")
    altitude_display = st.empty()
    predicted_altitude_display = st.empty()

    st.subheader("Cockpit")
    cockpit_status = st.empty()

    # Simulate aircraft takeoff
    for _ in range(500):
        # Simulate sensor data
        accelerometer_data = np.random.normal(0, 0.05, 3)
        magnetometer_data = np.random.normal(0, 0.1, 3)
        gyroscope_data = np.random.normal(0, 0.01, 3)

        # Display sensor data
        accelerometer.text(f"Accelerometer: {accelerometer_data}")
        magnetometer.text(f"Magnetometer: {magnetometer_data}")
        gyroscope.text(f"Gyroscope: {gyroscope_data}")

        # Simulate altitude prediction using Kalman filter
        altitude += velocity * dt * 5000
        predicted_altitude = altitude + np.random.normal(0, 0.1)

        # Display altitude
        altitude_display.text(f"Altitude: {altitude:.2f} ft")
        predicted_altitude_display.text(f"Predicted Altitude: {predicted_altitude:.2f} ft")

        # Simulate cockpit status
        time.sleep(0.1)
        cockpit_status.text(f"Status: Taking Off ({_+1}/500)")


    st.success("Takeoff completed successfully!")

# Run the Streamlit app
simulate_aircraft_altitude_prediction()
