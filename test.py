import streamlit as st
import numpy as np
import pandas as pd
import time

# Function to implement Kalman filter for height estimation
def kalman_filter(height_measurement, prev_height_estimate, prev_height_covariance,
                  process_height_variance, measurement_variance):
    # State transition matrix (constant velocity model)
    F = np.array([[1, 1], [0, 1]])

    # Measurement matrix (we directly measure height)
    H = np.array([[1, 0]])

    # Prediction
    height_estimate = np.dot(F, np.array([prev_height_estimate, 0]))
    height_covariance = np.dot(np.dot(F, prev_height_covariance), F.T) + np.diag([process_height_variance, 0])

    # Kalman gain calculation
    K = np.dot(np.dot(height_covariance, H.T),
               np.linalg.inv(np.dot(np.dot(H, height_covariance), H.T) + measurement_variance))

    # Measurement update
    correction = np.dot(K, (height_measurement - np.dot(H, height_estimate)))
    height_estimate = height_estimate + correction
    height_covariance = np.dot((np.eye(2) - np.dot(K, H)), height_covariance)

    return height_estimate[0], height_covariance

# Streamlit app
def kalman_filter_app():
    # Load data from CSV file
    filename = '/home/akash/Desktop/maths project/aircraft_dataset.csv'
    data = pd.read_csv(filename)

    # Extract sensor measurements
    accelerometer_data = data['Accelerometer'].values
    gyroscope_data = data['Gyroscope'].values
    magnetometer_data = data['Magnetometer'].values
    height_measurement = data['Height'].values
    vertical_speed_measurement = data['VerticalSpeed'].values

    # Kalman filter initialization
    initial_height_estimate = height_measurement[0]
    initial_height_covariance = np.diag([1, 1])  # Initial uncertainty
    process_height_variance = 0.01
    measurement_variance = 1

    # Streamlit layout
    st.title("Aircraft Altitude Estimation")

    # Display initial sensor values
    st.subheader("Sensor Data")
    accelerometer_slot = st.empty()
    gyroscope_slot = st.empty()
    magnetometer_slot = st.empty()

    # Display initial actual height, estimated height, vertical speed, and flight status
    st.subheader("Flight Telemetry")
    actual_height_slot = st.empty()
    estimated_height_slot = st.empty()
    vertical_speed_slot = st.empty()
    flight_status_slot = st.empty()

    st.subheader("Cockpit")
    cockpit_status = st.empty()

    # Kalman filter loop
    for i in range(1, len(data)):
        # Apply Kalman filter
        prev_height_estimate, prev_height_covariance = kalman_filter(
            height_measurement[i],
            initial_height_estimate, initial_height_covariance,
            process_height_variance, measurement_variance
        )

        # Update sensor values
        accelerometer_slot.text(f"Accelerometer: {accelerometer_data[i]}")
        gyroscope_slot.text(f"Gyroscope: {gyroscope_data[i]}")
        magnetometer_slot.text(f"Magnetometer: {magnetometer_data[i]}")

        # Display and update actual height, estimated height, vertical speed, and flight status
        actual_height_slot.text(f"Actual Height: {height_measurement[i]:.2f}")
        estimated_height_slot.text(f"Estimated Height: {prev_height_estimate:.2f}")


        # Determine flight status
        if prev_height_estimate < height_measurement[i]:
            flight_status = "Ascending"
        else:
            flight_status = "Descending"

        # Update flight status dynamically
        flight_status_slot.text(f"Flight Status: {flight_status}")

        # Simulate cockpit status
        time.sleep(0.1)
        cockpit_status.text(f"Status: Taking Off ({i + 1}/100)")
        # Delay for visualization
        time.sleep(0.2)

    # Display success message after takeoff is completed
    st.success("Takeoff completed successfully!")

# Run the Streamlit app
kalman_filter_app()
