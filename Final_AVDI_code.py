from ultralytics import YOLO
import cv2
import os

# Load the YOLO pretrained model (YOLOv8 small model as an example)
model = YOLO('yolov8s.pt')

# Define the video input path and output path
video_path = r"C:\ML\AVDI file\temp\input.mp4"      # Input video path
output_path = r'C:\ML\AVDI file\output\output.mp4'  # Output video path
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    print("Video opened successfully.")

# Define class IDs for vehicles in YOLO (COCO dataset classes) with specific colors
vehicle_classes = {
    2: ('Car', (0, 255, 0)),        # Green
    3: ('Motorcycle', (255, 0, 0)), # Blue
    5: ('Bus', (0, 0, 255)),        # Red
    7: ('Truck', (255, 255, 0))     # Cyan
}

# Initialize counters and tracking dictionary
vehicle_counts = {class_name: 0 for class_name, _ in vehicle_classes.values()}
incoming_counts = {class_name: 0 for class_name, _ in vehicle_classes.values()}
outgoing_counts = {class_name: 0 for class_name, _ in vehicle_classes.values()}
tracked_vehicles = {}
vehicle_id_counter = 0

# Define the half-line position (right side to middle for incoming and left side to middle for outgoing)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
incoming_line_y = int(frame_height * 0.4)  # Set incoming line position slightly above the middle
outgoing_line_y = int(frame_height * 0.4)  # Set outgoing line position at the middle
incoming_line_start_x = frame_width // 2   # Start the incoming line at the middle
incoming_line_end_x = frame_width           # Extend line to the right edge
outgoing_line_start_x = 0                   # Start the outgoing line at the left edge
outgoing_line_end_x = frame_width // 2      # Extend line to the middle

# Define a time-to-live (TTL) for tracking vehicles after they cross the line
ttl_frames = 20  # Number of frames to keep a vehicle in memory to prevent re-counting

# Set up video writer to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or cannot read the frame.")
        break

    # Run inference on the current frame
    results = model(frame)

    # Draw the incoming line from center to the right edge of the frame
    cv2.line(frame, (incoming_line_start_x, incoming_line_y), (incoming_line_end_x, incoming_line_y), (255, 0, 0), 2)
    # Draw the outgoing line from left edge to the middle of the frame
    cv2.line(frame, (outgoing_line_start_x, outgoing_line_y), (outgoing_line_end_x, outgoing_line_y), (0, 255, 255), 2)

    # Display the incoming counts 
    y_offset = 20
    cv2.putText(frame, "INCOMING:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 30
    for class_name, count in vehicle_counts.items():
        cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

    # Display the outgoing counts
    cv2.putText(frame, "OUTGOING:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 30
    for class_name, count in outgoing_counts.items():
        cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

    # Process each detected box
    new_tracked_vehicles = {}  # Temporary dictionary for vehicles detected in this frame

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in vehicle_classes:  # Filter for vehicle classes
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                class_name, color = vehicle_classes[class_id]  # Get class name and color
                confidence = box.conf[0]

                # Calculate the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Check if the vehicle is crossing the incoming line on the right side
                if incoming_line_start_x <= center_x <= incoming_line_end_x and abs(center_y - incoming_line_y) < 10:
                    # Search for a match in tracked vehicles to avoid duplicate counting
                    vehicle_found = False
                    for vehicle_id, (tracked_class, position, frames) in tracked_vehicles.items():
                        if (tracked_class == class_name and
                            abs(center_x - position[0]) < 50 and
                            abs(center_y - position[1]) < 50):
                            # Update the position and TTL
                            new_tracked_vehicles[vehicle_id] = (class_name, (center_x, center_y), ttl_frames)
                            vehicle_found = True
                            break

                    # If not counted yet, add to tracking and increment incoming count
                    if not vehicle_found:
                        vehicle_counts[class_name] += 1
                        incoming_counts[class_name] += 1  # Increment class-specific incoming count
                        new_tracked_vehicles[vehicle_id_counter] = (class_name, (center_x, center_y), ttl_frames)
                        vehicle_id_counter += 1

                # Check if the vehicle is crossing the outgoing line
                if outgoing_line_start_x <= center_x <= outgoing_line_end_x and abs(center_y - outgoing_line_y) < 10:
                    # Search for a match in tracked vehicles to avoid duplicate counting
                    vehicle_found = False
                    for vehicle_id, (tracked_class, position, frames) in tracked_vehicles.items():
                        if (tracked_class == class_name and
                            abs(center_x - position[0]) < 50 and
                            abs(center_y - position[1]) < 50):
                            # Update the position and TTL
                            new_tracked_vehicles[vehicle_id] = (class_name, (center_x, center_y), ttl_frames)
                            vehicle_found = True
                            break

                    # If not counted yet, add to tracking and increment outgoing count
                    if not vehicle_found:
                        outgoing_counts[class_name] += 1
                        new_tracked_vehicles[vehicle_id_counter] = (class_name, (center_x, center_y), ttl_frames)
                        vehicle_id_counter += 1

                # Draw the bounding box and label with unique color
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update the tracked vehicles, removing any that have been in memory for too long
    tracked_vehicles = {vid: (cls, pos, frames - 1) for vid, (cls, pos, frames) in tracked_vehicles.items() if frames > 0}
    tracked_vehicles.update(new_tracked_vehicles)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame with vehicle detections and counts
    cv2.imshow('Detected Vehicles', frame)

    # Press 'q' to exit the video early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object, video writer, and close display window
cap.release()
out.release()
cv2.destroyAllWindows()

# Print final counts
print("Final Counts:")
print("Incoming Vehicles:")
for class_name, count in incoming_counts.items():
    print(f"{class_name}: {count}")

print("Outgoing Vehicles:")
for class_name, count in outgoing_counts.items():
    print(f"{class_name}: {count}")
    