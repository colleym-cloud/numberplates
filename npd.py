import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from pyzbar.pyzbar import decode

def detect_number_plate(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

    plate_numbers = []

    # Draw rectangles around the filtered contours and decode barcodes
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Crop the region of interest (ROI) for barcode decoding
        roi = gray[y:y+h, x:x+w]

        # Decode barcodes using pyzbar
        barcodes = decode(roi)

        for barcode in barcodes:
            barcode_data = barcode.data.decode('utf-8')
            plate_numbers.append(barcode_data)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Plate: {barcode_data}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, plate_numbers

# Initialize PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
raw_capture = PiRGBArray(camera, size=(640, 480))

# Allow the camera to warm up
time.sleep(2)

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    # Get the NumPy array representing the image
    image = frame.array

    # Detect number plates
    result_frame, plate_numbers = detect_number_plate(image)

    # Display the result
    cv2.imshow("Number Plate Detection", result_frame)

    # Print the plate numbers found
    if plate_numbers:
        print("Plate Numbers Found:", plate_numbers)

    # Clear the stream for the next frame
    raw_capture.truncate(0)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the camera resources
cv2.destroyAllWindows()
camera.close()
