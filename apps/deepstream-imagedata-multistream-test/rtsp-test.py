import cv2

# RTSP stream URL
rtsp_url = "rtsp://admin:1960@192.168.3.242"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Fail openin RTSP stream")
else:
    print("RTSP stream opened successfully")

# Display the stream
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("RTSP Stream", frame)

    # Press 'q' to exit the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
