import cv2
import os

# Path to the folder containing PNG images
folder_path = "path_to_your_folder"

# Get the list of image files in the folder
images = [img for img in os.listdir(folder_path) if img.endswith(".png")]
images.sort()  # Ensure they are in the correct order

# Set the video file name and frame size (you can adjust this as per your images)
output_video = "output_video.mp4"

# Assuming all images have the same size
frame = cv2.imread(os.path.join(folder_path, images[0]))
height, width, layers = frame.shape
fps = 30  # Set the frames per second for the video

# Create a video writer object
video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Iterate over each image and add it to the video
for image in images:
    img_path = os.path.join(folder_path, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Release the video writer
video.release()
cv2.destroyAllWindows()