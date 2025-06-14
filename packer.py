import glob
import os
import re
import cv2

def main():
    folder = "./vid"
    files = glob.glob(os.path.join(folder, "*.ppm"))
    def sort_key(filename):
        base = os.path.basename(filename)
        match = re.search(r'(\d+(?:\.\d+)?)', base)
        return float(match.group(1)) if match else 0

    files.sort(key=sort_key)
    files = files[::-1]
    
    if not files:
        raise ValueError("No .ppm files found.")

    # Read the first image to determine frame dimensions
    frame = cv2.imread(files[0])
    if frame is None:
        raise ValueError("Failed to read the first image.")

    height, width, _ = frame.shape
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

    for filename in files:
        frame = cv2.imread(filename)
        if frame is None:
            continue
        video.write(frame)

    video.release()

if __name__ == "__main__":
    main()