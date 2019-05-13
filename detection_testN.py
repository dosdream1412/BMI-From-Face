import os
import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import glob



def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

# Load image
list_file = os.listdir("thin")
for filename in list_file:
    img_path = filename
    image = io.imread(os.path.join('thin',img_path))

    # Detect faces
    detected_faces = detect_faces(image)

    # Crop faces and plot
    for n, face_rect in enumerate(detected_faces):
        face = Image.fromarray(image).crop(face_rect)
        plt.subplot(1, len(detected_faces), n+1)
        plt.axis('off')
        face.save('./detec_thin/detec_'+filename, 'png')
        #face.show()
