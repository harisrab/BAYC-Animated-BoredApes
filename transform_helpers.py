from shapely import affinity
from shapely.geometry import Polygon
import numpy as np


def ShrinkMouthTransform(face_shape):
    """Single frame landmarks. Mouth is """
    print("(Shrink Function) The Face Shape: ")
    print(face_shape)

    landmarks_xy = []
    landmarks_xyz = face_shape

    # Iterate over each point and pair it to form coordinates
    c = 0
    for i in range(0, len(face_shape)):
        landmarks_xy.append(face_shape[i][:-1])

    mouth_xy = landmarks_xy[48:59]
    mouth_xyz = landmarks_xyz[48:59]

    mouth_polygon = Polygon(mouth_xy)

    shrunk_mouth = affinity.scale(
        mouth_polygon, xfact=-0.4, yfact=-0.4, origin='centroid')
    shrunk_mouth = affinity.rotate(shrunk_mouth, angle=180, origin="centroid")

    shrunk_mouth_xy = shrunk_mouth.exterior.coords[:-1]

    final_mouth = []

    for i, coord in enumerate(shrunk_mouth_xy):
        c = [coord[0], coord[1], mouth_xyz[i][2]]
        final_mouth.append(c)

    final_mouth = np.array(final_mouth)
    # print(final_mouth)
    print(type(landmarks_xyz))
    print("Landmarks XYZ (1st) = ", landmarks_xyz[:48])
    print("final Mouth         = ", final_mouth)
    print("Landmarks XYZ (2nd) = ", landmarks_xyz[59:])

    recovered_face_shape = np.concatenate(
        (landmarks_xyz[:48], final_mouth, landmarks_xyz[59:]), axis=0)

    # recovered_face_shape = landmarks_xyz[:48] + final_mouth + landmarks_xyz[59:]

    print("Recovered_face_shape =====> ", recovered_face_shape)

    return recovered_face_shape


def SwellMouthTransform(args, fls_names):
    """Frame for each sample. Reverse Transform for mouth on each"""

    outputLandmarksPath = f"./ape_src/{fls_names[0]}"

    print("Swell Function: ", fls_names)
    lines = []

    with open(outputLandmarksPath, "r") as f:
        lines = f.readlines()

    with open(outputLandmarksPath, "w") as f:
        # Go through all audio samples

        for line in lines:

            # Split the line into a list of strings and remove new line character
            line_items = line[:-1].split(" ")
            line_items = list(map(float, line_items))
            # lines_items = [float(item) for item in line_items]

            landmarks_xy = []
            landmarks_xyz = []

            # Iterate over each point and pair it to form coordinates
            c = 0
            landmark = []
            for i in range(0, len(line_items)):
                landmark.append(line_items[i])

                if (c == 2):
                    landmarks_xy.append(landmark[:-1])
                    landmarks_xyz.append(landmark[:])
                    landmark = []
                    c = 0

                else:
                    c += 1

            # print(len(landmarks_xy))
            # print(len(landmarks_xyz))

            # Pick out mouth coordinates.
            mouth_xy = landmarks_xy[48:59]
            mouth_xyz = landmarks_xyz[48:59]

            # Convert to polygon.
            mouth_polygon = Polygon(mouth_xy)

            # Apply affine transformation. Swell.
            swollen_mouth = affinity.scale(
                mouth_polygon, xfact=2.5, yfact=2.5, origin='centroid')

            swollen_mouth_xy = swollen_mouth.exterior.coords[:-1]

            print(len(mouth_xy))
            print(len(mouth_xyz))
            print(len(swollen_mouth_xy))

            final_mouth = []

            for i, coord in enumerate(swollen_mouth_xy):
                c = [coord[0], coord[1], mouth_xyz[i][2]]
                final_mouth.append(c)

            # print(final_mouth)

            final_line = landmarks_xyz[:48] + final_mouth + landmarks_xyz[59:]

            # print(landmarks)
            # Restore the line to original file.
            line = ""
            for eachCoord in final_line:
                for eachItem in eachCoord:

                    eachItem = '{:.5f}'.format(round(eachItem, 5))
                    line += eachItem + " "

            print(line)
            f.write(line + "\n")

            # break
            # break
