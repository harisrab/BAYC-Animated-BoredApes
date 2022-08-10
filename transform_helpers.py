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

    # # Inner mouth landmarks transform
    # inner_mouth_xy = landmarks_xy[60:67]
    # inner_mouth_xyz = landmarks_xyz[60:67]

    # inner_mouth_polygon = Polygon(inner_mouth_xy)

    # shrunk_inner_mouth = affinity.scale(
    #     inner_mouth_polygon, xfact=-0.2, yfact=-0.2, origin='centroid')
    # shrunk_inner_mouth = affinity.rotate(
    #     shrunk_inner_mouth, angle=180, origin="centroid")

    # shrunk_inner_mouth_xy = shrunk_inner_mouth.exterior.coords[:-1]

    # final_inner_mouth = []

    # for i, coord in enumerate(shrunk_inner_mouth_xy):
    #     c = [coord[0], coord[1], inner_mouth_xyz[i][2]]
    #     final_inner_mouth.append(c)

    # final_inner_mouth = np.array(final_inner_mouth)

    # # Recover the mouth landmarks
    # face_shape = np.concatenate(
    #     (landmarks_xyz[:60], final_inner_mouth, landmarks_xyz[67:]), axis=0)

    # landmarks_xy = []
    # landmarks_xyz = face_shape

    # Iterate over each point and pair it to form coordinates
    # c = 0
    # for i in range(0, len(face_shape)):
    #     landmarks_xy.append(face_shape[i][:-1])

    # Outer mouth landmarks transform
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

    # Recover the mouth landmarks
    recovered_face_shape = np.concatenate(
        (landmarks_xyz[:48], final_mouth, landmarks_xyz[59:]), axis=0)

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
            outer_mouth_xy = landmarks_xy[48:59]
            outer_mouth_xyz = landmarks_xyz[48:59]

            inner_mouth_xy = landmarks_xy[60:67]
            inner_mouth_xyz = landmarks_xyz[60:67]
            # landmarks_xyz[60:67]

            # Convert to polygon.
            outer_mouth_polygon = Polygon(outer_mouth_xy)
            inner_mouth_polygon = Polygon(inner_mouth_xy)

            # Apply affine transformation. Swell.
            swollen_outer_mouth = affinity.scale(
                outer_mouth_polygon, xfact=2.8, yfact=2.8, origin='centroid')

            swollen_outer_mouth_xy = swollen_outer_mouth.exterior.coords[:-1]

            swollen_inner_mouth = affinity.scale(
                inner_mouth_polygon, xfact=2.8, yfact=2.8, origin='centroid')

            swollen_inner_mouth_xy = swollen_inner_mouth.exterior.coords[:-1]

            final_outer_mouth = []

            for i, coord in enumerate(swollen_outer_mouth_xy):
                c = [coord[0], coord[1], outer_mouth_xyz[i][2]]
                final_outer_mouth.append(c)

            final_inner_mouth = []

            for i, coord in enumerate(swollen_inner_mouth_xy):
                c = [coord[0], coord[1], inner_mouth_xyz[i][2]]
                final_inner_mouth.append(c)

            # final_line = landmarks_xyz[:48] + final_outer_mouth + \
            #     landmarks_xyz[59:60] + final_inner_mouth + landmarks_xyz[67:]
            final_line = landmarks_xyz[:48] + final_outer_mouth + \
                landmarks_xyz[59:]

            # print(landmarks)
            # Restore the line to original file.
            line = ""
            for eachCoord in final_line:
                for eachItem in eachCoord:

                    eachItem = '{:.5f}'.format(round(eachItem, 5))
                    line += eachItem + " "

            print(line)
            f.write(line + "\n")
