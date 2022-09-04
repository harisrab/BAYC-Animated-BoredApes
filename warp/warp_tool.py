# import necessary packages and functions
import cv2, dlib, os
import mls as mls
import numpy as np
import math

from warp.helper_funcs import getFaceRect, landmarks2numpy, createSubdiv2D, calculateDelaunayTriangles, insertBoundaryPoints
from warp.helper_funcs import getVideoParameters, warpTriangle, getRigidAlignment
from warp.helper_funcs import teethMaskCreate, erodeLipMask, getLips, getLipHeight, drawDelaunay
from warp.helper_funcs import mainWarpField, copyMouth
from warp.helper_funcs import hallucinateControlPoints, getInterEyeDistance

global im_fn, video_fn

""" Select the trackbar parameters"""
# images
imageArray = ['',        # indicates the first frame of the video
              "Frida1.jpg", "Frida2.jpg",
              "MonaLisa.jpg", "Rosalba_Carriera.jpg",
              "IanGillan.jpg", "AfghanGirl.jpg"]
imageIndex = 0 
maximageIndex = len(imageArray) -1
imageTrackbarName = "Images: \n 0: First frame \n 1: Frida Kahlo #1 \n 2: Frida Kahlo #2 \n 3: Mona Lisa \n 4: Rosalba Carriera \n 5: Ian Gillan \n 6: Afghan Girl"

# video
videoArray = ["anger.avi", "smile.avi", "teeth_smile.avi", "surprise.avi"]
videoIndex = 0
maxvideoIndex = len(videoArray) -1
videoTrackbarName = "Videos: \n 0: Anger \n 1: Smile \n 2: Smile with teeth \n 3: Surprise"


# state of the process
OnOFF = 0
maxOnOFF = 1
processTrackbarName = "Process: \n 0 - Still image \n 1 - Animate"

# create a folder for results, if it doesn't exist yet
os.makedirs("video_generated", exist_ok=True) 

""" Get face and landmark detectors"""
faceDetector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "../common/shape_predictor_68_face_landmarks.dat"  # Landmark model location
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)  

""" Function for control window """
# Bug: multi-line titles of the OpenCV trackbars don't work. So instead the correspondence 
# between indices and images is listed on an additional image in control window
controlWindowName = 'Control'


""" A "gatekeeper" function """
def warp(pred_video_landmarks, im_path, im_lm_path):
    # Function that checks the image and video and chooses between  showing the original video, image 
    # or running the main algorithm
    print("Pred Video Landmarks DS: ", pred_video_landmarks.shape)
    print("Pred Video Landmarks Scale: ", pred_video_landmarks[0])
    print("im_path: ", im_path)
    print("im_lm_path: ", im_lm_path)

    global OnOFF # needed to drop "Process" trackbar to 0-state in the end
    
    video_fn = "anger_0001.avi"
    # Create a VideoCapture object
    import cv2 as cv
     
    video_path = os.path.join("/home/haris/BAYC-Animated-BoredApes/warp", "video_recorded", video_fn)
    print("Video Path: ", video_path)

    cap = cv.VideoCapture(video_path)

    # Check if camera is opened successfully
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")
  
    # read and process the image
    import cv2 as cv
    im = cv.imread(im_path)
    
    if im is None:
        print("Unable to read the photo")
    else:
        # scale the image to have a 600 pixel height
        scaleY = 600./im.shape[0]
        im = cv2.resize(src=im, dsize=None, fx=scaleY, fy=scaleY, interpolation=cv2.INTER_LINEAR )  

    onOFF = 1
    runLivePortrets(im_path, im, pred_video_landmarks, cap, im_lm_path)



""" Main algorithm """
def runLivePortrets(im_path, im, pred_video_landmarks, cap, im_lm_path):
    global OnOFF # needed to drop "Process" trackbar to 0-state in the end

    ########## Get the parameters and landmarks of the image #########
    im_height, im_width, im_channels = im.shape
    im_fn = os.path.basename(im_path)

    # detect the face and the landmarks
    newRect = getFaceRect(im, faceDetector)
    landmarks_im = landmarks2numpy(landmarkDetector(im, newRect))
    
    print("Original LM: ", landmarks_im)
    print("Original LM.shape: ", landmarks_im.shape)
    
    landmarks_im = []

    # Pick original facial landmarks that were already detected.
    with open(im_lm_path, 'r') as f:
        # landmarks_im = f.readlines()[3:-1]
        
        lines = f.readlines()
        pts = []
        for i in range(3, 3 + 68):
            line = lines[i]
            line = line[:-1].split(' ')
            pts += [float(item) for item in line]
        pts0 = np.array(pts).reshape((68, 2))

        landmarks_im = pts0
        

    landmarks_vd_np = []
    
    for everyFrame in pred_video_landmarks:
        pts = []

        for i in range(0, len(everyFrame), 2):
            point = [everyFrame[i], everyFrame[i + 1]]
            pts.append(point)

        landmarks_vd_np.append(np.array(pts))

    landmarks_vd_np = np.array(landmarks_vd_np)
    print("Landmarks shape: ", landmarks_im.shape)

    print("Video Landmarks: ", landmarks_vd_np[0])
    print("Video Landmarks.shape: ", landmarks_vd_np.shape)

    video_fn = "Angry.avi"

    ###########  Get the parameters of the driving video ##########
    # Obtain default resolutions of the frame (system dependent) and convert from float to integer.
    #(time_video, length_video, fps, frame_width, frame_height) = getVideoParameters(cap)

    ############### Create new video ######################
    #output_fn = im_fn[:-4] + "_" + video_fn
    #out = cv2.VideoWriter(os.path.join("video_generated", output_fn),
     #                     cv2.VideoWriter_fourcc('M','J','P','G'), fps, (im_width, im_height))

    ############### Initialize the algorithm parameters #################
    frame = [] 
    tform = [] # similarity transformation that alignes video frame to the input image
    srcPoints_frame = []
    numCP = 68 # number of control points
    newRect_frame = []

    # Optical Flow
    points=[]
    pointsPrev=[] 
    pointsDetectedCur=[] 
    pointsDetectedPrev=[]
    eyeDistanceNotCalculated = True
    eyeDistance = 0
    isFirstFrame = True

    ############### Go over frames #################
    count = 0

    for i, eachFrame_l in enumerate(landmarks_vd_np):
        ret, frame = cap.read()

        # frame_l = landmarks_vd_np[count]
        
        print("Frame: ", count)
        count += 1

        print("Each new frame dimension: ", eachFrame_l.shape)

        # if ret == False:
        #     # before breaking the loop drop OnOFF to zero and update the "Process trackbar"
        #     OnOFF = 0
        #     # cv2.createTrackbar("Process", controlWindowName, OnOFF, maxOnOFF, runProcess)
        #     break  
        # else:
        # the orginal video doesn't have information about the frame orientation, so  we need to rotate it
        frame = np.rot90(frame, 3).copy()

        # initialize a new frame for the input image
        im_new = im.copy()          

        ###############    Similarity alignment of the frame #################
        # detect the face (only for the first frame) and landmarks
        if isFirstFrame: 
            newRect_frame = getFaceRect(frame, faceDetector)

            # [1] Pick out facial landmarks for the frame

            #landmarks_frame_init = landmarks2numpy(landmarkDetector(frame, newRect_frame))
            #print("Original Detection of Landmarks.shape: ", landmarks_frame_init.shape)

            landmarks_frame_init = eachFrame_l

            # landmarks_frame_init = eachFrame_l

            # compute the similarity transformation in the first frame
            tform = getRigidAlignment(landmarks_frame_init, landmarks_im)    
        else:
            # [1] Pick out facial landmarks for the frame

            #landmarks_frame_init = landmarks2numpy(landmarkDetector(eachFrame_l, landmarks_im))
            landmarks_frame_init = eachFrame_l

            if np.array_equal(tform, []):
                print("ERROR: NO SIMILARITY TRANSFORMATION")

        # Apply similarity transform to the frame
        frame_aligned = np.zeros((im_height, im_width, im_channels), dtype=im.dtype)
        frame_aligned = cv2.warpAffine(frame, tform, (im_width, im_height))

        # Change the landmarks locations
        landmarks_frame = np.reshape(landmarks_frame_init, (landmarks_frame_init.shape[0], 1, landmarks_frame_init.shape[1]))
        landmarks_frame = cv2.transform(landmarks_frame, tform)
        landmarks_frame = np.reshape(landmarks_frame, (landmarks_frame_init.shape[0], landmarks_frame_init.shape[1]))

        print("Current Directory: ", os.getcwd())
        cwd = os.path.join(os.getcwd(), "..", "..", "..", "warp")
        

        # hallucinate additional control points
        if isFirstFrame: 
            (subdiv_temp, dt_im, landmarks_frame) = hallucinateControlPoints(landmarks_init = landmarks_frame, 
                                                                            im_shape = frame_aligned.shape, 
                                                                            INPUT_DIR=cwd, 
                                                                            performTriangulation = True)
            # number of control points
            numCP = landmarks_frame.shape[0]
        else:
            landmarks_frame = np.concatenate((landmarks_frame, np.zeros((numCP-68,2))), axis=0)

        ############### Optical Flow and Stabilization #######################
        # Convert to grayscale.
        imGray = cv2.cvtColor(frame_aligned, cv2.COLOR_BGR2GRAY)

        # prepare data for an optical flow
        if (isFirstFrame==True):
            [pointsPrev.append((p[0], p[1])) for p in landmarks_frame[68:,:]]
            [pointsDetectedPrev.append((p[0], p[1])) for p in landmarks_frame[68:,:]]
            imGrayPrev = imGray.copy()

        # pointsDetectedCur stores results returned by the facial landmark detector
        # points stores the stabilized landmark points

        print("### Landmarks Frame: ", np.unique(np.array(landmarks_frame)))
        print("## Landmarks Frame.shape: ", landmarks_frame.shape)

        points = []
        pointsDetectedCur = []
        [points.append((p[0], p[1])) for p in landmarks_frame[68:,:]]
        [pointsDetectedCur.append((p[0], p[1])) for p in landmarks_frame[68:,:]]
        print("Points Detected Current: ", pointsDetectedCur)
        print("Points Detected Current.shape: ", np.array(pointsDetectedCur).shape)
        # Convert to numpy float array
        pointsArr = np.array(points, np.float32)
        pointsPrevArr = np.array(pointsPrev,np.float32)

        # If eye distance is not calculated before
        if eyeDistanceNotCalculated:
            eyeDistance = getInterEyeDistance(landmarks_frame)
            eyeDistanceNotCalculated = False

        dotRadius = 3 if (eyeDistance > 100) else 2
        sigma = eyeDistance * eyeDistance / 400
        s = 2*int(eyeDistance/4)+1

        #  Set up optical flow params
        lk_params = dict(winSize  = (s, s), maxLevel = 5, criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))
        pointsArr, status, err = cv2.calcOpticalFlowPyrLK(imGrayPrev,imGray,pointsPrevArr,pointsArr,**lk_params)
        sigma = 100

        # Converting to float and back to list
        points = np.array(pointsArr,np.float32).tolist()   

        print("pointsArr: ", pointsArr)
        print("pointsArr.shape: ", pointsArr.shape)

        # Facial landmark points are the detected landmark and additional control points are tracked landmarks  
        landmarks_frame[68:,:] = pointsArr
        landmarks_frame = landmarks_frame.astype(np.int32)

        # getting ready for the next frame
        imGrayPrev = imGray        
        pointsPrev = points
        pointsDetectedPrev = pointsDetectedCur

        ############### End of Optical Flow and Stabilization #######################

        # save information of the first frame for the future
        if isFirstFrame: 
            # hallucinate additional control points for a still image
            landmarks_list = landmarks_im.copy().tolist()
            for p in landmarks_frame[68:]:
                landmarks_list.append([p[0], p[1]])
            srcPoints = np.array(landmarks_list)
            srcPoints = insertBoundaryPoints(im_width, im_height, srcPoints) 
            
            print("1st Frame landmarks: ", landmarks_list)
            print("1st Frame srcPoints: ", srcPoints.shape)
            lip_height = getLipHeight(landmarks_im)            
            (_, _, maskInnerLips0, _) = teethMaskCreate(im_height, im_width, srcPoints)    
            mouth_area0=maskInnerLips0.sum() / 255  

            # get source location on the first frame
            srcPoints_frame = landmarks_frame.copy()
            srcPoints_frame = insertBoundaryPoints(im_width, im_height, srcPoints_frame)  

            # Write the original image into the output file
            cwd = os.path.join(os.getcwd(), "..", "..", "..", "warp")
            save_image_path = f"{cwd}/tmp_output/out_{i}.jpg" 
            cv2.imwrite(save_image_path, im_new)

            # Display the original image           
            # cv2.imshow('Live Portrets', im_new)
            # # stop for a short while
            # if cv2.waitKey(1) & 0xFF == 1:#10:
            #     continue
            # # Go out of the loop if OnOFF trackbar was changed to 0
            # if OnOFF==0:
            #     break

            # no need in additional wraps for the first frame
            isFirstFrame = False
            continue

        ############### Warp Field #######################         
        print("Landmarks Frame Shape: ", landmarks_frame.shape)
        dstPoints_frame = landmarks_frame
        dstPoints_frame = insertBoundaryPoints(im_width, im_height, dstPoints_frame)
    
        print("landmarks_frame: ", landmarks_frame)
        print("landmarks_frame.shape: ", landmarks_frame.shape)

        print("dstPoints_frame: ", dstPoints_frame)
        print("dstPoints_frame.shape: ", dstPoints_frame.shape)

        # get the new locations of the control points
        dstPoints = dstPoints_frame - srcPoints_frame + srcPoints   
        
        print("DT: ", dt_im)
        print("DT.shape: ", np.array(dt_im).shape)
        # get a warp field, smoothen it and warp the image
        im_new = mainWarpField(im, srcPoints, dstPoints, dt_im)       

        ############### Mouth cloning #######################
        # get the lips and teeth mask
        (maskAllLips, hullOuterLipsIndex, maskInnerLips, hullInnerLipsIndex) = teethMaskCreate(im_height, im_width, dstPoints)
        mouth_area = maskInnerLips.sum()/255        

        # erode the outer mask based on lipHeight
        maskAllLipsEroded = erodeLipMask(maskAllLips, lip_height)
        
        # smooth the mask of inner region of the mouth
        maskInnerLips = cv2.GaussianBlur(np.stack((maskInnerLips,maskInnerLips,maskInnerLips), axis=2),(3,3), 10)

        # clone/blend the moth part from 'frame_aligned' if needed (for mouth_area/mouth_area0 > 1)
        im_new = copyMouth(mouth_area, mouth_area0,
                            landmarks_frame, dstPoints,
                            frame_aligned, im_new,
                            maskAllLipsEroded, hullOuterLipsIndex, maskInnerLips)           


        save_image_path = os.path.join(f"{cwd}/tmp_output/out_{i}.jpg")
        cv2.imwrite(save_image_path, im_new)

        # Write the frame into the file 'output.avi'
        #out.write(im_new)

        # Display the resulting frame    
        # cv2.imshow('Live Portrets', im_new)

        onOFF = 1
        continue
        
        # stop for a short while
        if cv2.waitKey(1) & 0xFF == 1:#10:
            continue
            
        # Go out of the loop if OnOFF trackbar was changed to 0
        if OnOFF==0:
            break  


    # When everything is done, release the video capture and video write objects
    cap.release()
    out.release()
          
