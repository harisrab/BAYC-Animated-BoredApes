from livePortrets_wGUI import WarpWrapper


vd_lm_path = "C:/Users/admin/Documents/projects/livePortraits/data/warped_points.txt"
im_path = "C:/Users/admin/Documents/projects/livePortraits/images_scaled/doll_pose.jpg"
im_lm_path = "C:/Users/admin/Documents/projects/landmark_editor/ape1.pts"

WarpWrapper(vd_lm_path, im_path, im_lm_path)
