import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
import argparse

def cvat_xml_to_keypoints_dict(cvat_xml):
    """

    """

    tree = ET.parse(cvat_xml)
    root = tree.getroot()
    videos = [child for child in root if child.tag == 'image']

    video_keypoints = {}
    noise = np.random.rand(1, 2) / 50

    for video in videos:
      file_name = video.attrib['name'][:-6] + ".mp4"
      kp_str = video[0].attrib['points'].split(';')
      kp_np = np.array([[float(kp.split(',')[0]), float(kp.split(',')[1])] for kp in kp_str])[:, np.newaxis, :].astype(np.float32)
      kp_jitter = np.vstack((kp_np[0] + noise, kp_np[1] + noise))[:, np.newaxis, :].astype(np.float32)
        # optical flow requires numpy array of type float32, shape (N, 1, 2)

      # Save file name and jittered keypoints to dictionary
      video_keypoints[file_name] = kp_jitter

    return video_keypoints

def jitter_keypoints(keypoints, n):
    np.random.seed(123)
    
    pass


def sparse_optical_flow_from_keypoints(video: str, keypoints):
    """

    """

    # TODO: save trajectories as lists of (x, y) coordinates

    # Count # frames in video
    # cap = cv.VideoCapture(video)
    # success = True
    # count = 0
    # while success:
    #     success , frame =cap.read()
    #     if not success:
    #         break
    #     count=count+1
    # print("# frames: ", count)

    cap = cv.VideoCapture(video)

    # parameters for Lucas-Kanade sparse optical flow
    # https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
    lk_params = dict(winSize = (15, 15), # (15, 15)
                        maxLevel = 4, # 2
                        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20000, 0.0))

    # Choose random color for optical flow trajectories
    color = np.random.randint(0, 255, (500, 3))

    # Read in first frame
    success, prev_frame = cap.read()

    # Initialized keypoints to track
    prev_pts = keypoints

    # Convert first frame to grayscale: don't need color for edge detection, save computation
    prev_frame_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    # Create mask initialized to 0's with same dimension as first frame for later drawing
    mask = np.zeros_like(prev_frame)

    while(True):
        # read next frame
        success, frame = cap.read()
        if not success:
            print("End of video")
            break

        # convert to grayscale
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # compute optical flow
        # https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
        new_pts, status, error = cv.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, prev_pts, None, **lk_params)
        # print("new_pts: ", type(new_pts), new_pts.shape, "\n", new_pts)
        # print("status: ", type(status), status.shape, "\n", status)

        # select good feature points for previous and new frames
        good_new = new_pts[status == 1]
        good_prev = prev_pts[status == 1]
        print("good_new: ", good_new, "\ngood_prev: ", good_prev)

        # draw optical flow trajectories
        for i, (new, prev) in enumerate(zip(good_new, good_prev)):

            # ravel() returns contiguous flattened array as (x, y) coordinates for points
            # i.e. new/old.ravel() -> [x, y]
            a, b = new.ravel()
            c, d = prev.ravel()
            #print("new: ", new, "\nnew.ravel(): ", new.ravel(), "\na: ", a, "\nb: ", b)

            # draw lines between new and old positions
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)

            # draw filled circles at new feature point positions
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        # overlay optical flow trajectories on original frame
        output = cv.add(frame, mask)

        # update previous frame and previous feature points
        prev_frame_gray = frame_gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

        # open new window and show output
        cv.imshow("sparse optical flow", output)
        # frames are read by intervals of 10 milliseconds.
        # The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    # Free resources and close all windows
    cap.release()
    cv.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("xml", type=str, help="path to annotation CVAT XML")
    parser.add_argument("video", type=str, help="path to video file")
    args = parser.parse_args()

    video_keypoints = cvat_xml_to_keypoints_dict(args.xml)
    sparse_optical_flow_from_keypoints(args.video, video_keypoints[args.video])


if __name__ == "__main__":
    main()

