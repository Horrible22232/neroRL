import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle

class VideoRecorder:
    """Initializes the VideoRecorder that sets some member variables."""
    def __init__(self, video_path):
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.scale = 0.4
        self.thickness = cv2.FILLED
        self.text_color = (255, 255, 255)
        self.margin = 2
        self.width = 420
        self.height = 420
        self.video_path = video_path
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    def render_video(self, trajectory_data):
        """Draws a debug frame that is concatenated to the Obstacle Tower frame. All resulting frames are output to a video."""
        pickle.dump(trajectory_data["vis_obs"], open("frames.save", "wb"))
        # Init VideoWriter
        out = cv2.VideoWriter(self.video_path, self.fourcc,trajectory_data["frame_rate"], (self.width * 2, self.height))
        for i in range(len(trajectory_data["vis_obs"])):
            # Obstacle Tower frame
            env_frame = trajectory_data["vis_obs"][i][...,::-1] # Convert RGB to BGR, OpenCV expects BGR
            env_frame = cv2.resize(env_frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            # Debug frame
            debug_frame = np.zeros((420, 420, 3), dtype=np.uint8)
            # # Current floor
            # self.draw_text_overlay(debug_frame, 5, 20, trajectory_data["floors"][i], "floor")
            # # Total reward
            # self.draw_text_overlay(debug_frame, 215, 20, round(trajectory_data["rewards"][i], 3), "total reward")
            # # Remaining time
            # self.draw_text_overlay(debug_frame, 5, 40, trajectory_data["remaining_times"][i], "time")
            # # Has key
            # self.draw_text_overlay(debug_frame, 215, 40, trajectory_data["keys"][i], "key")
            # # Value Function
            # self.draw_text_overlay(debug_frame, 5, 60, round(trajectory_data["values"][i], 5), "value")
            # # Current entropy
            # self.draw_text_overlay(debug_frame, 215, 60, round(sum(trajectory_data["entropies"][i]), 5), "entropy")
            # Selected action
            # for index, action in enumerate(video_dict["actions"][i]):
            #     self.draw_text_overlay(debug_frame, 5 + (210 * (index % 2)), 95 + (20 * (int(index / 2))), str(action) + " " + Game.action_names[index][action], "action " + str(index))
            # # Action probabilities
            # next_y = 100
            # for x, probs in enumerate(trajectory_data["probs"][i]):
            #     self.draw_text_overlay(debug_frame, 5 , next_y, round(trajectory_data["entropies"][i][x], 5), "entropy " + str(x))
            #     next_y += 20
            #     for y, prob in enumerate(probs.squeeze(dim=0)):
            #         self.draw_bar(debug_frame, 0, next_y, round(prob.item(), 5), str(Game.action_names[x][y]), y == trajectory_data["actions"][i][x])
            #         next_y += 20
            #     next_y += 10

            # Concatenate ot and debug frames
            output_image = np.hstack((env_frame, debug_frame))

            # Write frame
            out.write(output_image)
        out.release()

    def draw_text_overlay(self, img, x, y, value, label):
        """Renders text (label + value) on a black rectangle."""
        bg_color = (0, 0, 0)
        pos = (x, y)
        text = label + ": " + str(value)
        txt_size = cv2.getTextSize(text, self.font_face, self.scale, self.thickness)
        end_x = pos[0] + txt_size[0][0] + self.margin
        end_y = pos[1] - txt_size[0][1] - self.margin
        cv2.rectangle(img, pos, (end_x, end_y), bg_color, self.thickness)
        cv2.putText(img, text, pos, self.font_face, self.scale, self.text_color, 1, cv2.LINE_AA)

    def draw_bar(self, img, x, y, ratio, label, chosen):
        """Renders a green rectangle to represent a bar that crosses the whole debug frame depending on the ratio of an action probability."""
        if chosen:
            bg_color = (0, 255, 0)
        else:
            bg_color = (0, 69, 255)
            
        pos = (x, y)
        text = label + ": " + str(ratio)
        txt_size = cv2.getTextSize(text, self.font_face, self.scale, self.thickness)
        end_x = int(420 * ratio)
        end_y = pos[1] - txt_size[0][1] - self.margin
        cv2.rectangle(img, pos, (end_x, end_y), bg_color, self.thickness)
        cv2.putText(img, text, (x + 5, y), self.font_face, self.scale, self.text_color, 1, cv2.LINE_AA)
