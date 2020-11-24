import cv2
import ray


@ray.remote
class VideoWriterActor:
    def __init__(self,
                 actor_setup_function,

                 video_framerate: float = 30,
                 video_fourcc: str = "HFYU",

                 frame_repeat: int = 1,
                 end_frame_still_frames: int = 1):
        """

        Args:
            actor_setup_function:
            frame_repeat: Each frame in the input video is repeated this number of times in the output video file.
            end_frame_still_frames: The number of additional repetitions of the final frame (beyond the `frame_repeat`
                                    repeats that are done anyways) in the output video
        """
        actor_setup_function()

        self.frame_repeat = frame_repeat
        self.end_frame_still_frames = end_frame_still_frames
        self.video_fourcc = video_fourcc
        self.video_framerate = video_framerate

    def write_video(self, path, frames):
        frame_width = frames[0].shape[1]
        frame_height = frames[0].shape[0]

        fourcc_code = cv2.VideoWriter_fourcc(*self.video_fourcc)
        video_writer = cv2.VideoWriter(path,
                                       fourcc_code,
                                       self.video_framerate,
                                       (frame_width, frame_height))
        for index in range(len(frames)):
            for _ in range(self.frame_repeat):
                video_writer.write(frames[index])
            if index == len(frames) - 1:
                for _ in range(self.end_frame_still_frames):
                    video_writer.write(frames[index])

        video_writer.release()
