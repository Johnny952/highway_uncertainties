from moviepy.editor import VideoFileClip
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Change Video Clip FPS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Environment Config
    config = parser.add_argument_group("Config")
    config.add_argument(
        "-IP",
        "--input-path",
        type=str,
        required=True,
        help="Path to the video to change fps",
    )
    config.add_argument(
        "-OP",
        "--output-path",
        type=str,
        required=False,
        help="Path to the video with fps changed",
    )
    config.add_argument(
        "-F",
        "--fps",
        type=int,
        required=False,
        default=30,
        help="Frames per second",
    )
    config.add_argument(
        "-G",
        "--gif",
        action="store_true",
        help="Whether to convert to gif or not",
    )
    args = parser.parse_args()

    output_path = args.input_path if args.output_path is None else args.output_path
    clip = VideoFileClip(args.input_path) 
    
    if args.gif:
        clip.write_gif(output_path, fps=args.fps)
    else:
        clip.write_videofile(output_path, fps=args.fps)
    #clip.reader.close()