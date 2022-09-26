import imageio
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def read_metadata(path):
    epochs = []
    val_idx = []
    reward = []
    sigma = []
    epist = []
    aleat = []
    with open(path, "r") as f:
        for row in f:
            data = np.array(row[:-1].split(",")).astype(np.float32)
            epochs.append(data[0])
            val_idx.append(data[1])
            reward.append(data[2])
            sigma.append(data[3])
            l = len(data) - 4
            epist.append(data[4 : l // 2 + 4])
            aleat.append(data[l // 2 + 4 :])
    return epochs, val_idx, reward, sigma, epist, aleat

def paste_unc(frame, unc, limits, extend_pixels=50, extend_color=0):
    frame = np.concatenate((frame, extend_color * np.zeros((extend_pixels, frame.shape[1], frame.shape[2]))), axis=0)
    im = Image.fromarray(np.uint8(frame))
    draw = ImageDraw.Draw(im)
    fnt = ImageFont.load_default()
    draw.text((350, 170), f'Uncertainty: {unc}', font=fnt)

    y_bounds = [150, 200]
    start = 0
    max_end = 300
    end = int((unc - limits[0]) / (limits[1] - limits[0]) * (max_end - start) + start)
    start_bar = [(start, y_bounds[0]), (start, y_bounds[1])]
    end_bar = [(end, y_bounds[1]), (end, y_bounds[0])]

    draw.polygon(start_bar + end_bar, fill=(255, 0, 0), outline=(0, 255, 0))
    
    return np.array(im)

model = 'ae'
uncertainties_idx = 0
uncertainties_file = f'uncertainties/test/{model}.txt'

filename = f'render/test/{model}/rl-video-episode-{uncertainties_idx}.mp4'
out_video = f'render/unc/{model}/rl-video-episode-{uncertainties_idx}.mp4'

epist = read_metadata(uncertainties_file)[-2][uncertainties_idx]
vid = imageio.get_reader(filename,  'ffmpeg')
vid_metadata = vid.get_meta_data()
fps = vid_metadata['fps']
writer = imageio.get_writer(out_video, fps=fps)

frames = int(vid_metadata['duration'] * fps)
steps = frames // len(epist)
rest = frames % len(epist)
uncert_limits = [min(epist), max(epist)]
for idx, image in enumerate(vid.iter_data()):
    if idx >= rest:
        unc = epist[(idx - rest ) // steps]
    else:
        unc = uncert_limits[0]
    new_image = paste_unc(image, unc, uncert_limits)
    writer.append_data(new_image)

writer.close()
vid.close()
