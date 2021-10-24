import torch
import json

class WLASLDataset(torch.utils.data.Dataset):
    def __init__(self, video_folder = '../../data/videos', labels_path = '../../data/WLASL.json'):
        super().__init__()
        with open(labels_path) as f:
            self.labels = json.load(f)
        self.video_folder = video_folder
        self.org_labels = []
        self.glosses = []
        for i, gloss in enumerate(self.labels):
            word = gloss['gloss']
            self.glosses.append(word)
            for instance in gloss['instances']:
                instance.update(('gloss', i))
                self.org_labels.append(instance)

    def load_rgb_frames_from_video(self, vid, start, num=1e9, resize=(256, 256)):
        video_path = os.path.join(self.video_folder, vid + '.mp4')
        vidcap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for offset in range(min(num, int(total_frames - start))):
            success, img = vidcap.read()
            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            if w > 256 or h > 256:
                img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))
            img = (img / 255.) * 2 - 1
            frames.append(img)
        return np.asarray(frames, dtype=np.float32)

    def __getitem__(self, idx):
        # frames: b, c, t, h, w
        labels = self.org_labels[idx]
        start = labels['frame_start']
        vid = labels['video_id']
        frames = self.load_rgb_frames_from_video(vid, start)
        return frames, labels['gloss']

    def __len__(self):
        return len(self.org_labels)