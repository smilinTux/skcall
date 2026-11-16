"""MuseTalk 1.5 realtime benchmark, device-agnostic (Intel XPU / CUDA / CPU).

Runs from inside a MuseTalk checkout (models/ downloaded). Avoids mmpose/mmcv:
face landmarks come from the `face_alignment` package (68 iBUG points, same
order as DWPose face keypoints 23:91 that MuseTalk uses), run once on CPU at
avatar preparation. The realtime path (whisper-tiny features, PE, UNet, VAE
decode, CPU blend) is MuseTalk's own code.

Usage:
  ZE_AFFINITY_MASK=0 python bench_musetalk.py --portrait p.png --audio a.wav \
      --device xpu --batch 8 --out sample.mp4
"""
import argparse, json, math, os, queue, subprocess, sys, threading, time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from transformers import WhisperModel  # noqa: E402
from musetalk.utils.utils import load_all_model, datagen  # noqa: E402
from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402
from musetalk.utils.blending import get_image_prepare_material, get_image_blending  # noqa: E402
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402


def sync(dev):
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def peak_mem(dev):
    if dev.type == "xpu":
        return torch.xpu.max_memory_allocated() / 2**20, torch.xpu.max_memory_reserved() / 2**20
    if dev.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20
    return 0, 0


def musetalk_bbox(frame, extra_margin=10):
    """Replicates preprocessing.get_landmark_and_bbox with face_alignment landmarks."""
    import face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False)
    lms = fa.get_landmarks_from_image(frame[:, :, ::-1])
    if not lms:
        raise SystemExit("no face found")
    lm = lms[0].astype(np.int32)
    half_face_coord = lm[29].copy()
    half_face_dist = np.max(lm[:, 1]) - half_face_coord[1]
    upper_bond = max(0, half_face_coord[1] - half_face_dist)
    x1, y1, x2, y2 = int(np.min(lm[:, 0])), int(upper_bond), int(np.max(lm[:, 0])), int(np.max(lm[:, 1]))
    y2 = min(y2 + extra_margin, frame.shape[0])
    return [x1, y1, x2, y2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portrait", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--max-side", type=int, default=720, help="downscale portrait so the published frame is this tall/wide")
    ap.add_argument("--out", default=None)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--first-chunk-s", type=float, default=2.0, help="audio length for the time-to-first-frame probe")
    args = ap.parse_args()

    dev = torch.device(args.device)
    res = {"device": str(dev), "batch": args.batch, "torch": torch.__version__}
    if dev.type == "xpu":
        res["gpu"] = torch.xpu.get_device_name(0)
    elif dev.type == "cuda":
        res["gpu"] = torch.cuda.get_device_name(0)

    # ---------- avatar preparation (one-off per portrait, cacheable) ----------
    t0 = time.time()
    frame = cv2.imread(args.portrait)
    h, w = frame.shape[:2]
    s = args.max_side / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
    res["frame_size"] = list(frame.shape[1::-1])
    bbox = musetalk_bbox(frame)
    res["bbox"] = bbox
    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    mask, crop_box = get_image_prepare_material(frame, bbox, fp=fp, mode="jaw")
    res["prep_landmark_parse_s"] = round(time.time() - t0, 2)

    t0 = time.time()
    vae, unet, pe = load_all_model(device=dev)
    vae.vae = vae.vae.half().to(dev)
    vae.device = dev
    unet.model = unet.model.half().to(dev)
    pe = pe.half().to(dev)
    whisper = WhisperModel.from_pretrained("models/whisper").to(device=dev, dtype=torch.float16).eval()
    audio_processor = AudioProcessor(feature_extractor_path="models/whisper")
    timesteps = torch.tensor([0], device=dev)
    x1, y1, x2, y2 = bbox
    crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
    with torch.no_grad():
        latent = vae.get_latents_for_unet(crop)
    latents_cycle = [latent]
    sync(dev)
    res["model_load_s"] = round(time.time() - t0, 2)

    def feats(path):
        f, n = audio_processor.get_audio_feature(path, weight_dtype=torch.float16)
        return audio_processor.get_whisper_chunk(f, dev, torch.float16, whisper, n, fps=args.fps,
                                                 audio_padding_length_left=2, audio_padding_length_right=2)

    @torch.no_grad()
    def gen_batches(chunks):
        for wb, lb in datagen(chunks, latents_cycle, args.batch, device=dev):
            a = pe(wb.to(dev))
            lb = lb.to(device=dev, dtype=unet.model.dtype)
            pred = unet.model(lb, timesteps, encoder_hidden_states=a).sample
            yield vae.decode_latents(pred.to(dtype=vae.vae.dtype))

    def blend(face):
        f = cv2.resize(face.astype(np.uint8), (x2 - x1, y2 - y1))
        return get_image_blending(frame.copy(), f, bbox, mask, crop_box)

    # warmup (kernel JIT on XPU is slow the first time)
    for _ in range(args.warmup):
        for _ in gen_batches(feats(args.audio)[: args.batch * 2]):
            pass
    sync(dev)
    if dev.type == "xpu":
        torch.xpu.reset_peak_memory_stats()
    elif dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---------- time to first frame on a short first chunk ----------
    import soundfile as sf
    data, sr = sf.read(args.audio)
    first = os.path.join(os.path.dirname(os.path.abspath(args.out or args.audio)), "_first_chunk.wav")
    sf.write(first, data[: int(sr * args.first_chunk_s)], sr)
    t0 = time.time()
    ch = feats(first)
    sync(dev)
    t_feat = time.time() - t0
    out0 = next(gen_batches(ch))
    blend(out0[0])
    res["ttff_ms"] = round((time.time() - t0) * 1000)
    res["ttff_whisper_feat_ms"] = round(t_feat * 1000)
    os.remove(first)

    # ---------- full clip throughput ----------
    t0 = time.time()
    chunks = feats(args.audio)
    sync(dev)
    t_feat = time.time() - t0
    n = len(chunks)
    res["frames"] = n
    res["audio_s"] = round(len(data) / sr, 2)
    res["whisper_feat_full_ms"] = round(t_feat * 1000)

    # GPU only (no blend)
    t0 = time.time()
    faces = []
    for out in gen_batches(chunks):
        faces.extend(out)
    sync(dev)
    t_gpu = time.time() - t0
    res["gpu_fps_unet_vae"] = round(n / t_gpu, 1)

    # pipelined: GPU producer, CPU blend consumer thread (as MuseTalk realtime does)
    q, frames_out = queue.Queue(maxsize=64), []

    def consumer():
        while True:
            f = q.get()
            if f is None:
                return
            frames_out.append(blend(f))

    th = threading.Thread(target=consumer)
    t0 = time.time()
    th.start()
    for out in gen_batches(chunks):
        for f in out:
            q.put(f)
    q.put(None)
    th.join()
    t_e2e = time.time() - t0
    res["e2e_fps_with_blend"] = round(n / t_e2e, 1)
    res["realtime_factor"] = round((n / args.fps) / (t_e2e + t_feat), 2)
    t0 = time.time()
    for f in faces[:50]:
        blend(f)
    res["cpu_blend_ms_per_frame"] = round((time.time() - t0) * 1000 / min(50, len(faces)), 1)
    a, r = peak_mem(dev)
    res["peak_alloc_mib"], res["peak_reserved_mib"] = round(a), round(r)

    if args.out:
        hh, ww = frames_out[0].shape[:2]
        tmp = args.out + ".noaudio.mp4"
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (ww, hh))
        for f in frames_out:
            vw.write(f)
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-i", args.audio, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-shortest", args.out], check=True)
        os.remove(tmp)
        res["out"] = args.out
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
