from flask import Flask, render_template, request, flash
import cv2
import numpy as np
from scipy.fftpack import dct
from scipy.signal import welch
from scipy.stats import entropy
import gc
import librosa
# import moviepy as mp
import os
import tempfile
import shutil
from yt_dlp import YoutubeDL
from moviepy import VideoFileClip, AudioFileClip, ColorClip
import contextlib
import io
import re
from urllib.parse import urlparse


app = Flask(__name__)
app.secret_key = "supersecret"
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB 限制

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/watermark/", methods=["GET", "POST"])
def watermark():
    # os._exit(0)
    result = None
    result2 = None
    if request.method == "POST":
        selected = request.form.get("type_select")

        if selected == "xxx":
            uploaded_file = request.files.get("file_input")

            if uploaded_file and uploaded_file.filename.endswith(".mp4"):
                # ✅ 建立一個暫存檔案（副檔名 .mp4）
                # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                #     uploaded_file.save(tmp.name)  # 儲存上傳檔案
                #     tmp_path = tmp.name           # 儲存路徑
                
                # ✅ 第一步：先建立一個臨時檔案名稱（但不在 with 裡儲存）
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_path = tmp_file.name
                tmp_file.close()  # ✅ 關閉檔案 → 不佔用（Windows 必須）

                # ✅ 第二步：將上傳檔案儲存到這個路徑
                uploaded_file.save(tmp_path)

                # ✅ 取得大小
                file_size_mb = round(os.path.getsize(tmp_path) / 1024 / 1024, 2)
                result = f"已上傳：{uploaded_file.filename}，大小約 {file_size_mb} MB"

                # os._exit(0)

                try:
                    result2 = analyze_video_for_watermarks(tmp_path)
                finally:
                    # 嘗試分析後再刪除，確保檔案釋放掉
                    try:
                        print(tmp_path)
                        os.remove(tmp_path)
                    except PermissionError:
                        print("⚠️ 無法刪除檔案，可能仍被佔用")
                        # flash("Error 1")
                        # flash(tmp_path)

            else:
                flash("請上傳 .mp4 格式的檔案")

        elif selected == "yyy":
            # text = request.form.get("text_input")
            # result = f"你輸入了文字：{text}"

            link = request.form.get("text_input")
            result = f"youtube影片或Shorts網址：{link}"
            result2 = analyze_video_for_watermarks(link)
            # try:
            #     result2 = analyze_video_for_watermarks(link)
            # finally:
            #     # 嘗試分析後再刪除，確保檔案釋放掉
            #     try:
            #         print(link)
            #         os.remove(link)
            #     except PermissionError:
            #         print("⚠️ 無法刪除檔案，可能仍被佔用")
            #         # flash("Error 1")
            #         # flash(link)


    # return render_template("xxx.html", result=result)
    return render_template("watermark.html", result=result, result2=result2)

def is_youtube_video_or_shorts(url):
    pattern = r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/(watch\?v=|shorts/)?[a-zA-Z0-9_-]{11}"
    return re.match(pattern, url) is not None

def check_file_size(file_path, max_size_mb=2000):
    if not os.path.exists(file_path):
        print(f"❌ 檔案不存在2：{file_path}")
        flash(f"檔案不存在2：{file_path}")
        return False

    file_size = os.path.getsize(file_path)  # 單位：位元組 (bytes)
    file_size_mb = file_size / (1024 * 1024)

    print(f"📦 檔案大小：{file_size_mb:.2f} MB")
    # flash(f"檔案大小：{file_size_mb:.2f} MB")

    if file_size_mb > max_size_mb:
        print(f"⚠️ 超過 {max_size_mb}MB 限制")
        flash(f"檔案大小：{file_size_mb:.2f} MB，超過 {max_size_mb}MB 限制")
        return False
    else:
        # print(f"✅ 檔案大小在允許範圍內")
        return True

# | 項目                                     | 優化方式                      |
# | -------------------------------------- | ------------------------- |
# | `gray[i:i+block_size, j:j+block_size]` | 改為視圖 (View)，避免複製          |
# | `magnitude /= magnitude.sum()`         | 改為 `np.sum()` 儲存變數，減少中間陣列 |
# | Entropy 輸出陣列                           | 預分配為 `float32` 降低記憶體壓力    |
# | 所有臨時變數結束後用 `del` 清除                    | 明確釋放 NumPy 資源             |
# | OpenCV Sobel 輸出避免高階浮點精度                | 使用 `float32` 而非 `float64` |

def block_entropy_dct(block):
    block = np.float32(block) / 255.0
    dct_block = cv2.dct(block)
    dct_block[0, 0] = 0
    magnitude = np.abs(dct_block)
    magnitude /= magnitude.sum() + 1e-8
    return entropy(magnitude.flatten())

def dct_entropy_map_single_image(image, block_size=16):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    entropy_map = np.zeros((h // block_size, w // block_size), dtype=np.float32)

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            e = block_entropy_dct(block)
            entropy_map[i // block_size, j // block_size] = e

    return entropy_map, gray

def estimate_complexity(gray_image):
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.mean(edge_magnitude)

def resize_frame_to_480p(frame):
    height, width = frame.shape[:2]
    new_height = 480
    new_width = int(width * (480 / height))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

def detect_watermark_in_frame(frame):
    frame = resize_frame_to_480p(frame)
    entropy_map, gray = dct_entropy_map_single_image(frame)
    flat = entropy_map.flatten()
    mean_val = np.mean(flat)
    q95_val = np.percentile(flat, 95)
    complexity = estimate_complexity(gray)

    # 🧯 防止 NaN 中斷分析
    if not np.isfinite(mean_val) or not np.isfinite(q95_val):
        print("❌ 熵統計無效，跳過此幀")
        return False

    threshold_mean = 0.9 + complexity * 0.05
    threshold_q95 = 1.4 + complexity * 0.1

    print(f"📊 熱統計: 平均={mean_val:.4f}, q95={q95_val:.4f}, 複雜度={complexity:.4f}")
    suspicious = (mean_val > threshold_mean) or (q95_val > threshold_q95)

    # 主動釋放記憶體
    del entropy_map, gray, flat
    gc.collect()
    return suspicious

def detect_watermark_in_video_frames(video_path, sample_rate=1800, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 無法打開影片：{video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    suspicious_count = 0
    analyzed_frames = 0

    print(f"🎨 影格分析開始：總影格 {total_frames}, 每 {sample_rate} 幀取樣一次")

    for i in range(0, total_frames, sample_rate):
        if analyzed_frames >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        analyzed_frames += 1
        if detect_watermark_in_frame(frame):
            suspicious_count += 1
            print(f"⚠️ 第 {i} 幀為可疑影格")
        else:
            print(f"✅ 第 {i} 幀正常")
        del frame
        gc.collect()

    cap.release()
    ratio = suspicious_count / analyzed_frames if analyzed_frames > 0 else 0
    print(f"📊 可疑影格比例：{ratio:.2%}")
    return ratio


# 🔹 音訊浮水印切片分析（每 2 秒）
def detect_audio_watermark_signal(filepath, segment_duration=2.0, threshold_db=-45):
    print(f"\n🎧 開始音訊分析：{filepath}")
    y, sr = librosa.load(filepath, sr=None)
    segment_length = int(segment_duration * sr)
    total_segments = len(y) // segment_length

    suspicious_segments = 0

    for i in range(total_segments):
        start = i * segment_length
        end = start + segment_length
        segment = y[start:end]
        f, Pxx = welch(segment, sr, nperseg=1024)
        Pxx_db = 10 * np.log10(Pxx + 1e-10)
        if np.any(Pxx_db > threshold_db):
            suspicious_segments += 1

    ratio = suspicious_segments / total_segments if total_segments > 0 else 0
    print(f"📊 可疑音訊片段比例：{ratio:.2%}")
    # return ratio > 0.3
    return ratio

# 🔹 YouTube 音訊下載

def download_audio_from_url(url, output_format='wav', max_filesize_mb=2000):
    output_folder = tempfile.mkdtemp()

    with YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get('title', 'unknown_title')
        expected_filename = f"{title}.{output_format}"
        expected_path = os.path.join(output_folder, expected_filename)

        if os.path.exists(expected_path):
            print(f"⏩ 檔案已存在：{expected_path}，跳過下載")
            flash(f"檔案已存在：{expected_path}，跳過下載")
            return expected_path, output_folder  # 已存在就直接回傳路徑

        filesize = info.get('filesize') or info.get('filesize_approx') or 0
        filesize_mb = filesize / (1024 * 1024)
        print('📏 預估檔案大小：', f"{filesize_mb:.2f} MB")
        # flash('預估檔案大小：', f"{filesize_mb:.2f} MB")
        if filesize_mb > max_filesize_mb:
            print(f"⚠️ 檔案過大：{filesize_mb:.2f} MB，超過 {max_filesize_mb} MB，已跳過下載")
            flash(f"檔案過大：{filesize_mb:.2f} MB，超過 {max_filesize_mb} MB，已跳過下載")
            return None, None

    output_template = os.path.join(output_folder, '%(title)s.%(ext)s')
    downloaded_file_path = None

    def post_hook(d):
        nonlocal downloaded_file_path
        if d['status'] == 'finished':
            downloaded_file_path = d['info_dict']['filepath']
            print(f"✅ 轉檔完成: {downloaded_file_path}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': output_format,
            'preferredquality': '192',
        }],
        'postprocessor_hooks': [post_hook],
        'quiet': True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            print(f"🌐 下載音訊中：{url}")
            ydl.download([url])
    except Exception as e:
        print(f"❌ 下載失敗，可能是直播或其他例外情況")
        flash(f"下載失敗，可能是直播或其他例外情況")
        return None, None

    return downloaded_file_path, output_folder

def classify_input_source(input_source):
    if not isinstance(input_source, str):
        return "invalid_type"

    # 預處理：去掉前後空白
    input_source = input_source.strip()

    # 第一種：完整 http(s) 網址（包含 YouTube）
    parsed = urlparse(input_source)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return "url"

    # 第二種：經驗法則判斷可能是網址但少了 http 前綴
    # if input_source.startswith("www.") or "youtube.com" in input_source or "youtu.be" in input_source:
    #     return "url"
    # 第二種：經驗法則判斷可能是網址但少了 http 前綴
    if input_source.startswith("www.") or "." in input_source.split("/")[0]:
        if os.path.isfile(input_source):
            return "file"
        else:
            return "url"

    # 第三種：本機已存在的檔案路徑（含相對路徑）
    if os.path.isfile(input_source):
        return "file"

    # 第四種：像是路徑格式，但找不到檔案
    # if "/" in input_source or "\\" in input_source or "." in os.path.basename(input_source):
    #     return "invalid_path"

    # 其他情況：不屬於網址也不是檔案
    return "unknown"

# 🔹 整體分析整合

def analyze_video_for_watermarks(input_source):
    input_source_result = classify_input_source(input_source)
    if input_source_result == 'url':
        is_url = True
    elif input_source_result == 'file':
        is_url = False
    else:
        print("❌ 請輸入正確youtube影片或Shorts網址")
        flash("請輸入正確youtube影片或Shorts網址")
        return

    # is_url = input_source.startswith("http")
    # if isinstance(input_source, str):
    #     is_url = input_source.startswith("http")
    # else:
    #     is_url = False  # 或根據你的邏輯處理非網址來源
    temp_audio = None
    temp_video = None
    temp_dir = None
    messages = []

# try:
    print(f"\n==========================")
    print(f"🧪 正在分析來源：{input_source}")
    print(f"==========================\n")

    if is_url:
        if not is_youtube_video_or_shorts(input_source):
            print("❌ 連結不是youtube影片或shorts網址，無法分析")
            flash("連結不是youtube影片或shorts網址，無法分析")
            return
        temp_audio, temp_dir = download_audio_from_url(input_source)
        if not temp_audio:
            print("❌ 無法分析（音訊下載失敗或檔案過大）")
            flash("無法分析（音訊下載失敗或檔案過大）")
            return

        # 🔧 使用黑畫面影片合成：便於影格分析
        temp_video = tempfile.mktemp(suffix=".mp4")
        audio_clip = AudioFileClip(temp_audio)
        duration = audio_clip.duration
        black_clip = ColorClip(size=(640, 480), color=(0, 0, 0), duration=duration)
        video_clip = black_clip.with_audio(audio_clip)
        video_clip.write_videofile(temp_video, fps=1, codec='libx264', audio_codec='aac', logger=None)
        video_ratio = detect_watermark_in_video_frames(temp_video)
        # suspicious_audio = detect_audio_watermark_signal(temp_audio)
        audio_ratio = detect_audio_watermark_signal(temp_audio)
    else:
        check_file_size_result = check_file_size(input_source)
        if not check_file_size_result: return
        temp_video = input_source
        video_ratio = detect_watermark_in_video_frames(temp_video)
        with contextlib.redirect_stdout(io.StringIO()):
            video = VideoFileClip(temp_video)
        temp_audio = tempfile.mktemp(suffix=".wav")
        video.audio.write_audiofile(temp_audio, logger=None)
        # suspicious_audio = detect_audio_watermark_signal(temp_audio)
        audio_ratio = detect_audio_watermark_signal(temp_audio)

    video_result = (video_ratio > 0.1)
    audio_result = (audio_ratio > 0.3)

    print("\n🔚 結果彙總：")
    messages.append(f"結果彙總：")
    if video_result:
    # if (video_result > 0.1):
        print(f"🎞️ 影片中影格偵測到可能浮水印，可疑影格比例：{video_ratio:.2%}")
        messages.append(f"影片中影格偵測到可能浮水印，可疑影格比例：{video_ratio:.2%}")
    else:
        print("✅ 影片中影格偵未偵測到明顯浮水印訊號 OK")
        messages.append("影片中影格偵未偵測到明顯浮水印訊號 OK")

    if audio_result:
        print(f"🎧 音訊中發現可能的浮水印頻段，可疑音訊片段比例：{audio_ratio:.2%}")
        messages.append(f"音訊中發現可能的浮水印頻段，可疑音訊片段比例：{audio_ratio:.2%}")
    else:
        print("✅ 音訊中未發現可能的浮水印頻段 OK")
        messages.append("音訊中未發現可能的浮水印頻段 OK")

    # if not video_result and not audio_result:
    #     print("✅ 影片與音訊皆未偵測到明顯浮水印訊號 OK")
    #     messages.append("影片與音訊皆未偵測到明顯浮水印訊號 OK")

# finally:
    if temp_audio and os.path.exists(temp_audio):
        os.remove(temp_audio)
    if temp_video and os.path.exists(temp_video) and is_url:
        os.remove(temp_video)
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    return "\n".join(messages)


if __name__ == "__main__":
    app.run(debug=True)