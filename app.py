import streamlit as st
import librosa
import numpy as np
import os
import sys
sys.setrecursionlimit(10000)
import tempfile
from pathlib import Path
from pydub import AudioSegment
from scipy.signal import welch, butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════
#  安全删除文件（解决 Windows 文件句柄未释放问题）
# ══════════════════════════════════════════════════════════
def safe_remove(filepath, retries=5, delay=0.5):
    """
    安全删除临时文件，Windows 上 ffmpeg 子进程可能延迟释放文件句柄。
    失败时重试 retries 次，每次间隔 delay 秒。
    """
    import time
    for _ in range(retries):
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
            return True
        except (PermissionError, OSError):
            time.sleep(delay)
    return False  # 放弃删除，文件留在临时目录，系统会自动清理

# ══════════════════════════════════════════════════════════
#  ffmpeg（云端部署使用系统自带 ffmpeg，无需本地路径配置）
# ══════════════════════════════════════════════════════════
_FFMPEG_PATH = None  # 云端使用系统 PATH 中的 ffmpeg

# ── 页面配置 ──────────────────────────────────────────────
st.set_page_config(
    page_title="NVH 音频分析 - 频谱与Colormap",
    page_icon="🎵",
    layout="wide"
)

# 中文字体（打包后优先使用系统字体，失败则回退）
def _setup_chinese_font():
    candidates = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                  'PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['font.sans-serif'] = candidates
    plt.rcParams['axes.unicode_minus'] = False

_setup_chinese_font()

st.markdown("""
<style>
    .main-header {font-size:2.2rem;font-weight:bold;color:#1f77b4;text-align:center;margin-bottom:.3rem}
    .sub-header  {font-size:1.05rem;color:#666;text-align:center;margin-bottom:1.5rem}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  A 计权 (IEC 61672-1)
# ══════════════════════════════════════════════════════════
def a_weighting_db(frequencies):
    """返回各频率点的 A 计权修正量 (dB)"""
    f = np.maximum(frequencies, 1e-10)
    f1, f2, f3, f4 = 20.6, 107.7, 737.9, 12194.0
    ra = (f4**2 * f**4) / (
        (f**2 + f1**2) *
        np.sqrt((f**2 + f2**2) * (f**2 + f3**2)) *
        (f**2 + f4**2)
    )
    return 20 * np.log10(ra + 1e-12) + 2.00


# ══════════════════════════════════════════════════════════
#  分辨率换算工具
# ══════════════════════════════════════════════════════════
def freq_res_to_nfft(sr, freq_res_hz):
    """频率分辨率(Hz) → FFT窗长(点数，取最接近的2的幂)"""
    n = max(256, int(np.ceil(sr / max(freq_res_hz, 0.1))))
    return 2 ** int(np.round(np.log2(n)))


def time_res_to_hop(sr, time_res_ms):
    """时间分辨率(ms) → 帧移(点数)"""
    return max(64, int(round(sr * max(time_res_ms, 0.1) / 1000)))


def apply_filter(y, sr, filter_type, low_hz=None, high_hz=None):
    """
    零相位 Butterworth 滤波（4阶）。
    filter_type: 'all' / 'bandpass' / 'bandstop'
    low_hz, high_hz: 截止频率 (Hz)
    """
    if filter_type == 'all' or sr <= 0:
        return y
    nyq = sr / 2.0
    if filter_type == 'bandpass':
        if low_hz is None or high_hz is None or low_hz >= high_hz:
            return y
        low = max(low_hz, 1.0) / nyq
        high = min(high_hz, nyq * 0.99) / nyq
        if low >= high:
            return y
        b, a = butter(4, [low, high], btype='bandpass')
    elif filter_type == 'bandstop':
        if low_hz is None or high_hz is None or low_hz >= high_hz:
            return y
        low = max(low_hz, 1.0) / nyq
        high = min(high_hz, nyq * 0.99) / nyq
        if low >= high:
            return y
        b, a = butter(4, [low, high], btype='bandstop')
    else:
        return y
    # filtfilt 要求信号长度 > 3*max(len(a),len(b))，否则用 lfilter
    padlen = 3 * max(len(a), len(b))
    if len(y) <= padlen:
        from scipy.signal import lfilter
        return lfilter(b, a, y)
    return filtfilt(b, a, y)


# ══════════════════════════════════════════════════════════
#  ffmpeg 可用性检测
# ══════════════════════════════════════════════════════════
def _check_ffmpeg():
    from shutil import which
    return which('ffmpeg') is not None


# ══════════════════════════════════════════════════════════
#  文件格式转换 → 临时 WAV
# ══════════════════════════════════════════════════════════
def convert_to_wav(uploaded_file, file_extension):
    try:
        audio_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_wav.close()
        wav_path = tmp_wav.name

        if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            st.info("🎬 检测到视频文件，正在提取音频轨道...")
            tmp_vid = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
            tmp_vid.write(audio_bytes)
            tmp_vid.close()
            try:
                AudioSegment.from_file(tmp_vid.name).export(wav_path, format="wav")
            finally:
                safe_remove(tmp_vid.name)
        elif file_extension in ['.mp3', '.aac', '.ogg', '.flac', '.m4a', '.wma']:
            AudioSegment.from_file(io.BytesIO(audio_bytes),
                                   format=file_extension[1:]).export(wav_path, format="wav")
        else:
            with open(wav_path, 'wb') as f:
                f.write(audio_bytes)
        return wav_path
    except Exception as e:
        st.error(f"❌ 转换失败: {e}")
        if not _check_ffmpeg():
            st.warning("⚠️ 未检测到 ffmpeg，mp3/mp4 等格式无法处理。请安装 ffmpeg 并加入系统 PATH。")
        return None


# ══════════════════════════════════════════════════════════
#  缓存加载
# ══════════════════════════════════════════════════════════
def load_audio_cached(audio_bytes, file_extension):
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    try:
        if file_extension == '.wav':
            # WAV 直接写入
            with open(tmp.name, 'wb') as f:
                f.write(audio_bytes)
        else:
            # 所有非 WAV 格式（mp3/m4a/aac/ogg/flac/mp4/mov 等）：
            # 先写临时文件再用 pydub 转换，避免 ffmpeg 无法从管道读取 m4a/mp4 等需要随机访问的格式
            tmp_src = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
            tmp_src.write(audio_bytes)
            tmp_src.close()
            try:
                AudioSegment.from_file(tmp_src.name).export(tmp.name, format="wav")
            finally:
                safe_remove(tmp_src.name)
        # 优先用 soundfile 读取（快且稳定）
        y = None
        sr = None
        try:
            import soundfile as sf
            y, sr = sf.read(tmp.name, dtype='float32')
            if y.ndim > 1:
                y = y.mean(axis=1)  # 多声道转单声道
        except Exception:
            y = None

        # soundfile 失败或返回空数组时，回退到 librosa（兼容 ADPCM/A-law 等特殊编码）
        if y is None or len(y) == 0:
            y, sr = librosa.load(tmp.name, sr=None, mono=True)

        duration = len(y) / sr
        return y, sr, duration
    finally:
        safe_remove(tmp.name)


# ══════════════════════════════════════════════════════════
#  频谱图 (Welch 平均功率谱)
# ══════════════════════════════════════════════════════════
def plot_spectrum(y, sr, nperseg=4096, overlap=0.5,
                  freq_scale='linear', a_weighted=False,
                  fmax=20000, db_range=60,
                  t_start=None, t_end=None):
    # 时间段裁剪
    if t_start is not None or t_end is not None:
        s = int((t_start or 0) * sr)
        e = int((t_end or len(y) / sr) * sr)
        y = y[max(0, s):min(len(y), e)]
    if len(y) < nperseg:
        nperseg = max(256, len(y) // 2)
    f, Pxx = welch(y, fs=sr, nperseg=nperseg,
                   noverlap=int(nperseg * overlap), window='hann')
    Pxx_db = 10 * np.log10(Pxx + 1e-12)
    Pxx_db -= Pxx_db.max()

    if a_weighted:
        Pxx_db += a_weighting_db(f)
        suffix, ylabel = ' (A计权)', '相对声压级 dB(A)'
    else:
        suffix, ylabel = '', '相对量级 dB'

    # 只画到 fmax，与 Colormap 横坐标对齐
    fmax = min(fmax, sr / 2)
    mask = f <= fmax
    f_plot = f[mask]
    Pxx_plot = Pxx_db[mask]

    time_tag = f' [{t_start:.1f}s - {t_end:.1f}s]' if (t_start or t_end) else ''
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(f_plot, Pxx_plot, color='#1f77b4', linewidth=1.2)
    ax.fill_between(f_plot, Pxx_plot, Pxx_plot.max() - db_range, alpha=0.2, color='#1f77b4')
    ax.set_xlabel('频率 (Hz)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'频谱图 (Welch 平均功率谱){suffix}{time_tag}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(Pxx_plot.max() - db_range, Pxx_plot.max() + 2)

    if freq_scale == 'log':
        ax.set_xscale('log')
        ax.set_xlim(20, fmax)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    else:
        ax.set_xlim(0, fmax)
        ax.axvspan(20, 250, alpha=0.06, color='red')
        ax.axvspan(250, 4000, alpha=0.06, color='green')
        ax.axvspan(4000, min(fmax, 20000), alpha=0.06, color='blue')
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════
#  瞬时频谱（指定时刻的短时 FFT）
# ══════════════════════════════════════════════════════════
def plot_instant_spectrum(y, sr, t_center, window=0.5,
                          n_fft=2048, a_weighted=False,
                          freq_scale='linear', fmax=20000, db_range=60):
    """
    显示指定时刻前后 window 秒的短时 FFT 频谱。
    t_center: 中心时刻（秒）
    window: 窗口长度（秒），默认 0.5s
    """
    half = int(window * sr / 2)
    center = int(t_center * sr)
    s = max(0, center - half)
    e = min(len(y), center + half)
    segment = y[s:e]

    if len(segment) < 32:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.text(0.5, 0.5, '所选时刻数据不足', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
        return fig

    # 加窗 FFT
    n_fft = min(n_fft, len(segment))
    window_arr = np.hanning(len(segment))
    segment_win = segment * window_arr
    fft_vals = np.fft.rfft(segment_win, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    magnitude = np.abs(fft_vals)
    mag_db = 20 * np.log10(magnitude + 1e-10)
    mag_db -= mag_db.max()

    if a_weighted:
        mag_db += a_weighting_db(freqs)
        suffix, ylabel = ' (A计权)', '相对声压级 dB(A)'
    else:
        suffix, ylabel = '', '相对量级 dB'

    # 只画到 fmax，与 Colormap/频谱图横坐标对齐
    fmax = min(fmax, sr / 2)
    mask = freqs <= fmax
    f_plot = freqs[mask]
    mag_plot = mag_db[mask]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(f_plot, mag_plot, color='#d62728', linewidth=1.2)
    ax.fill_between(f_plot, mag_plot, mag_plot.max() - db_range, alpha=0.2, color='#d62728')
    ax.set_xlabel('频率 (Hz)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'瞬时频谱 @ {t_center:.2f}s（窗口 {window}s）{suffix}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(mag_plot.max() - db_range, mag_plot.max() + 2)
    if freq_scale == 'log':
        ax.set_xscale('log')
        ax.set_xlim(20, fmax)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    else:
        ax.set_xlim(0, fmax)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════
#  Colormap (Test.Lab 风格: X=频率, Y=时间, A计权)
# ══════════════════════════════════════════════════════════
def plot_colormap(y, sr, n_fft=2048, hop_length=512,
                  freq_max=20000, a_weighted=True,
                  freq_scale='linear', db_range=60,
                  t_start=None, t_end=None):
    # 时间段裁剪
    time_offset = 0.0
    if t_start is not None or t_end is not None:
        s = int((t_start or 0) * sr)
        e = int((t_end or len(y) / sr) * sr)
        y = y[max(0, s):min(len(y), e)]
        time_offset = t_start or 0.0

    max_samples = int(sr * 600)
    if len(y) > max_samples:
        y = y[:max_samples]
        st.warning("⏱️ 所选时段超过 10 分钟，仅分析前 10 分钟")

    if len(y) < n_fft:
        n_fft = max(256, len(y) // 2)

    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann'))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length) + time_offset

    DB = librosa.amplitude_to_db(D, ref=np.max)

    if a_weighted:
        DB += a_weighting_db(freqs)[:, np.newaxis]
        cbar_label = '相对声压级 dB(A)'
        title = 'Colormap 时频图 (A计权 · Test.Lab风格)'
    else:
        cbar_label = '相对量级 dB'
        title = 'Colormap 时频图 (线性 · Test.Lab风格)'

    vmax = DB.max()
    vmin = vmax - db_range
    DB = np.clip(DB, vmin, vmax)

    freq_max = min(freq_max, sr / 2)
    mask = freqs <= freq_max
    f_plot, D_plot = freqs[mask], DB[mask, :]

    fig, ax = plt.subplots(figsize=(14, 8))

    if freq_scale == 'log':
        valid = f_plot > 0
        f_log, D_log = f_plot[valid], D_plot[valid, :]
        X, Y = np.meshgrid(f_log, times)
        im = ax.pcolormesh(X.T, Y.T, D_log, cmap='jet',
                           vmin=vmin, vmax=vmax, shading='auto')
        ax.set_xscale('log')
        ax.set_xlim(20, freq_max)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    else:
        X, Y = np.meshgrid(f_plot, times)
        im = ax.pcolormesh(X.T, Y.T, D_plot, cmap='jet',
                           vmin=vmin, vmax=vmax, shading='auto')
        ax.set_xlim(0, freq_max)

    ax.set_xlabel('频率 (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('时间 (s)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', color='white')
    ax.locator_params(axis='y', nbins=12)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════
#  主界面
# ══════════════════════════════════════════════════════════
def main():
    st.markdown('<div class="main-header">🎵 NVH 音频分析工具</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Welch 频谱图 + Test.Lab 风格 Colormap（A计权）| 支持 WAV/MP3/MP4/M4A/FLAC/OGG</div>',
                unsafe_allow_html=True)

    # ffmpeg 状态提示
    if not _check_ffmpeg():
        st.warning("⚠️ 未检测到 ffmpeg。WAV 格式可正常分析；MP3/MP4/M4A 等格式需 ffmpeg 支持。")

    with st.sidebar:
        st.header("📁 文件上传")
        uploaded_file = st.file_uploader(
            "拖拽或点击上传",
            type=['wav', 'mp3', 'mp4', 'm4a', 'ogg', 'flac',
                  'aac', 'avi', 'mov', 'mkv', 'webm', 'wma']
        )

        st.markdown("---")
        st.header("⚙️ 分析参数")

        freq_res = st.selectbox(
            "频率分辨率", [1, 2, 5, 10, 20], index=3,
            format_func=lambda x: f"{x} Hz"
        )
        time_res = st.selectbox(
            "时间分辨率", [5, 10, 20], index=1,
            format_func=lambda x: f"{x} ms"
        )
        use_a_weight = st.checkbox("A 计权", value=True)
        cm_scale = st.radio("频率轴", ["线性", "对数"], horizontal=True)
        cm_fmax = st.number_input("最高频率 (Hz)", 100, 96000, 20000, 1000)
        cm_dbr = st.slider("动态范围 (dB)", 30, 100, 60, 5)

    if uploaded_file is None:
        st.info("👈 请在左侧上传音频或视频文件")
        return

    file_name = uploaded_file.name
    ext = Path(file_name).suffix.lower()
    st.success(f"✅ 已上传：**{file_name}**")

    audio_bytes = uploaded_file.getvalue()
    try:
        y, sr, duration = load_audio_cached(audio_bytes, ext)
        if y is None or len(y) == 0:
            st.error("❌ 文件解析失败：未读取到有效音频数据。该文件可能使用了不支持的编码格式（如手机录音机的特殊编码），建议先用格式工厂等工具转为标准 WAV/MP3 后再上传。")
            return
    except Exception as e:
        import traceback
        err_msg = str(e).lower()
        if isinstance(e, PermissionError) or '拒绝访问' in err_msg or 'permission' in err_msg:
            st.error("❌ ffmpeg 执行被拒绝（PermissionError）。可能原因：")
            st.markdown("""
            1. **杀毒软件/Windows Defender 拦截** — 请将程序目录加入白名单，或暂时关闭杀毒软件测试
            2. **程序放在受保护目录** — 建议移到 D 盘根目录（如 `D:\\NVHAudioAnalyzer\\`），不要放在 `C:\\Software\\` 等系统目录
            3. **权限不足** — 右键 exe → **以管理员身份运行**
            4. **ffmpeg.exe 文件损坏** — 检查文件大小是否为 80-100MB，过小请重新复制
            """)
        elif 'ffmpeg' in err_msg or 'decode' in err_msg or 'couldn' in err_msg:
            st.error("❌ 音频/视频解码失败。请确认已安装 ffmpeg 并加入系统 PATH，或将 ffmpeg.exe 打包进 bin 目录。")
        else:
            st.error(f"❌ 文件解析失败: {e}")
        with st.expander("查看详细错误信息"):
            st.code(traceback.format_exc())
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("采样率", f"{sr:,} Hz")
    c2.metric("时长", f"{duration:.2f} s")
    c3.metric("采样点数", f"{len(y):,}")
    c4.metric("通道", "单声道")

    # ── 滤波器（仅影响回放，不影响分析） ──
    st.markdown("---")
    st.markdown("### 🔊 回放滤波")
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    with col_f1:
        filter_type = st.selectbox(
            "滤波器类型",
            ["全通（原始）", "带通", "带阻"],
            index=0
        )
    with col_f2:
        low_cut = st.slider("低截止 (Hz)", 20, int(sr / 2) - 100, 200, 10)
    with col_f3:
        high_cut = st.slider("高截止 (Hz)", 50, int(sr / 2) - 50, 2000, 50)

    ft_map = {"全通（原始）": "all", "带通": "bandpass", "带阻": "bandstop"}
    y_play = apply_filter(y, sr, ft_map[filter_type], low_cut, high_cut)

    # 播放
    tmp_play = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_play.close()
    try:
        import soundfile as sf
        sf.write(tmp_play.name, y_play, sr)
        st.audio(tmp_play.name, format='audio/wav')
    finally:
        safe_remove(tmp_play.name)

    # ── 分析范围：时间段选择 ──
    st.markdown("---")
    st.markdown("### 🎯 分析范围")
    t_min = 0.0
    t_max = float(duration)
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        t_range = st.slider(
            "选择分析时间段（秒）",
            min_value=0.0, max_value=t_max,
            value=(0.0, t_max), step=0.1,
            format="%.1f"
        )
    with col_sel2:
        st.markdown("**选中时长**")
        st.markdown(f"### {t_range[1] - t_range[0]:.1f} s")
    t_start, t_end = t_range

    # ── 瞬时频谱（跟随时刻） ──
    st.markdown("---")
    st.markdown("### ⏱️ 瞬时频谱（指定时刻）")
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        t_instant = st.slider(
            "时刻定位（秒）— 播放时可拖动查看对应时刻频谱",
            min_value=0.0, max_value=t_max,
            value=min(t_max / 2, 5.0), step=0.05,
            format="%.2f"
        )
    with col_t2:
        inst_window = st.selectbox("瞬时窗口", [0.1, 0.2, 0.5, 1.0], index=2)
    st.pyplot(plot_instant_spectrum(
        y, sr, t_instant, window=inst_window,
        a_weighted=use_a_weight,
        freq_scale='log' if cm_scale == '对数' else 'linear',
        fmax=cm_fmax, db_range=cm_dbr
    ))

    # ── 全局频谱图（基于选中时间段） ──
    st.markdown("---")
    st.markdown("### 📊 频谱图 (Welch 平均功率谱)")
    spec_nperseg = freq_res_to_nfft(sr, freq_res)
    st.pyplot(plot_spectrum(y, sr, spec_nperseg, 0.5,
                            'log' if cm_scale == '对数' else 'linear', use_a_weight,
                            fmax=cm_fmax, db_range=cm_dbr,
                            t_start=t_start, t_end=t_end))

    # ── Colormap（基于选中时间段） ──
    st.markdown("---")
    st.markdown("### 🌈 Colormap 时频图 (Test.Lab 风格)")
    cm_nfft = freq_res_to_nfft(sr, freq_res)
    cm_hop = time_res_to_hop(sr, time_res)
    st.pyplot(plot_colormap(y, sr, cm_nfft, cm_hop, cm_fmax,
                            use_a_weight, 'log' if cm_scale == '对数' else 'linear', cm_dbr,
                            t_start=t_start, t_end=t_end))

    st.info("💡 右键图表可保存 PNG；拖动上方滑块可切换时刻和分析范围")


if __name__ == "__main__":
    main()
