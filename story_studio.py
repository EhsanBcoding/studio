import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json, re, os, sys, shutil, subprocess, time, math
    import importlib.util
    from pathlib import Path

    WORK = Path("story_out")
    WORK.mkdir(exist_ok=True)
    return (
        Path,
        WORK,
        importlib,
        json,
        math,
        mo,
        re,
        shutil,
        subprocess,
        sys,
        time,
    )


@app.cell
def _(mo):
    mo.md("""
    # 🎬 Story Studio — story → scenes → images → narration → video

    **How to use:** wait for the setup cells to finish (first run builds tools and downloads models),
    then go step by step: **1️⃣ Analyze → (edit the character bible) → 2️⃣ Images → 3️⃣ Narration → 4️⃣ Video**.
    Everything runs locally on this notebook's GPU. Outputs are saved in `story_out/`.
    """)
    return


@app.cell
def _(importlib, shutil, subprocess, sys):
    # --- Install missing Python packages ---
    _need = {
        "kokoro": "kokoro>=0.9.4",
        "soundfile": "soundfile",
        "imageio_ffmpeg": "imageio-ffmpeg",
        "requests": "requests",
        "huggingface_hub": "huggingface_hub",
        "numpy": "numpy",
    }
    _missing = [p for m, p in _need.items() if importlib.util.find_spec(m) is None]
    if _missing:
        _cmd = (
            ["uv", "pip", "install", "--python", sys.executable]
            if shutil.which("uv")
            else [sys.executable, "-m", "pip", "install"]
        ) + _missing
        subprocess.run(_cmd, check=True)
    deps_ok = True
    return (deps_ok,)


@app.cell
def _(Path, deps_ok, mo, subprocess):
    # --- Build stable-diffusion.cpp (images) and llama.cpp (story director) with CUDA ---
    _ = deps_ok

    def find_bin(folder, names):
        for n in names:
            p = Path(folder) / "build" / "bin" / n
            if p.exists():
                return str(p)
        return None

    with mo.status.spinner("Building tools (first time only, ~5–10 min)..."):
        if not find_bin("stable-diffusion.cpp", ["sd-cli", "sd"]):
            if not Path("stable-diffusion.cpp").exists():
                subprocess.run(
                    ["git", "clone", "--recursive", "https://github.com/leejet/stable-diffusion.cpp"],
                    check=True,
                )
            subprocess.run(
                "cd stable-diffusion.cpp && cmake -B build -DSD_CUDA=ON && cmake --build build -j --config Release",
                shell=True,
                check=True,
            )
        if not find_bin("llama.cpp", ["llama-server"]):
            if not Path("llama.cpp").exists():
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp"],
                    check=True,
                )
            subprocess.run(
                "cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j --config Release --target llama-server",
                shell=True,
                check=True,
            )

    SD_BIN = find_bin("stable-diffusion.cpp", ["sd-cli", "sd"])
    LLAMA_BIN = find_bin("llama.cpp", ["llama-server"])
    mo.md(f"✅ Tools ready — sd: `{SD_BIN}` · llama-server: `{LLAMA_BIN}`")
    return LLAMA_BIN, SD_BIN


@app.cell
def _(deps_ok, mo):
    # --- Download models (cached after the first run) ---
    _ = deps_ok
    from huggingface_hub import hf_hub_download, list_repo_files

    _DIT_REPO = "AtomicChat/Qwen-Image-2.1-Turbo-GGUF"
    _files = [f for f in list_repo_files(_DIT_REPO) if f.endswith(".gguf") and "mmproj" not in f]
    _pick = None
    for _key in ["BF16", "Q8_0", "AD-Q6_K", "AD-Q4_K"]:
        _match = [f for f in _files if _key in f]
        if _match:
            _pick = _match[0]
            break
    _pick = _pick or _files[0]

    with mo.status.spinner(f"Downloading models (denoiser: {_pick})..."):
        DIT = hf_hub_download(_DIT_REPO, _pick)
        LLM = hf_hub_download(
            "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF",
            "Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf",
        )
        VAE = hf_hub_download("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors")
    mo.md(f"✅ Models ready — denoiser: `{_pick}`")
    return DIT, LLM, VAE


@app.cell
def _(LLAMA_BIN, LLM, json, mo, subprocess, time):
    # --- Start the "story director" LLM (Qwen3-VL-8B, same file as the image text encoder) ---
    import requests

    LLM_URL = "http://127.0.0.1:8081"

    def _healthy():
        try:
            return requests.get(LLM_URL + "/health", timeout=2).status_code == 200
        except Exception:
            return False

    if not _healthy():
        subprocess.Popen(
            [LLAMA_BIN, "-m", LLM, "-ngl", "99", "-c", "16384", "--host", "127.0.0.1", "--port", "8081"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with mo.status.spinner("Starting the story-director LLM..."):
            for _ in range(180):
                if _healthy():
                    break
                time.sleep(2)
    if not _healthy():
        raise RuntimeError("llama-server did not start. Check that llama.cpp built correctly.")

    def ask_json(system, user, max_tokens=1500, temperature=0.6):
        r = requests.post(
            LLM_URL + "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=900,
        )
        r.raise_for_status()
        t = r.json()["choices"][0]["message"]["content"]
        t = t[t.find("{") : t.rfind("}") + 1]
        return json.loads(t)

    mo.md("✅ Story director is online")
    return (ask_json,)


@app.cell
def _(mo):
    mo.md("""
    ## 1️⃣ Story
    """)
    return


@app.cell
def _(mo):
    story = mo.ui.text_area(
        label="📜 Story (English — this exact text is narrated)",
        full_width=True,
        rows=14,
    )
    style = mo.ui.text(
        value="cinematic dark illustration, moody volumetric lighting, muted desaturated colors, film grain, highly detailed",
        label="🎨 Visual style (applied to every scene)",
        full_width=True,
    )
    words_per_scene = mo.ui.slider(25, 120, value=55, step=5, label="Words per scene", show_value=True)
    analyze_btn = mo.ui.run_button(label="1️⃣ Analyze story")
    mo.vstack([story, style, mo.hstack([words_per_scene, analyze_btn], justify="start")])
    return analyze_btn, story, style, words_per_scene


@app.cell
def _(
    WORK,
    analyze_btn,
    ask_json,
    json,
    mo,
    re,
    story,
    style,
    words_per_scene,
):
    mo.stop(not analyze_btn.value, mo.md("_Paste a story and press **1️⃣ Analyze story**._"))
    mo.stop(not story.value.strip(), mo.md("⚠️ The story is empty."))

    # Clear outputs from a previous story
    for _old in list(WORK.glob("scene_*")) + [WORK / "final.mp4"]:
        if _old.exists():
            _old.unlink()

    def _chunk(text, target):
        out, cur = [], []
        for para in re.split(r"\n\s*\n", text.strip()):
            for s in re.split(r"(?<=[.!?…])\s+", para.strip()):
                if not s:
                    continue
                cur.append(s)
                if len(" ".join(cur).split()) >= target:
                    out.append(" ".join(cur))
                    cur = []
            if cur and len(" ".join(cur).split()) >= target * 0.6:
                out.append(" ".join(cur))
                cur = []
        if cur:
            if out and len(" ".join(cur).split()) < target * 0.4:
                out[-1] += " " + " ".join(cur)
            else:
                out.append(" ".join(cur))
        return out

    with mo.status.spinner("Reading the story and designing characters..."):
        bible = ask_json(
            "You are the art director of a narrated story video. Reply with JSON only.",
            "Story:\n"
            + story.value
            + "\n\nList every recurring character (people, creatures, or important animals). "
            "Give each a FIXED visual description of 25-40 words that an image model can repeat in every scene: "
            "apparent age, build, face, hair, clothing with colors, and one distinctive feature. "
            "No personality, no actions.\n"
            'Return: {"characters": [{"name": "...", "look": "..."}]}',
        )
    bible["style"] = style.value.strip()
    _names = [c["name"] for c in bible.get("characters", [])]

    scenes = []
    _prev = ""
    for _i, _txt in mo.status.progress_bar(
        list(enumerate(_chunk(story.value, words_per_scene.value))), title="Directing scenes"
    ):
        _r = ask_json(
            "You are a storyboard artist. Reply with JSON only.",
            f"Known characters: {', '.join(_names) or 'none'}\n\n"
            f"Previous scene: {_prev or '(start of story)'}\n\n"
            f"This scene's narration:\n{_txt}\n\n"
            "Describe ONE still image for this moment: setting, action, composition, camera angle, lighting, mood. "
            "Refer to characters by name only. Do NOT describe their appearance. 40-70 words, English.\n"
            'Return: {"characters": ["names present in the image"], "prompt": "..."}',
        )
        scenes.append(
            {
                "id": _i + 1,
                "narration": _txt,
                "characters": [n for n in _r.get("characters", []) if n in _names],
                "prompt": _r.get("prompt", _txt),
            }
        )
        _prev = _txt

    (WORK / "scenes.json").write_text(
        json.dumps({"bible": bible, "scenes": scenes}, indent=2, ensure_ascii=False)
    )
    mo.md(f"✅ **{len(scenes)} scenes**, **{len(_names)} characters**. Review the character bible below.")
    return bible, scenes


@app.cell
def _(bible, json, mo):
    bible_editor = mo.ui.code_editor(
        value=json.dumps(bible, indent=2, ensure_ascii=False),
        language="json",
        min_height=200,
    )
    mo.vstack(
        [
            mo.md(
                "### 🧬 Character bible\n"
                "These descriptions are pasted into **every** scene where the character appears. "
                "That is what keeps faces and outfits consistent. Edit them freely before generating images."
            ),
            bible_editor,
        ]
    )
    return (bible_editor,)


@app.cell
def _(bible_editor, json, mo, scenes):
    try:
        BIBLE = json.loads(bible_editor.value)
    except Exception as _e:
        mo.stop(True, mo.md(f"⚠️ The character bible is not valid JSON: `{_e}`"))

    def compose(scene, bible_data):
        looks = {c["name"]: c["look"] for c in bible_data.get("characters", [])}
        cast = "; ".join(f"{n}: {looks[n]}" for n in scene["characters"] if n in looks)
        parts = [scene["prompt"]]
        if cast:
            parts.append("Characters: " + cast)
        if bible_data.get("style"):
            parts.append("Style: " + bible_data["style"])
        return ". ".join(parts)

    mo.ui.table(
        [{"#": s["id"], "narration": s["narration"], "image prompt": compose(s, BIBLE)} for s in scenes],
        selection=None,
        page_size=8,
    )
    return BIBLE, compose


@app.cell
def _(mo):
    mo.md("""
    ## 2️⃣ Images
    """)
    return


@app.cell
def _(mo):
    seed = mo.ui.number(value=42, label="Seed (same for every scene = more consistent look)")
    size = mo.ui.dropdown(["1344x768", "1024x1024", "768x1344"], value="1344x768", label="Image size")
    images_btn = mo.ui.run_button(label="2️⃣ Generate all images")
    mo.hstack([seed, size, images_btn], justify="start")
    return images_btn, seed, size


@app.cell
def _(DIT, LLM, SD_BIN, VAE, WORK, mo, subprocess):
    SIGMAS = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0"

    def render(prompt, seed_value, size_value, out_path):
        w, h = size_value.split("x")
        try:
            subprocess.run(
                [
                    SD_BIN, "--diffusion-model", DIT, "--llm", LLM, "--vae", VAE,
                    "-p", prompt, "--steps", "8", "--cfg-scale", "1.0",
                    "--sampling-method", "euler", "--sigmas", SIGMAS,
                    "-W", w, "-H", h, "--seed", str(int(seed_value)),
                    "--diffusion-fa", "-o", str(out_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError("sd-cli failed:\n" + (e.stderr or e.stdout or "")[-3000:])

    def gallery(n_scenes):
        tiles = []
        for i in range(1, n_scenes + 1):
            p = WORK / f"scene_{i:02d}.png"
            if p.exists():
                tiles.append(mo.vstack([mo.image(str(p), width=300), mo.md(f"**Scene {i}**")]))
        rows = [mo.hstack(tiles[k : k + 3], justify="start") for k in range(0, len(tiles), 3)]
        return mo.vstack(rows) if rows else mo.md("_No images yet._")

    return gallery, render


@app.cell
def _(
    BIBLE,
    WORK,
    compose,
    gallery,
    images_btn,
    mo,
    render,
    scenes,
    seed,
    size,
):
    mo.stop(not images_btn.value, gallery(len(scenes)))
    for _s in mo.status.progress_bar(scenes, title="Painting scenes (~20–40 s each)"):
        render(compose(_s, BIBLE), seed.value, size.value, WORK / f"scene_{_s['id']:02d}.png")
    gallery(len(scenes))
    return


@app.cell
def _(mo, scenes):
    regen_id = mo.ui.number(start=1, stop=max(1, len(scenes)), value=1, label="Scene #")
    regen_seed = mo.ui.number(value=7, label="New seed")
    regen_extra = mo.ui.text(
        label="Extra direction (optional)",
        full_width=True,
        placeholder="e.g. closer shot, she is holding the lantern",
    )
    regen_btn = mo.ui.run_button(label="🔁 Regenerate this scene")
    mo.vstack(
        [
            mo.md("### Fix a single scene"),
            mo.hstack([regen_id, regen_seed, regen_btn], justify="start"),
            regen_extra,
        ]
    )
    return regen_btn, regen_extra, regen_id, regen_seed


@app.cell
def _(
    BIBLE,
    WORK,
    compose,
    mo,
    regen_btn,
    regen_extra,
    regen_id,
    regen_seed,
    render,
    scenes,
    size,
):
    mo.stop(not regen_btn.value)
    _s = dict(scenes[int(regen_id.value) - 1])
    if regen_extra.value.strip():
        _s["prompt"] = _s["prompt"] + ". " + regen_extra.value.strip()
    _out = WORK / f"scene_{_s['id']:02d}.png"
    with mo.status.spinner(f"Repainting scene {_s['id']}..."):
        render(compose(_s, BIBLE), regen_seed.value, size.value, _out)
    mo.image(str(_out), width=480)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3️⃣ Narration
    """)
    return


@app.cell
def _(mo):
    voice = mo.ui.dropdown(
        ["am_onyx", "am_michael", "am_fenrir", "am_eric", "am_liam", "am_adam", "am_echo", "am_puck"],
        value="am_onyx",
        label="🎙️ Voice (American male)",
    )
    speed = mo.ui.slider(0.8, 1.15, value=0.95, step=0.05, label="Speed", show_value=True)
    preview_btn = mo.ui.run_button(label="▶️ Preview voice")
    voice_btn = mo.ui.run_button(label="3️⃣ Narrate all scenes")
    mo.hstack([voice, speed, preview_btn, voice_btn], justify="start")
    return preview_btn, speed, voice, voice_btn


@app.cell
def _(deps_ok):
    _ = deps_ok
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    tts = KPipeline(lang_code="a")  # "a" = American English

    def synth(text, voice_name, speed_value, out_path):
        sr = 24000
        parts = []
        for _gs, _ps, a in tts(text, voice=voice_name, speed=speed_value):
            if a is None:
                continue
            parts.append(a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a))
        full = np.concatenate([np.zeros(int(sr * 0.25))] + parts + [np.zeros(int(sr * 0.6))])
        sf.write(str(out_path), full, sr)
        return len(full) / sr

    return sf, synth


@app.cell
def _(WORK, mo, preview_btn, speed, synth, voice):
    mo.stop(not preview_btn.value)
    _p = WORK / "voice_preview.wav"
    synth(
        "The door creaked open. Somewhere in the dark, something had been waiting for me.",
        voice.value,
        speed.value,
        _p,
    )
    mo.audio(str(_p))
    return


@app.cell
def _(WORK, mo, scenes, speed, synth, voice, voice_btn):
    mo.stop(not voice_btn.value)
    _total = 0.0
    for _s in mo.status.progress_bar(scenes, title="Narrating"):
        _total += synth(_s["narration"], voice.value, speed.value, WORK / f"scene_{_s['id']:02d}.wav")
    mo.md(f"✅ Narration done — total **{_total / 60:.1f} min**")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4️⃣ Video
    """)
    return


@app.cell
def _(mo):
    video_size = mo.ui.dropdown(["1920x1080", "1080x1920", "1080x1080"], value="1920x1080", label="Video format")
    video_btn = mo.ui.run_button(label="4️⃣ Build video")
    mo.hstack([video_size, video_btn], justify="start")
    return video_btn, video_size


@app.cell
def _(WORK, math, mo, sf, shutil, subprocess, video_btn, video_size):
    mo.stop(not video_btn.value)
    import imageio_ffmpeg

    _ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

    _pairs = []
    for _img in sorted(WORK.glob("scene_*.png")):
        _wav = _img.with_suffix(".wav")
        if _wav.exists():
            _pairs.append((_img, _wav))
    mo.stop(not _pairs, mo.md("⚠️ You need images **and** narration first (steps 2 and 3)."))

    _tw, _th = map(int, video_size.value.split("x"))
    _fps = 30
    _zmax = 1.12
    _clips = []
    for _k, (_img, _wav) in enumerate(mo.status.progress_bar(_pairs, title="Rendering clips")):
        _dur = sf.info(str(_wav)).duration
        _frames = max(1, math.ceil(_dur * _fps))
        _inc = (_zmax - 1) / _frames
        # Slow Ken Burns: even scenes zoom in, odd scenes zoom out
        _z = f"min(1+{_inc:.7f}*on,{_zmax})" if _k % 2 == 0 else f"max({_zmax}-{_inc:.7f}*on,1)"
        _vf = (
            f"scale={2 * _tw}:{2 * _th}:force_original_aspect_ratio=increase,crop={2 * _tw}:{2 * _th},"
            f"zoompan=z='{_z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={_frames}:s={_tw}x{_th}:fps={_fps},"
            "format=yuv420p"
        )
        _clip = _img.with_suffix(".mp4")
        subprocess.run(
            [
                _ffmpeg, "-y", "-loglevel", "error",
                "-i", str(_img), "-i", str(_wav),
                "-vf", _vf, "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-shortest", str(_clip),
            ],
            check=True,
        )
        _clips.append(_clip)

    _list = WORK / "clips.txt"
    _list.write_text("".join(f"file '{c.resolve()}'\n" for c in _clips))
    _final = WORK / "final.mp4"
    subprocess.run(
        [
            _ffmpeg, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(_list),
            "-c", "copy", "-movflags", "+faststart", str(_final),
        ],
        check=True,
    )
    mo.vstack(
        [
            mo.video(str(_final), controls=True),
            mo.download(data=_final.read_bytes(), filename="story.mp4", label="⬇️ Download video"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
