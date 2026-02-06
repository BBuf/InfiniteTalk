## InfiniteTalk Serving（LightX2V 接口兼容）

这个目录提供一个 **FastAPI + 任务队列 +（可选 torchrun 多进程）** 的 serving 框架，接口路径/返回结构对齐 `LightX2V/lightx2v/server`：

- **POST** `/v1/tasks/video`（JSON）
- **POST** `/v1/tasks/video/form`（multipart form-data）
- **GET** `/v1/tasks/{task_id}/status`
- **GET** `/v1/tasks/{task_id}/result`
- **GET** `/v1/tasks/queue/status`
- **DELETE** `/v1/tasks/{task_id}`
- **GET** `/v1/files/download/{path}`
- **GET** `/v1/service/status`
- **GET** `/v1/service/metadata`

### 关键差异：支持“视频作为条件输入”

为了保持和 LightX2V 一致的字段名，这里仍然使用 `image_path` / `image_file` 作为条件输入字段，但它可以是：

- 图片（`.png/.jpg/...`）
- **视频（`.mp4/.mov/.mkv/...`）**

InfiniteTalk 内部会根据类型自动走：
- 图片：直接作为条件帧
- 视频：从视频中抽取稀疏帧作为条件帧

### 启动

在 `InfiniteTalk/` 目录下运行：

```bash
pip install -r requirements.txt

python -m serving \
  --ckpt_dir weights/Wan2.1-I2V-14B-480P \
  --wav2vec_dir weights/chinese-wav2vec2-base \
  --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
  --host 0.0.0.0 --port 8000
```

多 GPU（可选）：

```bash
torchrun --nproc_per_node=2 -m serving \
  --ckpt_dir ... --wav2vec_dir ... --infinitetalk_dir ... \
  --host 0.0.0.0 --port 8000
```

### 请求示例

#### JSON：图片/视频条件都走 `image_path`

```json
{
  "prompt": "a person is talking",
  "image_path": "examples/single/ref_video.mp4",
  "audio_path": "examples/single/1.wav",
  "infer_steps": 40,
  "target_video_length": 1000,
  "seed": 42
}
```

#### Form：上传图片或视频（字段名仍为 `image_file`）

- `image_file`: 上传 `.png/.jpg` 或 `.mp4`
- `audio_file`: 上传音频（必填）

