# API 参考

仅列出稳定的公开接口；带 `_` 前缀的均为内部实现。

## 配置

### `sonoloc.config.SonolocConfig`

共享参数数据类。关键字段：`sample_rate`、`n_fft`、`hop_length`、`n_mels`、
`fmin`、`fmax`、`label_rate`、`sound_speed`、`array`。

- `frames_per_second -> float`
- `label_hop -> int`
- `to_dict() / from_dict(d)`
- `save(path) / load(path)`：YAML 序列化。

## IO 与几何

- `sonoloc.io.geometry.MicArray(name, positions)` — `n_mics`、`pairs()`、
  `from_spherical(name, az_deg, el_deg, radius)`。
- `tetrahedral_array(radius=0.042) -> MicArray`
- `get_array(name) -> MicArray`
- `sph2cart(az, el, r)` / `cart2sph(xyz)`
- `sonoloc.io.audio.load_audio(path, sample_rate=None) -> (signal, sr)`
- `save_audio(path, signal, sample_rate)` / `resample_signal(signal, orig, target)`

## 特征

- `sonoloc.features.stft.stft(signal, n_fft, hop_length, ...) -> ndarray`
- `sonoloc.features.logmel.LogMelExtractor(sample_rate, n_fft, n_mels, ...)`
- `sonoloc.features.gcc.gcc_phat(sig, ref, ...)` / `estimate_tdoa(sig, ref)`
- `sonoloc.features.intensity.intensity_vectors(foa_spec)` / `intensity_doa(foa_spec)`
- `sonoloc.features.pipeline.FeaturePipeline(config, array)` — 可调用。

## 定位

- `sonoloc.localization.srp_phat(signal, array, config, grid=None) -> (az_deg, el_deg)`
- `sonoloc.localization.music(signal, array, config, n_sources=1) -> (az_deg, el_deg)`
- `sonoloc.localization.make_grid(az_step, el_step, el_min, el_max) -> SphereGrid`
- `sonoloc.localization.steering_vector(positions, directions, freqs, c)`

## 检测

- `sonoloc.detection.pooling.pool(frame_probs, method, axis)`
- `sonoloc.detection.mil.clip_labels(frame_probs, method, threshold)`
- `median_filter(frame_probs, size)` / `frames_to_events(frame_active, hop_seconds)`

## 标签

- `sonoloc.labels.accdoa.encode_accdoa(activity, azimuth, elevation)`
- `decode_accdoa(accdoa, threshold=0.5) -> dict`
- `encode_multi_accdoa(...)` / `decode_multi_accdoa(...)`
- `sonoloc.labels.events.EventVocabulary(classes=None)`

## 指标

- `sonoloc.metrics.segment_detection_scores(reference, prediction, frames_per_segment=10)`
- `localization_scores(ref_active, ref_az, ref_el, pred_active, pred_az, pred_el)`
- `seld_score(error_rate, f_score, localization_error, localization_recall)`
- `aggregate_seld(detection, localization) -> dict`

## 数据

- `sonoloc.data.scene.Scene` / `SoundEvent`
- `sonoloc.data.simulate.simulate_scene(scene, array, config, snr_db=None, seed=0)`
- `random_scene(n_classes, duration, sample_rate, n_events, seed)`
- `sonoloc.data.noise.add_noise_at_snr(signal, noise, snr_db)`

## 模型（需要 `[torch]`）

- `sonoloc.models.crnn.CRNN(in_channels, n_mels, ...)`
- `sonoloc.models.seld_model.SeldModel(in_channels, n_classes, ...)`
- `sonoloc.models.accdoa.ACCDOAHead(in_dim, n_classes)`
