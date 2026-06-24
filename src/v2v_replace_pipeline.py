"""Video→Video 角色替换流水线。

原视频按 ≤15s 切分，逐段调用 dreamina multimodal2video：
  --image 角色参考图 + --video 原片片段 → 替换主角后重生成。
  第 2 段起可（v2v_replace.segment_anchor）传入上一段成片末帧作 anchor。
  最后拼接成片。
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

from src.v2v_replace.models import V2VExtension, V2VReplaceJob, V2VSegment
from src.video.video_splitter import (
    JIMENG_MAX_SEGMENT_SEC,
    JIMENG_UPLOAD_MAX_SEC,
    concat_videos,
    ensure_clip_uploadable,
    extract_audio_clip,
    extract_last_frame,
    extract_video_clip,
    plan_segment_spans,
    probe_video_duration,
)

log = logging.getLogger("v2v_replace")

DEFAULT_REPLACE_PROMPT = (
    "参考图片为目标角色或动物的外观，必须严格保持其面容、毛色、体型与画风。"
    "参考视频为原片片段：保持相同镜头、背景、运镜与动作节奏，"
    "将画面中的主要人物或动物替换为参考图片中的形象，环境与其他物体尽量不变。"
)

CONTINUITY_PROMPT_SUFFIX = (
    "第一张参考图为上一段成片末帧，作为场景衔接锚点，本片开头须与之自然衔接；"
    "第二张参考图为目标角色形象，须严格保持一致。"
)


class V2VReplacePipeline:
    """Video→Video 主角替换（参考图定形象）。"""

    def __init__(
        self,
        config_path: Path | str | None = None,
        workspace: Path | str | None = None,
        config: dict | None = None,
    ) -> None:
        if config:
            base_cfg = config
        else:
            from src.config_manager import load_config

            base_cfg = load_config(config_path)

        from src.config_manager import resolve_v2v_replace_config

        self.config = resolve_v2v_replace_config(base_cfg)
        self.workspace = Path(workspace or "workspace/v2v_replace")
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        source_video: Path | str,
        character_image: Path | str,
        *,
        prompt: str | None = None,
        extension_prompt: str | None = None,
        extension_duration: float | None = None,
        max_segment_sec: float | None = None,
        keep_audio: bool | None = None,
        segment_anchor: bool | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        source_video = Path(source_video).resolve()
        character_image = Path(character_image).resolve()
        if not source_video.is_file():
            raise FileNotFoundError(f"源视频不存在: {source_video}")
        if not character_image.is_file():
            raise FileNotFoundError(f"角色参考图不存在: {character_image}")

        v2v_cfg = self.config.get("v2v_replace") or {}
        run_id = uuid.uuid4().hex[:8]
        run_dir = self.workspace / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        char_dest = run_dir / "character_ref.png"
        shutil.copy2(character_image, char_dest)

        job = V2VReplaceJob(
            run_id=run_id,
            source_video=str(source_video),
            character_image=str(char_dest),
            prompt=(prompt or v2v_cfg.get("prompt") or DEFAULT_REPLACE_PROMPT).strip(),
            max_segment_sec=float(
                max_segment_sec
                if max_segment_sec is not None
                else v2v_cfg.get("max_segment_sec", JIMENG_MAX_SEGMENT_SEC)
            ),
            keep_audio=(
                keep_audio
                if keep_audio is not None
                else bool(v2v_cfg.get("keep_audio", True))
            ),
            segment_anchor=(
                segment_anchor
                if segment_anchor is not None
                else bool(v2v_cfg.get("segment_anchor", False))
            ),
        )

        ext_prompt = (extension_prompt or v2v_cfg.get("extension_prompt") or "").strip()
        if ext_prompt:
            job.extension = V2VExtension(
                prompt=ext_prompt,
                duration_sec=float(
                    extension_duration
                    if extension_duration is not None
                    else v2v_cfg.get("extension_duration_sec", 5.0)
                ),
                status="pending",
            )

        if progress_callback:
            progress_callback(0.05, "分析原视频时长…")
        total = probe_video_duration(source_video)
        spans = plan_segment_spans(total, max_seg=job.max_segment_sec)
        if not spans:
            raise RuntimeError("原视频时长为 0，无法切分")

        seg_root = run_dir / "segments"
        seg_root.mkdir(parents=True, exist_ok=True)
        for span in spans:
            seg_dir = seg_root / f"seg_{span.id:03d}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            clip_path = seg_dir / "source.mp4"
            extract_video_clip(
                source_video,
                clip_path,
                start_sec=span.start_sec,
                duration_sec=span.duration_sec,
            )
            audio_path = ""
            if job.keep_audio:
                audio_out = seg_dir / "source_audio.m4a"
                extracted = extract_audio_clip(
                    source_video,
                    audio_out,
                    start_sec=span.start_sec,
                    duration_sec=span.duration_sec,
                    max_duration=job.max_segment_sec,
                )
                if extracted is not None:
                    audio_path = str(extracted)

            job.segments.append(
                V2VSegment(
                    id=span.id,
                    start_sec=span.start_sec,
                    end_sec=span.end_sec,
                    source_clip=str(clip_path),
                    audio_clip=audio_path,
                )
            )

        self._save_job(run_dir, job)
        log.info("v2v 切分完成: %d 段, run_dir=%s", len(job.segments), run_dir)

        return self._process_segments(
            run_dir, job, progress_callback=progress_callback, start_pct=0.15
        )

    def resume(
        self,
        run_dir: Path | str,
        *,
        extension_prompt: str | None = None,
        extension_duration: float | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        run_dir = Path(run_dir)
        job = self._load_job(run_dir)
        v2v_cfg = self.config.get("v2v_replace") or {}
        ext_prompt = (
            extension_prompt or v2v_cfg.get("extension_prompt") or ""
        ).strip()
        if ext_prompt:
            if job.extension and job.extension.enabled and job.extension.status == "done":
                log.info("扩演段已完成，忽略新的 extension_prompt")
            elif job.extension and job.extension.enabled:
                job.extension.prompt = ext_prompt
                if extension_duration is not None:
                    job.extension.duration_sec = float(extension_duration)
                if job.extension.status == "failed":
                    job.extension.status = "pending"
                    job.extension.submit_id = ""
            else:
                job.extension = V2VExtension(
                    prompt=ext_prompt,
                    duration_sec=float(
                        extension_duration
                        if extension_duration is not None
                        else v2v_cfg.get("extension_duration_sec", 5.0)
                    ),
                    status="pending",
                )
            self._save_job(run_dir, job)
        return self._process_segments(
            run_dir, job, progress_callback=progress_callback, start_pct=0.05
        )

    def _process_segments(
        self,
        run_dir: Path,
        job: V2VReplaceJob,
        *,
        progress_callback: Callable[[float, str], None] | None,
        start_pct: float,
    ) -> dict[str, Any]:
        video_gen = self._create_backend()

        if job.segment_anchor:
            self._process_segments_sequential(
                run_dir, job, video_gen,
                progress_callback=progress_callback,
                start_pct=start_pct,
            )
        else:
            self._process_segments_parallel(
                run_dir, job, video_gen,
                progress_callback=progress_callback,
                start_pct=start_pct,
            )

        segments_ready = all(
            seg.output_clip and Path(seg.output_clip).is_file()
            for seg in job.segments
        )
        if segments_ready and job.extension and job.extension.enabled:
            self._process_extension(
                run_dir,
                job,
                video_gen,
                progress_callback=progress_callback,
                start_pct=start_pct,
            )

        try:
            video_gen.close()
        except Exception:
            pass

        pending = job.pending_count()
        if pending > 0:
            return {
                "status": "pending",
                "pending_count": pending,
                "run_dir": str(run_dir),
                "job": job.model_dump(),
                "message": (
                    f"仍有 {pending} 段未就绪，请稍后执行 "
                    f"python main.py resume-replace-video {run_dir}"
                ),
            }

        if progress_callback:
            progress_callback(0.88, "拼接成片…")
        final_path = self._assemble(run_dir, job)
        job.final_video = str(final_path)
        self._save_job(run_dir, job)

        if progress_callback:
            progress_callback(1.0, "完成")
        return {
            "status": "completed",
            "video_path": str(final_path),
            "run_dir": str(run_dir),
            "job": job.model_dump(),
            "segment_count": len(job.segments),
            "has_extension": bool(job.extension and job.extension.enabled),
        }

    def _process_segments_parallel(
        self,
        run_dir: Path,
        job: V2VReplaceJob,
        video_gen,
        *,
        progress_callback: Callable[[float, str], None] | None,
        start_pct: float,
    ) -> None:
        total = len(job.segments)
        for i, seg in enumerate(job.segments):
            if not seg.submit_id or seg.output_clip:
                continue
            out_dir = run_dir / "segments" / f"seg_{seg.id:03d}"
            if progress_callback:
                pct = start_pct + 0.6 * (i / max(total, 1))
                progress_callback(pct, f"轮询片段 {seg.id}/{total}…")
            self._poll_segment(run_dir, job, seg, video_gen, out_dir)

        for j, seg in enumerate(job.segments):
            if seg.output_clip and Path(seg.output_clip).is_file():
                seg.status = "done"
                continue
            if seg.submit_id and not seg.output_clip:
                continue
            if progress_callback:
                pct = start_pct + 0.6 * ((total + j) / max(total * 2, 1))
                progress_callback(pct, f"提交替换 {seg.id}/{total}…")
            try:
                self._submit_segment(run_dir, job, seg, video_gen)
            except Exception as exc:
                log.error("片段 %d 提交失败: %s", seg.id, exc)
                seg.status = "failed"
            self._save_job(run_dir, job)

    def _process_segments_sequential(
        self,
        run_dir: Path,
        job: V2VReplaceJob,
        video_gen,
        *,
        progress_callback: Callable[[float, str], None] | None,
        start_pct: float,
    ) -> None:
        """segment_anchor 开启时：按 id 顺序逐段轮询/提交。"""
        total = len(job.segments)
        for seg in sorted(job.segments, key=lambda s: s.id):
            if seg.output_clip and Path(seg.output_clip).is_file():
                seg.status = "done"
                continue

            if seg.submit_id:
                out_dir = run_dir / "segments" / f"seg_{seg.id:03d}"
                if progress_callback:
                    pct = start_pct + 0.6 * ((seg.id - 1) / max(total, 1))
                    progress_callback(pct, f"轮询片段 {seg.id}/{total}…")
                self._poll_segment(run_dir, job, seg, video_gen, out_dir)
                if not (seg.output_clip and Path(seg.output_clip).is_file()):
                    break
                continue

            if not self._previous_segment_ready(job, seg):
                break

            if progress_callback:
                pct = start_pct + 0.6 * ((total + seg.id - 1) / max(total * 2, 1))
                progress_callback(pct, f"提交替换 {seg.id}/{total}…")
            try:
                self._submit_segment(run_dir, job, seg, video_gen)
            except Exception as exc:
                log.error("片段 %d 提交失败: %s", seg.id, exc)
                seg.status = "failed"
            self._save_job(run_dir, job)
            if seg.submit_id and not seg.output_clip:
                break

    def _poll_segment(
        self,
        run_dir: Path,
        job: V2VReplaceJob,
        seg: V2VSegment,
        video_gen,
        out_dir: Path,
    ) -> None:
        try:
            result = video_gen.poll_submit_id(
                seg.submit_id,
                out_dir,
                duration_hint=seg.end_sec - seg.start_sec,
            )
            out_path = out_dir / "output.mp4"
            if result.video_path.is_file():
                if result.video_path.resolve() != out_path.resolve():
                    shutil.copy2(result.video_path, out_path)
                seg.output_clip = str(out_path)
                seg.status = "done"
        except Exception as exc:
            log.error("片段 %d 轮询失败: %s", seg.id, exc)
            seg.status = "failed"
        self._save_job(run_dir, job)

    def _process_extension(
        self,
        run_dir: Path,
        job: V2VReplaceJob,
        video_gen,
        *,
        progress_callback: Callable[[float, str], None] | None,
        start_pct: float,
    ) -> None:
        ext = job.extension
        if not ext or not ext.enabled:
            return

        if ext.output_clip and Path(ext.output_clip).is_file() and ext.status == "done":
            return

        ext_dir = run_dir / "extension"
        ext_dir.mkdir(parents=True, exist_ok=True)

        if ext.submit_id and not ext.output_clip:
            if progress_callback:
                progress_callback(start_pct + 0.72, "轮询扩演片段…")
            try:
                result = video_gen.poll_submit_id(
                    ext.submit_id,
                    ext_dir,
                    duration_hint=ext.duration_sec,
                )
                out_path = ext_dir / "output.mp4"
                if result.video_path.is_file():
                    if result.video_path.resolve() != out_path.resolve():
                        shutil.copy2(result.video_path, out_path)
                    ext.output_clip = str(out_path)
                    ext.status = "done"
            except Exception as exc:
                log.error("扩演片段轮询失败: %s", exc)
                ext.status = "failed"
            self._save_job(run_dir, job)
            return

        if ext.submit_id:
            return

        if progress_callback:
            progress_callback(start_pct + 0.75, "提交扩演片段…")

        last_seg = max(job.segments, key=lambda s: s.id)
        last_clip = Path(last_seg.output_clip)
        if not last_clip.is_file():
            raise RuntimeError(f"扩演锚点缺失：最后一段 output 不存在: {last_clip}")

        anchor_path = ext_dir / "anchor_frame.png"
        if not ext.anchor_frame or not Path(ext.anchor_frame).is_file():
            extract_last_frame(last_clip, anchor_path)
            ext.anchor_frame = str(anchor_path)
        else:
            anchor_path = Path(ext.anchor_frame)

        out_dir = ext_dir
        duration = min(
            ext.duration_sec,
            JIMENG_UPLOAD_MAX_SEC - 0.01,
        )
        try:
            result = video_gen.generate(
                ext.prompt,
                image_path=anchor_path,
                duration=duration,
                output_dir=out_dir,
            )
        except Exception as exc:
            log.error("扩演片段提交失败: %s", exc)
            ext.status = "failed"
            self._save_job(run_dir, job)
            return

        if result.pending and result.submit_id:
            ext.submit_id = result.submit_id
            ext.status = "submitted"
            log.info("扩演片段已提交: submit_id=%s", result.submit_id)
            self._save_job(run_dir, job)
            return

        out_path = ext_dir / "output.mp4"
        if result.video_path.is_file():
            if result.video_path.resolve() != out_path.resolve():
                shutil.copy2(result.video_path, out_path)
            ext.output_clip = str(out_path)
            ext.status = "done"
        self._save_job(run_dir, job)

    def _submit_segment(
        self,
        run_dir: Path,
        job: V2VReplaceJob,
        seg: V2VSegment,
        video_gen,
    ) -> None:
        if not hasattr(video_gen, "generate_multimodal"):
            raise RuntimeError(
                "v2v 角色替换需 dreamina multimodal2video（videogen.backend=jimeng-cli）"
            )

        source = Path(seg.source_clip)
        if not source.is_file():
            raise FileNotFoundError(f"片段 {seg.id} 缺少 source_clip: {source}")

        ensure_clip_uploadable(source)

        out_dir = run_dir / "segments" / f"seg_{seg.id:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        duration = min(
            probe_video_duration(source),
            JIMENG_UPLOAD_MAX_SEC - 0.01,
        )
        audio_paths = []
        if seg.audio_clip and Path(seg.audio_clip).is_file():
            audio_paths.append(Path(seg.audio_clip))

        char_image = Path(job.character_image)
        image_paths = [char_image]
        prompt = job.prompt
        anchor_path = self._resolve_segment_anchor(run_dir, job, seg)
        if anchor_path is not None:
            image_paths = [anchor_path, char_image]
            prompt = f"{job.prompt.strip()} {CONTINUITY_PROMPT_SUFFIX}".strip()

        result = video_gen.generate_multimodal(
            prompt,
            image_paths=image_paths,
            video_paths=[source],
            audio_paths=audio_paths or None,
            duration=duration,
            output_dir=out_dir,
        )

        if result.pending and result.submit_id:
            seg.submit_id = result.submit_id
            seg.status = "submitted"
            log.info("片段 %d 已提交: submit_id=%s", seg.id, result.submit_id)
            return

        out_path = out_dir / "output.mp4"
        if result.video_path.is_file():
            if result.video_path.resolve() != out_path.resolve():
                shutil.copy2(result.video_path, out_path)
            seg.output_clip = str(out_path)
            seg.status = "done"

    @staticmethod
    def _get_segment(job: V2VReplaceJob, seg_id: int) -> V2VSegment | None:
        for seg in job.segments:
            if seg.id == seg_id:
                return seg
        return None

    @classmethod
    def _previous_segment_ready(cls, job: V2VReplaceJob, seg: V2VSegment) -> bool:
        if seg.id <= 1:
            return True
        prev = cls._get_segment(job, seg.id - 1)
        if prev is None:
            return False
        return bool(prev.output_clip and Path(prev.output_clip).is_file())

    def _resolve_segment_anchor(
        self,
        run_dir: Path,
        job: V2VReplaceJob,
        seg: V2VSegment,
    ) -> Path | None:
        """第 2 段起：取上一段 output 末帧作为衔接 anchor（须 job.segment_anchor）。"""
        if not job.segment_anchor or seg.id <= 1:
            return None
        prev = self._get_segment(job, seg.id - 1)
        if prev is None:
            raise RuntimeError(f"片段 {seg.id} 缺少上一段 seg_{seg.id - 1:03d}")
        prev_out = Path(prev.output_clip)
        if not prev_out.is_file():
            raise RuntimeError(
                f"片段 {seg.id} 衔接锚点缺失：上一段 output 不存在: {prev_out}"
            )
        out_dir = run_dir / "segments" / f"seg_{seg.id:03d}"
        anchor_path = out_dir / "anchor_frame.png"
        if seg.anchor_frame and Path(seg.anchor_frame).is_file():
            return Path(seg.anchor_frame)
        extract_last_frame(prev_out, anchor_path)
        seg.anchor_frame = str(anchor_path)
        log.info("片段 %d anchor 取自 seg_%03d 末帧", seg.id, prev.id)
        return anchor_path

    def _assemble(self, run_dir: Path, job: V2VReplaceJob) -> Path:
        clips = []
        for seg in sorted(job.segments, key=lambda s: s.id):
            path = Path(seg.output_clip)
            if not path.is_file():
                raise RuntimeError(f"片段 {seg.id} 缺少输出视频: {path}")
            clips.append(path)
        ext = job.extension
        if ext and ext.enabled and ext.output_clip:
            ext_path = Path(ext.output_clip)
            if not ext_path.is_file():
                raise RuntimeError(f"扩演片段缺少输出视频: {ext_path}")
            clips.append(ext_path)
        final = run_dir / f"replaced_{job.run_id}.mp4"
        return concat_videos(clips, final)

    def _create_backend(self):
        from src.videogen.jimeng_cli_backend import merge_jimeng_cli_videogen_config
        from src.videogen.video_generator import create_video_generator

        vg = merge_jimeng_cli_videogen_config(
            dict(self.config.get("videogen", {})),
            self.config.get("imagegen"),
        )
        if vg.get("backend") != "jimeng-cli":
            raise RuntimeError(
                "v2v_replace 需要 videogen.backend=jimeng-cli（dreamina multimodal2video）"
            )
        return create_video_generator(vg)

    @staticmethod
    def _save_job(run_dir: Path, job: V2VReplaceJob) -> None:
        (run_dir / "job.json").write_text(
            job.model_dump_json(indent=2), encoding="utf-8"
        )

    @staticmethod
    def _load_job(run_dir: Path) -> V2VReplaceJob:
        path = run_dir / "job.json"
        if not path.is_file():
            raise FileNotFoundError(f"未找到 job.json: {path}")
        return V2VReplaceJob.model_validate_json(path.read_text(encoding="utf-8"))
