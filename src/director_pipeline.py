"""DirectorPipeline - AI短视频导演流水线

新流程：灵感 → 视频方案 → 结构化脚本 → 逐段生成素材 → 合成视频

与旧 Pipeline 的区别：
- 旧流程：文本 → 分段 → prompt → 图 → 音 → 合成（pipeline.py）
- 新流程：灵感 → 视频方案 → 结构化脚本 → 逐段素材 → 合成
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Callable

from src.scriptplan.character_consistency import DirectorCharacterConsistency

log = logging.getLogger("director")

# 画面描述 → 图片 prompt 的系统提示词（专用于导演流水线）
_VISUAL_TO_IMAGE_PROMPT = """\
你是一个中英翻译专家，负责将中文画面描述翻译为 Stable Diffusion 图片生成 prompt。

规则：
1. 直接翻译画面描述，保留所有细节
2. 角色性别是最重要的信息，必须明确翻译：
   - 男人/男性/男孩 → man/male/boy
   - 女人/女性/女孩 → woman/female/girl
   - 如果原文有多个角色，每个角色的性别和外观必须分别翻译
3. 翻译角色的外观：年龄、发型、发色、服装、体型、表情
4. 翻译场景：环境、光线、氛围
5. 输出格式：英文关键词短语，逗号分隔
6. 末尾添加：highly detailed, cinematic composition, dramatic lighting, 4K

输出：仅输出英文 prompt，不要任何解释。
"""

# 图生视频：从首帧 image_prompt 派生时的运镜描述
_MOTION_VIDEO_HINTS: dict[str, str] = {
    "static": "static camera, subtle natural movement in scene",
    "push_in": "slow dolly in toward the main subject",
    "pan": "gentle horizontal pan following the action",
    "zoom": "slow cinematic zoom on the focal point",
    "orbit": "smooth orbital camera movement around the subject",
    "reveal": "slow reveal pan uncovering the scene",
}

_VIDEO_PROMPT_SUFFIX = (
    "stable character appearance, natural smooth movements, cinematic quality, 4K"
)


class DirectorPromptError(RuntimeError):
    """导演流水线视觉 prompt LLM 翻译失败（不回退本地规则）。"""


class DirectorPipeline:
    """AI短视频导演流水线。

    用法::

        pipe = DirectorPipeline(config_path="config.yaml")
        result = pipe.run("一个关于时间旅行者的悬疑故事")
        print(result["video_path"])
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        workspace: Path | str | None = None,
        config: dict | None = None,
    ):
        """初始化导演流水线。

        Args:
            config_path: YAML 配置文件路径
            workspace: 工作目录，默认 workspace/videos
            config: 直接传入配置字典（优先于 config_path）
        """
        if config:
            base_cfg = config
        else:
            from src.config_manager import load_config

            base_cfg = load_config(config_path)

        from src.config_manager import resolve_pipeline_config

        self.config = resolve_pipeline_config(base_cfg, "director")

        self.workspace = Path(workspace or "workspace/videos")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._llm_cached = None  # 懒初始化 LLM client

    def run(
        self,
        inspiration: str,
        target_duration: int = 45,
        budget: str = "low",
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        """完整流程：灵感 → 视频。

        Args:
            inspiration: 用户灵感/创意/故事梗概
            target_duration: 目标时长(秒)，默认 45
            budget: 预算档位 (free/low/medium/high)
            progress_callback: 进度回调 (progress_pct, description)

        Returns:
            包含 video_path, script, idea, segments, duration, run_dir 的字典
        """
        run_id = uuid.uuid4().hex[:8]
        run_dir = self.workspace / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        def _notify(pct: float, desc: str) -> None:
            if progress_callback:
                progress_callback(pct, desc)

        # 初始化 LLM
        llm = self._get_llm()

        # === Stage 1: 视频方案 ===
        _notify(0.05, "正在策划视频方案...")
        idea = self._plan_idea(llm, inspiration, target_duration)
        log.info(
            "视频方案: type=%s, duration=%ds, segments=%d",
            idea.video_type, idea.target_duration, idea.segment_count,
        )

        # === Stage 2: 结构化脚本 ===
        _notify(0.10, "正在生成脚本...")
        try:
            script = self._plan_script(llm, idea, inspiration)
        except Exception as exc:
            from src.scriptplan.script_planner import ScriptPlanError

            if isinstance(exc, ScriptPlanError):
                raise RuntimeError(str(exc)) from exc
            raise
        if not script.segments:
            raise RuntimeError(
                "脚本生成失败：未产生任何分段。"
                "请重试或缩短目标时长（例如 -d 45）。"
            )
        log.info(
            "脚本生成: title=%s, segments=%d, duration=%.1fs",
            script.title, len(script.segments), script.total_duration,
        )

        # === Stage 3: 素材策略 ===
        _notify(0.15, "正在规划素材...")
        script = self._assign_assets(script, budget)

        # 保存脚本
        self._save_script(run_dir, script)

        # === Stage 4: 逐段生成配音 ===
        _notify(0.20, "正在生成配音...")
        self._generate_voices(script, run_dir, progress_callback)
        self._save_script(run_dir, script)

        # === Stage 5: 逐段生成画面 ===
        _notify(0.50, "正在生成画面...")
        self._generate_visuals(script, run_dir, budget, progress_callback)
        self._save_script(run_dir, script)

        pending = self._count_pending_videos(script)
        if pending > 0:
            _notify(0.84, f"已提交 {pending} 段视频，即梦排队中…")
            return {
                "status": "pending_video",
                "pending_count": pending,
                "video_path": "",
                "script": script.model_dump(),
                "idea": idea.model_dump(),
                "segments": [s.model_dump() for s in script.segments],
                "duration": script.total_duration,
                "run_dir": str(run_dir),
                "message": (
                    f"已异步提交 {pending} 段视频任务（即梦后台排队）。"
                    f"稍后执行: python main.py resume-video {run_dir}"
                ),
            }

        # === Stage 6: 合成视频 ===
        _notify(0.85, "正在合成视频...")
        import re as _re
        safe_title = _re.sub(r'[^\w\u4e00-\u9fff-]', '_', script.title or 'video')[:50]
        output_path = run_dir / f"{safe_title}_{run_id}.mp4"
        final_path = self._assemble_video(script, run_dir, output_path)

        _notify(1.0, "完成!")

        return {
            "status": "completed",
            "video_path": str(final_path),
            "script": script.model_dump(),
            "idea": idea.model_dump(),
            "segments": [s.model_dump() for s in script.segments],
            "duration": script.total_duration,
            "run_dir": str(run_dir),
        }

    def resume(
        self,
        run_dir: Path | str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        """续跑：轮询 pending 视频任务 → 合成成片。"""
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"工作目录不存在: {run_dir}")

        script = self._load_script(run_dir)
        video_gen = self._try_create_video_generator()
        if video_gen is None:
            raise RuntimeError("无法创建视频生成器，请检查 director.videogen 配置")

        pending_segs = [
            s for s in script.segments
            if s.video_submit_id and not self._segment_has_video_asset(s, run_dir)
        ]
        retry_segs = self._segments_needing_video_resubmit(script, run_dir)
        total = len(pending_segs) + len(retry_segs)

        for i, seg in enumerate(pending_segs):
            if progress_callback:
                pct = 0.10 + 0.70 * (i / max(total, 1))
                progress_callback(pct, f"轮询视频 {i + 1}/{total} (seg {seg.id})…")
            clip_dir = run_dir / f"video_{seg.id:03d}"
            clip_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = video_gen.poll_submit_id(
                    seg.video_submit_id,
                    clip_dir,
                    duration_hint=seg.duration_sec,
                )
                out_path = run_dir / f"video_{seg.id:03d}.mp4"
                if result.video_path.is_file() and result.video_path.resolve() != out_path.resolve():
                    import shutil
                    shutil.copy2(result.video_path, out_path)
                elif result.video_path.is_file():
                    out_path = result.video_path
                seg.asset_path = str(out_path)
                log.info("segment %d 视频就绪: %s", seg.id, out_path.name)
            except Exception as exc:
                log.error("segment %d 视频轮询失败: %s", seg.id, exc)
            self._save_script(run_dir, script)

        for j, seg in enumerate(retry_segs):
            if progress_callback:
                idx = len(pending_segs) + j
                pct = 0.10 + 0.70 * (idx / max(total, 1))
                progress_callback(pct, f"重新提交视频 {j + 1}/{len(retry_segs)} (seg {seg.id})…")
            image_path = run_dir / f"img_{seg.id:03d}.png"
            if not image_path.is_file():
                log.warning("segment %d 缺少 img，跳过视频重提交", seg.id)
                continue
            try:
                self._generate_segment_visual(
                    seg, run_dir, None, video_gen, visual_bible=script.visual_bible,
                    consistency=DirectorCharacterConsistency.from_config(
                        self.config, script.visual_bible
                    ),
                )
            except Exception as exc:
                log.error("segment %d 视频重提交失败: %s", seg.id, exc)
            self._save_script(run_dir, script)

        if video_gen:
            try:
                video_gen.close()
            except Exception:
                pass

        still_pending = self._count_pending_videos(script, run_dir)
        if still_pending > 0:
            if progress_callback:
                progress_callback(0.85, f"仍有 {still_pending} 段排队/失败")
            return {
                "status": "pending_video",
                "pending_count": still_pending,
                "video_path": "",
                "script": script.model_dump(),
                "segments": [s.model_dump() for s in script.segments],
                "duration": script.total_duration,
                "run_dir": str(run_dir),
                "message": (
                    f"仍有 {still_pending} 段视频未就绪，请稍后再次执行 "
                    f"python main.py resume-video {run_dir}"
                ),
            }

        if progress_callback:
            progress_callback(0.88, "正在合成视频…")
        import re as _re
        run_id = run_dir.name.replace("run_", "")
        safe_title = _re.sub(r'[^\w\u4e00-\u9fff-]', '_', script.title or 'video')[:50]
        output_path = run_dir / f"{safe_title}_{run_id}.mp4"
        final_path = self._assemble_video(script, run_dir, output_path)
        if progress_callback:
            progress_callback(1.0, "完成!")
        return {
            "status": "completed",
            "video_path": str(final_path),
            "script": script.model_dump(),
            "segments": [s.model_dump() for s in script.segments],
            "duration": script.total_duration,
            "run_dir": str(run_dir),
        }

    @staticmethod
    def _save_script(run_dir: Path, script) -> None:
        path = run_dir / "script.json"
        path.write_text(script.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _load_script(run_dir: Path):
        from src.scriptplan.models import VideoScript

        path = run_dir / "script.json"
        if not path.is_file():
            raise FileNotFoundError(f"未找到 script.json: {path}")
        return VideoScript.model_validate_json(path.read_text(encoding="utf-8"))

    def _fallback_to_image(self) -> bool:
        director = self.config.get("director") or {}
        vg = director.get("videogen") or self.config.get("videogen") or {}
        return bool(vg.get("fallback_to_image", False))

    @staticmethod
    def _segment_has_video_asset(seg, run_dir: Path) -> bool:
        if seg.asset_path:
            p = Path(seg.asset_path)
            if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}:
                return True
        fallback = run_dir / f"video_{seg.id:03d}.mp4"
        return fallback.is_file()

    @staticmethod
    def _segments_needing_video_resubmit(script, run_dir: Path) -> list:
        from src.scriptplan.models import AssetType

        out = []
        for seg in script.segments:
            if seg.asset_type not in (AssetType.IMAGE2VIDEO, AssetType.VIDEO):
                continue
            if seg.video_submit_id:
                continue
            if DirectorPipeline._segment_has_video_asset(seg, run_dir):
                continue
            if (run_dir / f"img_{seg.id:03d}.png").is_file():
                out.append(seg)
        return out

    @staticmethod
    def _count_pending_videos(script, run_dir: Path | None = None) -> int:
        from src.scriptplan.models import AssetType

        count = 0
        for seg in script.segments:
            if seg.asset_type not in (AssetType.IMAGE2VIDEO, AssetType.VIDEO):
                continue
            if run_dir is not None and DirectorPipeline._segment_has_video_asset(seg, run_dir):
                continue
            if seg.video_submit_id and not seg.asset_path:
                count += 1
            elif seg.video_submit_id and seg.asset_path:
                if not Path(seg.asset_path).is_file():
                    count += 1
            elif not seg.video_submit_id and not seg.asset_path:
                count += 1
        return count

    def _get_llm(self):
        """获取或创建缓存的 LLM client。"""
        if self._llm_cached is None:
            from src.llm.llm_client import create_llm_client
            self._llm_cached = create_llm_client(self.config.get("llm", {}))
        return self._llm_cached

    # ------------------------------------------------------------------
    # Stage 1: 视频方案
    # ------------------------------------------------------------------

    @staticmethod
    def _plan_idea(llm, inspiration: str, target_duration: int):
        """调用 IdeaPlanner 生成视频方案。"""
        from src.scriptplan.idea_planner import IdeaPlanner
        return IdeaPlanner(llm).plan(inspiration, target_duration)

    # ------------------------------------------------------------------
    # Stage 2: 结构化脚本
    # ------------------------------------------------------------------

    @staticmethod
    def _plan_script(llm, idea, inspiration: str):
        """调用 ScriptPlanner 生成结构化脚本。"""
        from src.scriptplan.script_planner import ScriptPlanner
        return ScriptPlanner(llm).plan(idea, inspiration)

    # ------------------------------------------------------------------
    # Stage 3: 素材策略
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_assets(script, budget: str):
        """调用 AssetStrategy 为每段分配素材类型。"""
        from src.scriptplan.asset_strategy import AssetStrategy
        return AssetStrategy().assign(script, budget)

    # ------------------------------------------------------------------
    # Stage 4: 逐段配音
    # ------------------------------------------------------------------

    def _generate_voices(
        self,
        script,
        run_dir: Path,
        progress_callback: Callable | None = None,
    ) -> None:
        """为每段生成配音和字幕。

        结果写入 seg.audio_path / seg.srt_path。
        """
        from src.tts.tts_engine import TTSEngine
        from src.tts.subtitle_generator import SubtitleGenerator

        tts_config = self.config.get("tts", {})
        sub_gen = SubtitleGenerator(self.config.get("subtitle", {}))

        total = len(script.segments)
        for i, seg in enumerate(script.segments):
            if not seg.voiceover:
                continue

            if progress_callback:
                pct = 0.20 + 0.30 * (i / max(total, 1))
                progress_callback(pct, f"配音 {i + 1}/{total}...")

            # 为每段创建独立 TTS（段落可能有不同语速）
            seg_tts_config = dict(tts_config)
            if seg.voice_params and seg.voice_params.speed:
                seg_tts_config["rate"] = seg.voice_params.speed

            engine = TTSEngine(seg_tts_config)
            audio_path = run_dir / f"audio_{seg.id:03d}.mp3"
            srt_path = run_dir / f"sub_{seg.id:03d}.srt"

            try:
                audio_file, boundaries = engine.synthesize(
                    seg.voiceover, audio_path,
                )
                srt_file = sub_gen.generate_srt(
                    boundaries, seg.voiceover, srt_path,
                )
                seg.audio_path = str(audio_file)
                seg.srt_path = str(srt_file)
            except Exception as exc:
                log.error("配音生成失败 segment %d: %s", seg.id, exc)

    # ------------------------------------------------------------------
    # Stage 5: 逐段画面
    # ------------------------------------------------------------------

    def _generate_visuals(
        self,
        script,
        run_dir: Path,
        budget: str,
        progress_callback: Callable | None = None,
    ) -> None:
        """为每段生成画面素材（图片或视频）。

        结果写入 seg.asset_path / seg.image_prompt / seg.video_prompt。
        如果视频生成器不可用，自动降级为静图。
        """
        from src.scriptplan.models import AssetType
        from src.imagegen.image_generator import create_image_generator

        consistency = DirectorCharacterConsistency.from_config(
            self.config, getattr(script, "visual_bible", None)
        )
        imagegen_cfg = dict(self.config.get("imagegen", {}))
        if consistency.enabled and script.visual_bible:
            imagegen_cfg["negative_prompt"] = consistency.merged_negative_prompt(
                str(imagegen_cfg.get("negative_prompt", ""))
            )
        image_gen = create_image_generator(imagegen_cfg)

        # 提取 visual_bible 用于全片一致性
        visual_bible = getattr(script, "visual_bible", None)

        # 初始化视频生成器（如果需要且可用）
        video_gen = None
        needs_video = any(
            s.asset_type in (AssetType.IMAGE2VIDEO, AssetType.VIDEO)
            for s in script.segments
        )
        if needs_video:
            video_gen = self._try_create_video_generator()
            if video_gen is None and not self._fallback_to_image():
                raise RuntimeError(
                    "视频生成器不可用。已设置 fallback_to_image=false，拒绝降级静图。"
                )

        total = len(script.segments)
        for i, seg in enumerate(script.segments):
            if progress_callback:
                pct = 0.50 + 0.35 * (i / max(total, 1))
                progress_callback(pct, f"画面 {i + 1}/{total}...")

            try:
                self._generate_segment_visual(
                    seg, run_dir, image_gen, video_gen,
                    visual_bible=visual_bible,
                    consistency=consistency,
                )
            except Exception as exc:
                log.error("素材生成失败 segment %d: %s", seg.id, exc)
                if not self._fallback_to_image():
                    seg.asset_path = ""
                    seg.video_submit_id = ""
                else:
                    seg.asset_path = ""
            self._save_script(run_dir, script)

        # 清理视频生成器
        if video_gen:
            try:
                video_gen.close()
            except Exception:
                pass

    def _try_create_video_generator(self):
        """尝试创建视频生成器，失败则返回 None。"""
        try:
            from src.videogen.jimeng_cli_backend import merge_jimeng_cli_videogen_config
            from src.videogen.video_generator import create_video_generator

            videogen_config = merge_jimeng_cli_videogen_config(
                dict(self.config.get("videogen", {})),
                self.config.get("imagegen"),
            )
            if videogen_config.get("backend"):
                return create_video_generator(videogen_config)
        except Exception as exc:
            if self._fallback_to_image():
                log.warning("视频生成器初始化失败，降级为静图: %s", exc)
            else:
                log.error("视频生成器初始化失败: %s", exc)
        return None

    def _apply_video_result(self, seg, run_dir: Path, result) -> None:
        """写入视频生成结果（含异步 pending）。"""
        import shutil

        if result.pending and result.submit_id:
            seg.video_submit_id = result.submit_id
            seg.asset_path = ""
            log.info(
                "segment %d 视频已提交排队: submit_id=%s",
                seg.id,
                result.submit_id,
            )
            return

        out_path = run_dir / f"video_{seg.id:03d}.mp4"
        if result.video_path.is_file():
            if result.video_path.resolve() != out_path.resolve():
                shutil.copy2(result.video_path, out_path)
            seg.asset_path = str(out_path)
        seg.video_submit_id = result.submit_id or seg.video_submit_id

    def _generate_segment_visual(
        self, seg, run_dir: Path, image_gen, video_gen,
        visual_bible=None,
        consistency=None,
    ) -> None:
        """为单个段落生成画面素材。"""
        from src.scriptplan.character_consistency import DirectorCharacterConsistency
        from src.scriptplan.models import AssetType

        if consistency is None:
            consistency = DirectorCharacterConsistency.from_config(
                self.config, visual_bible
            )
        imagegen_cfg = self.config.get("imagegen", {})
        image_path = run_dir / f"img_{seg.id:03d}.png"
        if not image_path.is_file():
            if image_gen is None:
                raise RuntimeError(f"segment {seg.id}: 缺少 img 且无法生图")
            image_prompt = self._visual_to_prompt(seg.visual, seg.id, visual_bible)
            image_prompt = consistency.enrich_image_prompt(image_prompt, seg.visual)
            seg.image_prompt = image_prompt

            anchor_path = consistency.get_anchor_path(run_dir)
            use_i2i = (
                consistency.should_use_anchor_i2img(seg.id, run_dir)
                and consistency.supports_reference_image(imagegen_cfg)
                and anchor_path is not None
            )
            if use_i2i and anchor_path is not None:
                edit_prompt = consistency.build_anchor_edit_prompt(image_prompt)
                log.info(
                    "segment %d: 首镜 img2img 参考 %s", seg.id, anchor_path.name
                )
                log.info("[ImageGen] segment %d 生图 prompt:\n%s", seg.id, edit_prompt)
                image = image_gen.generate(
                    edit_prompt, reference_images=[anchor_path]
                )
            else:
                log.info("[ImageGen] segment %d 生图 prompt:\n%s", seg.id, image_prompt)
                image = image_gen.generate(image_prompt)
            image.save(str(image_path))
            if consistency.is_anchor_segment(seg.id):
                consistency.set_anchor(run_dir, image_path, seg.id)
        elif not seg.image_prompt:
            seg.image_prompt = consistency.enrich_image_prompt(
                self._visual_to_prompt(seg.visual, seg.id, visual_bible),
                seg.visual,
            )

        if self._segment_has_video_asset(seg, run_dir):
            seg.asset_path = str(run_dir / f"video_{seg.id:03d}.mp4")
            return

        if seg.asset_type == AssetType.IMAGE or video_gen is None:
            if self._fallback_to_image() or seg.asset_type == AssetType.IMAGE:
                seg.asset_path = str(image_path)
                if video_gen is None and seg.asset_type != AssetType.IMAGE:
                    log.info(
                        "segment %d: 视频生成器不可用，降级为静图", seg.id,
                    )
                    seg.asset_type = AssetType.IMAGE
            else:
                raise RuntimeError(
                    f"segment {seg.id}: 需要 AI 视频但生成器不可用（fallback_to_image=false）"
                )

        elif seg.asset_type == AssetType.IMAGE2VIDEO:
            if not seg.image_prompt:
                raise DirectorPromptError(
                    f"segment {seg.id}: 缺少 image_prompt，无法派生 video_prompt"
                )
            video_prompt = self._derive_video_prompt_from_image(
                seg.image_prompt, seg.motion
            )
            seg.video_prompt = video_prompt
            clip_dir = run_dir / f"video_{seg.id:03d}"
            result = video_gen.generate(
                prompt=video_prompt,
                image_path=image_path,
                duration=seg.duration_sec,
                output_dir=clip_dir,
            )
            self._apply_video_result(seg, run_dir, result)

        elif seg.asset_type == AssetType.VIDEO:
            if not seg.image_prompt:
                image_prompt = self._visual_to_prompt(seg.visual, seg.id, visual_bible)
                seg.image_prompt = consistency.enrich_image_prompt(
                    image_prompt, seg.visual
                )
            video_prompt = self._derive_video_prompt_from_image(
                seg.image_prompt, seg.motion
            )
            seg.video_prompt = video_prompt
            clip_dir = run_dir / f"video_{seg.id:03d}"
            result = video_gen.generate(
                prompt=video_prompt,
                duration=seg.duration_sec,
                output_dir=clip_dir,
            )
            self._apply_video_result(seg, run_dir, result)

    # ------------------------------------------------------------------
    # 视觉描述 → 英文 Prompt 翻译（专用于导演流水线）
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bible_context(visual_bible) -> str:
        """构建 visual_bible 注入 LLM 的上下文（场景 + 角色 + 风格）。"""
        if not visual_bible:
            return ""
        bible_context = ""
        scene = str(getattr(visual_bible, "scene_anchor", "") or "").strip()
        if scene:
            bible_context += (
                "【全片固定场景锚点 - 翻译时必须保留在同一空间/布景内】\n"
                f"{scene}\n"
                "所有 segment 的画面必须发生在此场景内，不得更换房间或改变关键道具布局。\n\n"
            )
        if visual_bible.characters:
            char_lines = []
            for ch in visual_bible.characters:
                name = ch.get("name", "")
                anchor = ch.get("prompt_anchor", "")
                if name and anchor:
                    char_lines.append(f"- {name} → {anchor}")
            if char_lines:
                bible_context += (
                    "【角色锚点 - 翻译时必须使用以下固定外观描述】\n"
                    + "\n".join(char_lines) + "\n"
                    "如果画面描述中提到以上角色，必须使用对应的英文锚点描述，不能自由发挥。\n\n"
                )
        if visual_bible.style_tags:
            bible_context += f"【全片风格标签（必须附加到末尾）】{visual_bible.style_tags}\n\n"
        return bible_context

    def _llm_required_for_visual(self) -> bool:
        from src.llm.llm_client import is_llm_available

        return is_llm_available(self.config.get("llm", {}))

    def _visual_to_prompt(self, visual: str, seg_id: int, visual_bible=None) -> str:
        """将中文画面描述直接翻译为英文图片生成 prompt。

        如果有 visual_bible，会将场景/角色锚点和风格标签注入 prompt，
        确保全片场景、角色外观和画面风格一致。
        """
        if not visual or not visual.strip():
            style = ""
            if visual_bible and visual_bible.style_tags:
                style = visual_bible.style_tags + ", "
            scene = ""
            if visual_bible and getattr(visual_bible, "scene_anchor", ""):
                scene = visual_bible.scene_anchor.strip() + ", "
            return f"{scene}{style}a cinematic scene, highly detailed, 4K"

        bible_context = self._build_bible_context(visual_bible)

        if not self._llm_required_for_visual():
            return self._visual_to_prompt_local(visual, visual_bible)

        try:
            llm = self._get_llm()
            system = bible_context + _VISUAL_TO_IMAGE_PROMPT if bible_context else _VISUAL_TO_IMAGE_PROMPT
            response = llm.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": visual},
                ],
                temperature=0.5,
            )
            prompt = (response.content or "").strip()
            if prompt:
                return prompt
            raise DirectorPromptError(f"segment {seg_id}: LLM 返回空图片 prompt")
        except DirectorPromptError:
            raise
        except Exception as exc:
            log.error("LLM 视觉翻译失败 seg %d: %s", seg_id, exc)
            raise DirectorPromptError(
                f"segment {seg_id}: LLM 图片 prompt 翻译失败: {exc}"
            ) from exc

    @staticmethod
    def _derive_video_prompt_from_image(image_prompt: str, motion) -> str:
        """从首帧 image_prompt 派生图生视频 prompt（不再单独调 LLM）。"""
        base = (image_prompt or "").strip().rstrip("., ")
        if not base:
            raise DirectorPromptError("无法从空的 image_prompt 派生 video_prompt")
        motion_key = getattr(motion, "value", motion) if motion is not None else "static"
        motion_key = str(motion_key or "static").lower()
        camera = _MOTION_VIDEO_HINTS.get(motion_key, _MOTION_VIDEO_HINTS["static"])
        if _VIDEO_PROMPT_SUFFIX.lower() in base.lower():
            return f"{base}. Camera: {camera}."
        return f"{base}. Camera: {camera}. {_VIDEO_PROMPT_SUFFIX}."

    @staticmethod
    def _visual_to_prompt_local(visual: str, visual_bible=None) -> str:
        """规则翻译兜底：从中文画面描述提取关键词 + 场景/角色锚点注入。"""
        import re
        parts = []

        if visual_bible and getattr(visual_bible, "scene_anchor", ""):
            scene = str(visual_bible.scene_anchor).strip()
            if scene:
                parts.append(scene)

        # 如果有 visual_bible，注入角色锚点（含别名/单主角「猫」匹配）
        if visual_bible and visual_bible.characters:
            from src.scriptplan.character_consistency import build_tracker_from_visual_bible

            tracker, seeded = build_tracker_from_visual_bible(visual_bible)
            chars = tracker.resolve_segment_characters(visual, seeded_names=seeded or None)
            if not chars and len(visual_bible.characters) == 1:
                primary = visual_bible.characters[0]
                if isinstance(primary, dict) and re.search(r"猫|它", visual):
                    name = str(primary.get("name", "")).strip()
                    if name:
                        chars = [name]
            anchor = tracker.get_character_prompt(chars)
            if anchor:
                parts.append(anchor)
            else:
                for ch in visual_bible.characters:
                    name = ch.get("name", "")
                    anchor_text = ch.get("prompt_anchor", "")
                    if name and anchor_text and name in visual:
                        parts.append(anchor_text)

        # 性别检测（仅在未通过角色锚点匹配时）
        if not parts:
            if re.search(r'女人|女性|女孩|少女|女子|姑娘|她', visual):
                parts.append("a young woman")
            elif re.search(r'男人|男性|男孩|少年|男子|他', visual):
                parts.append("a young man")

        # 外观关键词
        appearance_map = [
            (r'西装|正装', 'wearing a suit'),
            (r'黑色', 'black'),
            (r'白色', 'white'),
            (r'红色', 'red'),
            (r'长发', 'long hair'),
            (r'短发', 'short hair'),
            (r'眼镜', 'wearing glasses'),
            (r'帽子', 'wearing a hat'),
        ]
        for pattern, desc in appearance_map:
            if re.search(pattern, visual):
                parts.append(desc)

        # 动作关键词
        action_map = [
            (r'站|站着|站立', 'standing'),
            (r'坐|坐着', 'sitting'),
            (r'跑|奔跑', 'running'),
            (r'走|行走|走路', 'walking'),
            (r'回头|转身', 'turning around'),
            (r'微笑|笑', 'smiling'),
            (r'哭|流泪', 'crying'),
            (r'俯瞰|俯视', 'looking down from above'),
        ]
        for pattern, desc in action_map:
            if re.search(pattern, visual):
                parts.append(desc)

        # 场景关键词
        scene_map = [
            (r'城市|都市|高楼', 'modern city'),
            (r'夜景|夜晚|深夜', 'night scene, city lights'),
            (r'办公室|工位', 'modern office'),
            (r'窗前|落地窗|窗户', 'standing by window'),
            (r'沙发|客厅', 'living room, sofa'),
            (r'厨房|做饭', 'kitchen'),
            (r'卧室|床', 'bedroom'),
            (r'街道|马路', 'city street'),
            (r'森林|树林', 'forest'),
            (r'海边|海滩|大海', 'beach, ocean'),
            (r'太空|宇宙|星空', 'outer space, stars'),
            (r'雨|下雨', 'rain'),
            (r'雪|下雪', 'snow'),
            (r'咖啡|咖啡店', 'coffee shop'),
            (r'医院|病房', 'hospital'),
            (r'学校|教室', 'school, classroom'),
            (r'车|汽车', 'car'),
        ]
        for pattern, desc in scene_map:
            if re.search(pattern, visual):
                parts.append(desc)
        if not parts:
            parts.append("a cinematic scene")

        # 注入全片风格标签
        if visual_bible and visual_bible.style_tags:
            parts.append(visual_bible.style_tags)

        parts.append("highly detailed, cinematic lighting, 4K")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Stage 6: 合成视频
    # ------------------------------------------------------------------

    def _assemble_video(
        self,
        script,
        run_dir: Path,
        output_path: Path,
    ) -> Path:
        """按脚本合成最终视频。"""
        from src.scriptplan.models import AssetType
        from src.video.video_assembler import VideoAssembler

        video_config = self.config.get("video", {
            "resolution": [1080, 1920],
            "fps": 30,
            "codec": "libx265",
        })
        assembler = VideoAssembler(video_config, run_dir)

        images: list[Path] = []
        audio_srt: list[dict] = []
        video_clips: list[Path | None] = []
        has_video_clips = False

        for seg in script.segments:
            if not seg.asset_path or not seg.audio_path:
                log.warning("segment %d 缺少素材或配音，跳过", seg.id)
                continue

            asset_path = Path(seg.asset_path)
            audio_path = Path(seg.audio_path)
            srt_path = Path(seg.srt_path) if seg.srt_path else None

            images.append(asset_path)

            if seg.asset_type in (AssetType.IMAGE2VIDEO, AssetType.VIDEO):
                has_video_clips = True
                video_clips.append(asset_path)
            else:
                video_clips.append(asset_path)  # 静图也传路径，assembler 会用 Ken Burns

            audio_srt.append({
                "audio": audio_path,
                "srt": srt_path,
            })

        if not images:
            raise RuntimeError("没有可用素材，无法合成视频")

        final_path = assembler.assemble(
            images=images,
            audio_srt=audio_srt,
            output_path=output_path,
            video_clips=video_clips if has_video_clips else None,
        )

        log.info("视频合成完成: %s", final_path)
        return final_path
