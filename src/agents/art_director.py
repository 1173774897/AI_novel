"""美术指导 Agent - 图片生成 + 质量控制 + 视频片段生成"""
from __future__ import annotations

from collections.abc import Callable

from pathlib import Path

from src.agents.state import AgentState, Decision, QualityEvaluation
from src.agents.utils import make_decision
from src.tools.prompt_gen_tool import PromptGenTool
from src.tools.image_gen_tool import ImageGenTool
from src.tools.evaluate_quality_tool import EvaluateQualityTool
from src.logger import log

_MIN_IMAGE_BYTES = 100


def _existing_image_path(img_dir: Path, index: int) -> Path | None:
    """若该段图片已存在则返回路径，供断点续传跳过生图。"""
    primary = img_dir / f"{index:04d}.png"
    if primary.exists() and primary.stat().st_size > _MIN_IMAGE_BYTES:
        return primary
    for path in sorted(img_dir.glob(f"{index:04d}_r*.png"), reverse=True):
        if path.stat().st_size > _MIN_IMAGE_BYTES:
            return path
    return None


class ArtDirectorAgent:
    MAX_RETRIES = 3
    MODERATION_SOFTEN_ATTEMPTS = 3
    MODERATION_REGEN_ATTEMPTS = 3
    QUALITY_THRESHOLD = 6.0

    def __init__(self, config: dict, budget_mode: bool = False):
        self.config = config
        self.budget_mode = budget_mode
        self.prompt_gen = PromptGenTool(config)
        self.image_gen = ImageGenTool(config)
        self.quality_tool = EvaluateQualityTool(config)
        self._video_gen = None  # 懒加载

    @property
    def video_gen(self):
        if self._video_gen is None:
            from src.tools.video_gen_tool import VideoGenTool
            self._video_gen = VideoGenTool(self.config)
        return self._video_gen

    @staticmethod
    def _write_fallback_image(out_path: Path) -> None:
        """即梦多次拒稿时写入暗色占位图，避免整段流水线中断。"""
        from PIL import Image

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1920, 1080), (18, 20, 28)).save(out_path)

    def _run_image_gen_with_moderation_fallback(
        self,
        prompt: str,
        out_path: Path,
        index: int,
        decisions: list[Decision],
        *,
        regen_prompt: Callable[[int], str] | None = None,
        person_count: int | None = None,
    ) -> None:
        """生图；不合规时先软化 3 次，再换角度重生 prompt 3 次。"""
        from src.imagegen.dashscope_backend import ContentModerationError
        from src.imagegen.jimeng_cli_backend import JimengGenerationError
        from src.imagegen.moderation import (
            _MODERATION_REGEN_ATTEMPTS,
            _MODERATION_SOFTEN_ATTEMPTS,
            is_jimeng_retryable_error,
            minimal_safe_fallback_prompt,
            soften_image_prompt_for_attempt,
            truncate_image_prompt_for_jimeng,
        )

        last_detail = ""
        attempts_used = 0

        def _try_generate(current: str, *, phase: str, step: int) -> bool:
            nonlocal last_detail, attempts_used
            try:
                self.image_gen.run(current, out_path, person_count=person_count)
            except ContentModerationError:
                last_detail = "content_moderation"
            except JimengGenerationError as exc:
                last_detail = exc.reason
            except RuntimeError as exc:
                last_detail = str(exc)
            else:
                if attempts_used > 0:
                    decisions.append(make_decision(
                        "ArtDirector",
                        f"moderation_fallback_seg{index}",
                        f"段{index} 生图被拒后重试成功 ({phase} #{step + 1})",
                        f"prompt: {current[:80]}...",
                    ))
                return True

            attempts_used += 1
            if not is_jimeng_retryable_error(last_detail):
                return False
            log.warning(
                "[ArtDirector] 段%d 生图不合规 (%s)，%s 重试 (第 %d 次)",
                index,
                last_detail[:80],
                phase,
                step + 1,
            )
            return False

        base_prompt = prompt

        # 阶段 1：原 prompt → 逐次软化（共 3 次）
        for soften_i in range(_MODERATION_SOFTEN_ATTEMPTS):
            if soften_i == 0:
                current = truncate_image_prompt_for_jimeng(base_prompt)
            else:
                current = soften_image_prompt_for_attempt(base_prompt, soften_i - 1)
            if _try_generate(current, phase="软化", step=soften_i):
                return
            if not is_jimeng_retryable_error(last_detail):
                break

        # 阶段 2：换角度重生 prompt（共 3 次）
        if regen_prompt is not None:
            for regen_i in range(_MODERATION_REGEN_ATTEMPTS):
                if not is_jimeng_retryable_error(last_detail):
                    break
                try:
                    base_prompt = regen_prompt(regen_i)
                except Exception as exc:
                    log.warning(
                        "[ArtDirector] 段%d 换角度 prompt 生成失败: %s",
                        index,
                        exc,
                    )
                    continue
                current = truncate_image_prompt_for_jimeng(base_prompt)
                if _try_generate(current, phase="换角度", step=regen_i):
                    return

        # 阶段 3：6 次均失败后，通用空镜保底生图（仍失败才写占位图）
        if is_jimeng_retryable_error(last_detail):
            current = minimal_safe_fallback_prompt(0)
            if _try_generate(current, phase="空镜保底", step=0):
                return

        total = _MODERATION_SOFTEN_ATTEMPTS + (
            _MODERATION_REGEN_ATTEMPTS if regen_prompt else 0
        ) + 1
        log.error(
            "[ArtDirector] 段%d 生图 %d 次均不合规 (%s)，写入占位图继续流水线",
            index,
            total,
            (last_detail or "unknown")[:120],
        )
        self._write_fallback_image(out_path)
        decisions.append(make_decision(
            "ArtDirector",
            f"moderation_placeholder_seg{index}",
            f"段{index} 拒稿，已用占位图跳过",
            f"detail={last_detail[:120]}",
        ))

    def _optimize_prompt(
        self,
        original_prompt: str,
        feedback: str,
        evaluation: QualityEvaluation,
    ) -> str:
        """根据质量反馈优化 prompt。"""
        additions = []
        if evaluation.get("clarity", 0) < 1.5:
            additions.append("sharp focus, high detail, 8k resolution")
        if evaluation.get("composition", 0) < 1.5:
            additions.append("well-composed, rule of thirds, balanced layout")
        if evaluation.get("color", 0) < 1.5:
            additions.append("vibrant colors, harmonious color palette")
        if evaluation.get("text_match", 0) < 2.0:
            additions.append("accurate depiction of the scene")

        if additions:
            return f"{original_prompt}, {', '.join(additions)}"
        return original_prompt

    def generate_image(
        self,
        text: str,
        index: int,
        workspace: Path,
        full_text: str | None = None,
        prev_text: str | None = None,
    ) -> tuple[Path, float, int, list[Decision]]:
        """生成图片，可选质量控制。返回 (path, score, retries, decisions)"""
        img_dir = Path(workspace) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        decisions: list[Decision] = []

        retry_count = 0
        best_path: Path | None = None
        best_score = 0.0

        threshold = (
            self.config.get("agent", {})
            .get("quality_check", {})
            .get("threshold", self.QUALITY_THRESHOLD)
        )
        max_retries = (
            self.config.get("agent", {})
            .get("quality_check", {})
            .get("max_retries", self.MAX_RETRIES)
        )
        quality_enabled = not self.budget_mode and self.config.get("agent", {}).get(
            "quality_check", {}
        ).get("enabled", False)

        last_evaluation: QualityEvaluation | None = None

        while retry_count <= max_retries:
            # 生成 prompt
            prompt = self.prompt_gen.run(
                text,
                segment_index=index,
                full_text=full_text,
                prev_text=prev_text,
            )

            # 重试时根据上次评估反馈优化 prompt
            if retry_count > 0 and last_evaluation is not None:
                feedback = last_evaluation.get("feedback", "")
                prompt = self._optimize_prompt(prompt, feedback, last_evaluation)
                log.info(
                    "[ArtDirector] 段%d 重试优化 prompt: %s",
                    index,
                    prompt[:100],
                )

            # 生成图片（内容审核：软化 3 次 → 换角度 prompt 3 次）
            suffix = f"_r{retry_count}" if retry_count > 0 else ""
            out_path = img_dir / f"{index:04d}{suffix}.png"

            def _regen_prompt(variant: int) -> str:
                return self.prompt_gen.run_alternate(
                    text,
                    segment_index=index,
                    full_text=full_text,
                    prev_text=prev_text,
                    variant=variant,
                )

            person_count = self.prompt_gen.count_characters(text)

            self._run_image_gen_with_moderation_fallback(
                prompt,
                out_path,
                index,
                decisions,
                regen_prompt=_regen_prompt,
                person_count=person_count,
            )

            if not quality_enabled:
                decisions.append(make_decision(
                    "ArtDirector",
                    f"image_seg{index}",
                    f"生成图片（{'省钱模式' if self.budget_mode else '质量检查关闭'}）",
                    f"prompt: {prompt[:80]}...",
                ))
                return out_path, -1.0, 0, decisions

            # 质量评估（使用独立工具）
            evaluation = self.quality_tool.run(out_path, text, prompt)
            score = evaluation.get("score", 5.0)
            feedback = evaluation.get("feedback", "")
            last_evaluation = evaluation

            decisions.append(make_decision(
                "ArtDirector",
                f"quality_seg{index}_try{retry_count}",
                f"评分={score:.1f}/10, {'通过' if score >= threshold else '未通过'}",
                f"反馈: {feedback}",
                data={
                    "score": score,
                    "feedback": feedback,
                    "composition": evaluation.get("composition", 0),
                    "clarity": evaluation.get("clarity", 0),
                    "text_match": evaluation.get("text_match", 0),
                    "color": evaluation.get("color", 0),
                    "consistency": evaluation.get("consistency", 0),
                },
            ))

            if score > best_score:
                best_path = out_path
                best_score = score

            if score >= threshold:
                return out_path, score, retry_count, decisions

            if retry_count >= max_retries:
                decisions.append(make_decision(
                    "ArtDirector",
                    f"retry_limit_seg{index}",
                    f"达到重试上限，使用最佳结果（评分={best_score:.1f}）",
                    "警告：质量未达标",
                ))
                return best_path, best_score, retry_count, decisions  # type: ignore[return-value]

            retry_count += 1
            log.info(
                "[ArtDirector] 段%d 评分%.1f < %.1f，重试第%d次",
                index,
                score,
                threshold,
                retry_count,
            )

        return best_path, best_score, retry_count, decisions  # type: ignore[return-value]

    def generate_video_clip(
        self,
        text: str,
        index: int,
        workspace: Path,
        image_path: Path | None = None,
    ) -> tuple[Path, list[Decision]]:
        """生成 AI 视频片段。返回 (path, decisions)"""
        clip_dir = Path(workspace) / "video_clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        decisions: list[Decision] = []

        # 生成视频专用 prompt
        video_prompt = self.prompt_gen.run_video_prompt(text, segment_index=index)

        # 是否使用图片作为首帧
        use_first_frame = self.config.get("videogen", {}).get(
            "use_image_as_first_frame", True
        )
        first_frame = image_path if use_first_frame else None

        out_path = clip_dir / f"{index:04d}.mp4"
        self.video_gen.run(video_prompt, out_path, image_path=first_frame)

        decisions.append(make_decision(
            "ArtDirector",
            f"video_seg{index}",
            f"视频片段生成完成",
            f"prompt: {video_prompt[:80]}...",
            data={"image_as_first_frame": first_frame is not None},
        ))

        return out_path, decisions


def art_director_node(state: AgentState) -> dict:
    """ArtDirector 节点"""
    config = state["config"]
    budget_mode = state.get("budget_mode", False)
    workspace = state["workspace"]
    agent = ArtDirectorAgent(config, budget_mode)

    segments = state["segments"]
    images: list[str] = []
    quality_scores: list[float] = []
    retry_counts: dict[int, int] = {}
    decisions: list[Decision] = []

    decisions.append(make_decision(
        "ArtDirector", "start",
        f"开始生成 {len(segments)} 张图片",
        f"风格={config.get('promptgen', {}).get('style') or state.get('suggested_style', 'default')}",
    ))

    suggested_style = state.get("suggested_style")
    config_style = config.get("promptgen", {}).get("style")
    style = config_style or suggested_style
    if style:
        agent.prompt_gen.set_style(style)

    pov_narrator = state.get("pov_narrator")
    if pov_narrator:
        agent.prompt_gen.set_pov_narrator(pov_narrator)

    era_override = state.get("era_override") or config.get("promptgen", {}).get("era")
    if era_override:
        agent.prompt_gen.set_era(era_override)

    registry_path = state.get("series_registry_path")
    if registry_path:
        from src.promptgen.character_registry import CharacterRegistry
        from src.promptgen.era_context import CLASSICAL, normalize_era, sanitize_classical_desc

        registry = CharacterRegistry.load(Path(registry_path))
        canonical = registry.to_seed_list()
        if era_override and normalize_era(era_override) == CLASSICAL:
            canonical = [
                {**entry, "desc": sanitize_classical_desc(entry.get("desc", ""))}
                for entry in canonical
            ]
        if canonical:
            seeded = agent.prompt_gen.seed_characters(canonical, canonical=True)
            if seeded:
                decisions.append(make_decision(
                    "ArtDirector", "series_registry_seed",
                    f"系列注册表预填 {seeded} 个角色外观",
                    str(registry_path),
                ))

    characters = state.get("characters") or []
    if characters:
        seeded = agent.prompt_gen.seed_characters(characters)
        if seeded:
            decisions.append(make_decision(
                "ArtDirector", "seed_characters",
                f"预填 {seeded} 个角色外观描述",
                f"来源=ContentAnalyzer, 角色={[c.get('name') for c in characters if c.get('desc')]}",
            ))
            log.info("[ArtDirector] 从 ContentAnalyzer 预填 %d 个角色描述", seeded)

    full_text = state.get("full_text")
    img_dir = Path(workspace) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(segments):
        existing = _existing_image_path(img_dir, i)
        if existing is not None:
            images.append(str(existing))
            quality_scores.append(-1.0)
            log.info(
                "[ArtDirector] 段 %d/%d 跳过（已有图片）",
                i + 1,
                len(segments),
            )
            continue

        prev_text = segments[i - 1]["text"] if i > 0 else None
        path, score, retries, seg_decisions = agent.generate_image(
            seg["text"],
            i,
            Path(workspace),
            full_text=full_text,
            prev_text=prev_text,
        )
        images.append(str(path))
        quality_scores.append(score)
        decisions.extend(seg_decisions)
        if retries > 0:
            retry_counts[i] = retries

        log.info(
            "[ArtDirector] 段 %d/%d 完成 (评分=%.1f, 重试=%d)",
            i + 1,
            len(segments),
            score,
            retries,
        )

    # 汇总
    valid_scores = [s for s in quality_scores if s >= 0]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else -1

    if registry_path:
        from src.promptgen.character_registry import CharacterRegistry

        tracker = agent.prompt_gen._get_gen().character_tracker
        if tracker is not None:
            registry = CharacterRegistry.load(Path(registry_path))
            merged = registry.merge_tracker(
                tracker, episode=state.get("episode_id")
            )
            registry.save()
            if merged:
                decisions.append(make_decision(
                    "ArtDirector", "series_registry_update",
                    f"回写系列角色表 +{merged} 条",
                    str(registry_path),
                ))

    decisions.append(make_decision(
        "ArtDirector", "summary",
        f"图片生成完成：平均质量={avg_score:.1f}, 总重试={sum(retry_counts.values())}",
        f"{len(images)} 张图片",
    ))

    result: dict = {
        "images": images,
        "quality_scores": quality_scores,
        "retry_counts": retry_counts,
        "decisions": decisions,
    }

    # --- 视频片段生成（可选） ---
    video_enabled = (state.get("pipeline_plan") or {}).get("video_enabled", False)
    if video_enabled:
        video_clips: list[str] = []
        decisions.append(make_decision(
            "ArtDirector", "video_start",
            f"开始生成 {len(segments)} 个视频片段",
            f"backend={config.get('videogen', {}).get('backend', '?')}",
        ))

        for i, seg in enumerate(segments):
            image_path = Path(images[i]) if i < len(images) else None
            clip_path, clip_decisions = agent.generate_video_clip(
                seg["text"], i, Path(workspace), image_path=image_path
            )
            video_clips.append(str(clip_path))
            decisions.extend(clip_decisions)
            log.info(
                "[ArtDirector] 视频片段 %d/%d 完成",
                i + 1, len(segments),
            )

        decisions.append(make_decision(
            "ArtDirector", "video_summary",
            f"视频片段生成完成：{len(video_clips)} 个",
            "全部完成",
        ))
        result["video_clips"] = video_clips

        # 释放视频生成资源
        agent.video_gen.close()

    return result
