"""Video→Video 角色替换任务数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class V2VSegment(BaseModel):
    """原视频切分后的单段替换任务。"""

    id: int
    start_sec: float
    end_sec: float
    source_clip: str = ""
    audio_clip: str = ""
    anchor_frame: str = ""  # 上一段 output 末帧（seg.id>=2 用于衔接）
    submit_id: str = ""
    output_clip: str = ""
    status: str = "pending"  # pending | submitted | done | failed


class V2VExtension(BaseModel):
    """替换成片末尾的扩演片段（首帧锚定替换视频最后一帧）。"""

    prompt: str = ""
    duration_sec: float = 5.0
    anchor_frame: str = ""
    submit_id: str = ""
    output_clip: str = ""
    status: str = "skipped"  # skipped | pending | submitted | done | failed

    @property
    def enabled(self) -> bool:
        return bool((self.prompt or "").strip())


class V2VReplaceJob(BaseModel):
    """v2v 角色替换任务状态（持久化 job.json）。"""

    run_id: str
    source_video: str
    character_image: str
    prompt: str
    max_segment_sec: float = 15.0
    keep_audio: bool = True
    segment_anchor: bool = False  # 第 2 段起用上一段 output 末帧衔接
    segments: list[V2VSegment] = Field(default_factory=list)
    extension: V2VExtension | None = None
    final_video: str = ""

    def pending_count(self) -> int:
        pending = sum(
            1
            for s in self.segments
            if not (s.output_clip and s.status == "done")
        )
        ext = self.extension
        if ext and ext.enabled:
            if not (ext.status == "done" and ext.output_clip):
                pending += 1
        return pending

    def done_segments(self) -> list[V2VSegment]:
        return [s for s in self.segments if s.output_clip and s.status == "done"]
