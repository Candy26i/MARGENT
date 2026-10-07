"""Teacher clients for sub-agent data synthesis (OpenAI, Anthropic, DeepSeek)."""
from .base import TeacherClient, TeacherResponse, build_teacher_client

__all__ = ["TeacherClient", "TeacherResponse", "build_teacher_client"]
