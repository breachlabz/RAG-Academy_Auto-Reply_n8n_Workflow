"""Email payload handling: what arrives over the wire, before any model sees it."""

from .text import html_to_text, strip_quoted, to_plain_text

__all__ = ["html_to_text", "strip_quoted", "to_plain_text"]
