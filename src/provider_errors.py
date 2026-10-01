"""Pure helpers for classifying provider quota errors."""

from typing import Iterator, Optional, Set


_KNOWN_DAILY_QUOTA_ID = "generaterequestsperdayperprojectpermodel-freetier"
_QUOTA_TEXT_MARKERS = (
    "quota",
    "rate limit",
    "rate-limit",
    "too many requests",
    "resource_exhausted",
    "resource exhausted",
)
_DAILY_TEXT_MARKERS = ("daily", "per day", "per-day", "per_day", "perday")


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its explicit or implicit causes without looping."""
    current: Optional[BaseException] = error
    seen: Set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _has_429_status(error: BaseException) -> bool:
    for item in _exception_chain(error):
        for attribute in ("code", "status_code"):
            value = getattr(item, attribute, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 429:
                return True
    return False


def _error_text(error: BaseException) -> str:
    return " ".join(str(item).lower() for item in _exception_chain(error))


def is_quota_error(error: BaseException) -> bool:
    """Return whether an exception chain describes a quota or rate-limit failure."""
    text = _error_text(error)
    return _has_429_status(error) or any(marker in text for marker in _QUOTA_TEXT_MARKERS)


def is_daily_quota_error(error: BaseException) -> bool:
    """Return whether a quota failure represents a non-recoverable daily limit."""
    if not is_quota_error(error):
        return False
    text = _error_text(error)
    return _KNOWN_DAILY_QUOTA_ID in text or any(
        marker in text for marker in _DAILY_TEXT_MARKERS
    )
