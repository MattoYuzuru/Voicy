def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def chunks(text: str, limit: int = 3500) -> list[str]:
    """Split without losing text or breaking supplementary Unicode characters."""
    result, buf, size = [], [], 0
    for c in text:
        units = 2 if ord(c) > 0xFFFF else 1
        if size + units > limit:
            result.append("".join(buf))
            buf, size = [], 0
        buf.append(c)
        size += units
    if buf:
        result.append("".join(buf))
    return result or ["Речь не обнаружена."]
