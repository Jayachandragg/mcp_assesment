import re


REDACTED = "[REDACTED]"

# Most text we'll ever hold back. Emails max out at 254 characters.
MAX_HOLD = 320

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\b"
)

# The two number patterns catch some harmless numbers too. That's on purpose:
# a redacted order ID is better than a leaked card.

# 13+ digits. No upper limit, so a number right before a card can't leave part
# of the card showing. No Luhn check, since it would let mistyped cards through.
CREDIT_CARD_PATTERN = re.compile(
    r"\b[0-9](?:[ -]*[0-9]){12,}\b"
)

SSN_PATTERN = re.compile(
    r"\b[0-9]{3}[ .-]?[0-9]{2}[ .-]?[0-9]{4}\b"
)

EMAIL_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._%+-@"
)
SEPARATORS = frozenset(" .-")
NUMBER_CHARACTERS = frozenset("0123456789") | SEPARATORS


def redact_text(text: str) -> str:
    text = EMAIL_PATTERN.sub(REDACTED, text)

    # Cards first, otherwise an SSN match could split a longer number.
    text = CREDIT_CARD_PATTERN.sub(REDACTED, text)
    text = SSN_PATTERN.sub(REDACTED, text)

    return text


def find_hold_start(text: str) -> int:
    # Work out where the unfinished part starts. Going backwards from the end
    # means we never look at more than MAX_HOLD characters.
    limit = max(0, len(text) - MAX_HOLD)

    # Could be an email that isn't finished yet.
    word_start = len(text)

    while word_start > limit and text[word_start - 1] in EMAIL_CHARACTERS:
        word_start -= 1

    # Could be a card or SSN that isn't finished yet.
    number_start = len(text)

    while number_start > limit and text[number_start - 1] in NUMBER_CHARACTERS:
        number_start -= 1

    # Skip leading separators, so a space after a word isn't held back.
    while number_start < len(text) and text[number_start] in SEPARATORS:
        number_start += 1

    return min(word_start, number_start)


class StreamRedactor:
    """
    Redacts PII from text that arrives in chunks.

    Only the last unfinished token is held back, since that's the only text the
    next chunk could turn into PII. Everything before it goes out right away.
    """

    def __init__(self):
        self.pending = ""

    def feed(self, chunk: str) -> str:
        text = self.pending + chunk
        cut = find_hold_start(text)
        self.pending = text[cut:]

        return redact_text(text[:cut])

    def flush(self) -> str:
        # Stream's over, so whatever is left is final.
        text = redact_text(self.pending)
        self.pending = ""

        return text
