from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


# Not \d, because that also matches digits from other scripts.
CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"

CustomerId = Annotated[
    str,
    Field(
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer ID in the format CUST-12345 (CUST- plus 5 digits).",
    ),
]

# The pattern requires one non-space character, so a blank reason is rejected.
RefundReason = Annotated[
    str,
    Field(
        min_length=10,
        pattern=r"\S",
        description="Why the refund is issued. At least 10 characters.",
    ),
]


class GetCustomerRecordArgs(BaseModel):
    # strict turns off type coercion, so "50" doesn't become 50.
    model_config = ConfigDict(strict=True, extra="forbid")

    customer_id: CustomerId


class TriggerRefundArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    customer_id: CustomerId

    # 50 is fine. "50", true, NaN and Infinity are not.
    amount: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Refund amount. Must be a positive, finite number.",
    )

    reason: RefundReason
