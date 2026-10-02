from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ParsedComplaint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str
    category_confidence: float = Field(default=0.5, ge=0, le=1)
    unknown_label: Optional[str] = None
    product: Optional[str] = None
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    sentiment: Literal["positive", "neutral", "frustrated", "angry"] = "neutral"
    symptoms: List[str] = Field(default_factory=list)
    steps_already_tried: List[str] = Field(default_factory=list)
    customer_impact: Optional[str] = None

    @field_validator("severity", "sentiment", mode="before")
    @classmethod
    def _lowercase(cls, v):
        return v.strip().lower() if isinstance(v, str) else v