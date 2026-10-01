from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator, ConfigDict

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "taxonomy.yaml"


class TicketClass(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    description: str
    products: List[str] = Field(default_factory=list)
    root_causes: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    status: Literal["active", "held_back"] = "active"


class Taxonomy(BaseModel):
    version: int
    products: List[str]
    classes: List[TicketClass]

    @model_validator(mode="after")
    def _check(self):
        ids = [c.id for c in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate class ids in taxonomy")
        for c in self.classes:
            unknown = set(c.products) - set(self.products)
            if unknown:
                raise ValueError(f"{c.id} references unknown products: {unknown}")
        return self

    def active_classes(self) -> List[TicketClass]:
        return [c for c in self.classes if c.status == "active"]

    def get(self, class_id: str) -> Optional[TicketClass]:
        return next((c for c in self.classes if c.id == class_id), None)


def load_taxonomy(path: Optional[Path] = None) -> Taxonomy:
    data = yaml.safe_load(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
    return Taxonomy(**data)