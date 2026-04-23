"""Output schema for medication extraction.

Single source of truth: data prep, eval, and constrained decoding all import
the models and JSON schema from here. Changing the schema means changing it
once and letting the rest of the pipeline pick it up automatically.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Medication(BaseModel):
    drug: str = Field(..., description="Drug name as it appears in the text (surface form).")
    dose: Optional[str] = Field(None, description="Amount + unit, e.g. '25 mg', '10 mL'.")
    route: Optional[str] = Field(None, description="Administration route, e.g. 'PO', 'IV', 'by mouth'.")
    frequency: Optional[str] = Field(None, description="Cadence, e.g. 'BID', 'daily', 'every 8 hours'.")
    duration: Optional[str] = Field(None, description="How long, e.g. '2 weeks', 'for 10 days'.")


class MedicationList(BaseModel):
    medications: List[Medication] = Field(default_factory=list)


JSON_SCHEMA = MedicationList.model_json_schema()


SYSTEM_PROMPT = (
    "You are a clinical information extractor. Given a clinical sentence or short "
    "paragraph, extract all medications mentioned and output strict JSON matching "
    "this schema:\n\n"
    '{"medications": [{"drug": str, "dose": str|null, "route": str|null, '
    '"frequency": str|null, "duration": str|null}]}\n\n'
    "Rules:\n"
    "- Output ONLY the JSON object. No prose, no markdown fences.\n"
    "- Use the exact surface form from the text for each field.\n"
    "- Use null (not empty string) for missing fields.\n"
    "- If no medications are mentioned, return {\"medications\": []}."
)
