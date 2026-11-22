from pydantic import BaseModel, ConfigDict, Field, field_validator

class SolarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")   # "exactly these fields"

    panel_area_m2: float = Field(..., le=100)   # client rule 1
    sunshine_hours: float
    temp_c: float
    dust_days: float = Field(..., ge=0)         # client rule 2
    panel_age_yr: float
    cloud_cover_pct: float
    roof_type: str
    area: str

    @field_validator("roof_type")
    @classmethod
    def clean_roof(cls, v: str) -> str:
        return v.strip().lower()