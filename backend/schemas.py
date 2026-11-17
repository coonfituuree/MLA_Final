from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from shared.text import FIELDS, MAX_TEXT, clean_text, prepare_text, prepare_full_text

ModelName = Literal['neural_network', 'logistic_regression', 'word_char_logistic', 'context_encoder']

class VacancyInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str = Field(default='', max_length=500)
    company_profile: str = Field(default='', max_length=20000)
    description: str = Field(min_length=1, max_length=60000)
    requirements: str = Field(default='', max_length=20000)
    benefits: str = Field(default='', max_length=20000)
    later_instructions: str = Field(default='', max_length=20000, description='Optional later recruiter instructions; heuristic review only, excluded from learned classifier inputs.')
    model: ModelName = 'neural_network'

    @model_validator(mode='after')
    def validate_text(self):
        if not clean_text(self.description):
            raise ValueError('Description must contain readable text, not only whitespace or HTML.')
        if sum(len(getattr(self, f)) for f in FIELDS) > MAX_TEXT:
            raise ValueError(f'Total input must not exceed {MAX_TEXT:,} characters.')
        if len(prepare_full_text(self.model_dump())) > MAX_TEXT:
            raise ValueError('Prepared text including field separators exceeds 60,000 characters; shorten the input. No content was silently removed.')
        if not prepare_text(self.model_dump()):
            raise ValueError('Vacancy text is empty after cleaning.')
        return self

class PredictionResponse(BaseModel):
    predicted_label: Literal[0, 1]
    label: Literal['legitimate','potentially_fraudulent']
    model_score: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    model_name: ModelName
    model_version: str
    score_note: str
    vocabulary_match: bool

class EvidencePassage(BaseModel):
    field: str
    text: str
    start: int
    end: int
    normalization: str

class ReviewSignal(BaseModel):
    code: str
    title: str
    explanation: str
    evidence: list[EvidencePassage]
    method: Literal['heuristic']
    source_url: str | None = None

class ReviewOutput(BaseModel):
    version: str
    method: Literal['heuristic']
    document_kind_hint: str
    signals: list[ReviewSignal]
    limitations: str
    context_note: str | None = None

class OverallReview(BaseModel):
    status: Literal['additional_review_recommended','no_additional_signal_found']
    reasons: list[str]
    models_disagree: bool
    policy_version: str
    employer_verified: Literal[False]
    confirmed_illegal_activity: Literal[False]

class InputSummary(BaseModel):
    prepared_characters: int
    prepared_sha256: str
    included_fields: list[str]
    preprocessing_version: str
    truncated: Literal[False]
    later_instructions_in_learned_input: Literal[False]

class ScreeningResponse(PredictionResponse):
    comparisons: list[PredictionResponse]
    review: ReviewOutput
    overall_review: OverallReview
    input_summary: InputSummary
    bundle_version: str
