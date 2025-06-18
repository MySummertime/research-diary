# Projects/Risk_Assessment_Models/backend/api/models.py

from pydantic import BaseModel
from typing import List, Optional


class SolverRequest(BaseModel):
    src: int
    dest: int
    alpha: float
    selected_file: Optional[str] = None


class SolverResponse(BaseModel):
    status: str
    alpha: Optional[float] = None
    optimal_var: Optional[float] = None
    f_value: Optional[float] = None
    alpha_bar: Optional[float] = None
    path: Optional[List[int]] = None
    message: Optional[str] = None
