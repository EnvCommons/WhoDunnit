from __future__ import annotations

import json
from typing import Optional

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool
from pydantic import BaseModel, Field


# --- Pydantic Models for Task Data ---

class Suspect(BaseModel):
    name: str
    description: str
    height: str
    handedness: str
    eye_color: str
    hair_color: str


class Location(BaseModel):
    name: str
    indoor: bool
    description: str


class Weapon(BaseModel):
    name: str
    weight: str
    description: str


class Statement(BaseModel):
    suspect: str
    statement: str


class ExhibitItem(BaseModel):
    name: str
    description: str


class Exhibit(BaseModel):
    exhibit_id: str
    title: str
    items: list[ExhibitItem]
    description: Optional[str] = None


class GroundTruth(BaseModel):
    who: str
    what: str
    where: str
    why: Optional[str] = None


class WhodunnitTaskSpec(BaseModel):
    problem_name: str
    problem_description: str
    suspects: list[Suspect]
    locations: list[Location]
    weapons: list[Weapon]
    clues: list[str]
    motives: Optional[list[str]] = None
    statements: Optional[list[Statement]] = None
    ground_truth: GroundTruth


# --- Module-level data loading (AIME pattern) ---
import os
import glob
from pathlib import Path

if os.path.exists("/orwd_data"):
    ENV_PATH = Path("/orwd_data")
else:
    ENV_PATH = Path(__file__).parent

# Load all tasks_*.json files
TASKS_RAW: list[dict] = []
tasks_files = sorted(glob.glob(str(ENV_PATH / "tasks_*.json")))
for tasks_file in tasks_files:
    with open(tasks_file, "r") as f:
        tasks_data = json.load(f)
        TASKS_RAW.extend(tasks_data)

# Preprocess tasks for list_tasks (only include minimal task_spec)
TASKS_SPEC = [
    {"task_id": i, "problem_name": task["problem_name"]}
    for i, task in enumerate(TASKS_RAW)
]

# Load exhibits data
with open(ENV_PATH / "exhibits.json", "r") as f:
    EXHIBITS_RAW: list[dict] = json.load(f)


# --- Tool Input Models ---

class EmptyParams(BaseModel):
    """No parameters needed for list-type tools"""
    pass


class SubmitAnswerParams(BaseModel):
    who: str = Field(..., description="Name of the suspect who committed the murder")
    what: str = Field(..., description="The murder weapon used")
    where: str = Field(..., description="The location where the murder occurred")
    why: Optional[str] = Field(None, description="The motive for the murder (optional)")


# --- Environment Class ---

class Whodunnit(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        # Load full task data
        task_id = task_spec["task_id"]
        self.task_data = WhodunnitTaskSpec.model_validate(TASKS_RAW[task_id])
        self.task_id = task_id

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            return TASKS_SPEC
        raise ValueError(f"Unknown split: {split}")

    def get_prompt(self) -> list[TextBlock]:
        # Return only the problem description (agent must use tools to discover details)
        return [TextBlock(type="text", text=self.task_data.problem_description)]

    # --- Information Retrieval Tools ---

    @tool
    async def list_suspects(self, params: EmptyParams) -> ToolOutput:
        """List all suspects in the case with their descriptions and physical features."""

        suspects_text = "SUSPECTS\n\n"
        for suspect in self.task_data.suspects:
            suspects_text += f"{suspect.name}\n"
            suspects_text += f"{suspect.description}\n"
            suspects_text += f"{suspect.height} • {suspect.handedness.upper()} • {suspect.eye_color.upper()} EYES • {suspect.hair_color.upper()} HAIR\n\n"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=suspects_text.strip())],
            metadata={"suspects": [s.model_dump() for s in self.task_data.suspects]},
            reward=0.0,
            finished=False,
        )

    @tool
    async def list_locations(self, params: EmptyParams) -> ToolOutput:
        """List all locations where the murder could have occurred."""

        locations_text = "LOCATIONS\n\n"
        for location in self.task_data.locations:
            indoor_status = "INDOORS" if location.indoor else "OUTDOORS"
            locations_text += f"{location.name}\n{indoor_status}\n{location.description}\n\n"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=locations_text.strip())],
            metadata={"locations": [loc.model_dump() for loc in self.task_data.locations]},
            reward=0.0,
            finished=False,
        )

    @tool
    async def list_weapons(self, params: EmptyParams) -> ToolOutput:
        """List all potential murder weapons found at the scene."""

        weapons_text = "WEAPONS\n\n"
        for weapon in self.task_data.weapons:
            weapons_text += f"{weapon.name}\n{weapon.weight.upper()}\n{weapon.description}\n\n"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=weapons_text.strip())],
            metadata={"weapons": [w.model_dump() for w in self.task_data.weapons]},
            reward=0.0,
            finished=False,
        )

    @tool
    async def list_clues(self, params: EmptyParams) -> ToolOutput:
        """List all clues and evidence found at the crime scene."""

        clues_text = "CLUES & EVIDENCE\n\n"
        for clue in self.task_data.clues:
            clues_text += f"• {clue}\n"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=clues_text.strip())],
            metadata={"clues": self.task_data.clues},
            reward=0.0,
            finished=False,
        )

    @tool
    async def list_motives(self, params: EmptyParams) -> ToolOutput:
        """List potential motives for the suspects (if available)."""

        if not self.task_data.motives:
            return ToolOutput(
                blocks=[
                    TextBlock(
                        type="text",
                        text="No motive information is available for this case.",
                    )
                ],
                metadata={"motives": []},
                reward=0.0,
                finished=False,
            )

        motives_text = "MOTIVES\n\n"
        for i, motive in enumerate(self.task_data.motives, 1):
            motives_text += f"{i}. {motive}\n"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=motives_text.strip())],
            metadata={"motives": self.task_data.motives},
            reward=0.0,
            finished=False,
        )

    @tool
    async def list_statements(self, params: EmptyParams) -> ToolOutput:
        """List statements from suspects (if available). Note: The murderer is lying, and the others are telling the truth."""

        if not self.task_data.statements:
            return ToolOutput(
                blocks=[
                    TextBlock(
                        type="text",
                        text="No suspect statements are available for this case.",
                    )
                ],
                metadata={"statements": []},
                reward=0.0,
                finished=False,
            )

        statements_text = "SUSPECT STATEMENTS\n"
        statements_text += "(Note: The murderer is lying, and the others are telling the truth.)\n\n"
        for stmt in self.task_data.statements:
            statements_text += f"{stmt.suspect}: \"{stmt.statement}\"\n"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=statements_text.strip())],
            metadata={"statements": [s.model_dump() for s in self.task_data.statements]},
            reward=0.0,
            finished=False,
        )

    @tool
    async def view_exhibits(self, params: EmptyParams) -> ToolOutput:
        """View exhibits referenced in the clues. These provide important context for solving the case."""

        if not EXHIBITS_RAW:
            return ToolOutput(
                blocks=[
                    TextBlock(
                        type="text",
                        text="No exhibits are available for this case.",
                    )
                ],
                metadata={"exhibits": []},
                reward=0.0,
                finished=False,
            )

        exhibits_text = "EXHIBITS\n\n"
        for exhibit_data in EXHIBITS_RAW:
            exhibit = Exhibit.model_validate(exhibit_data)
            exhibits_text += f"EXHIBIT {exhibit.exhibit_id}\n"
            exhibits_text += f"{exhibit.title}\n\n"

            # If exhibit has a description (for exhibits with no items), display it
            if exhibit.description and not exhibit.items:
                exhibits_text += f"{exhibit.description}\n\n"
            else:
                # Otherwise, display items as before
                for item in exhibit.items:
                    exhibits_text += f"• {item.name}\n"
                    exhibits_text += f"  {item.description}\n\n"

            exhibits_text += "\n"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=exhibits_text.strip())],
            metadata={"exhibits": EXHIBITS_RAW},
            reward=0.0,
            finished=False,
        )

    # --- Submit Answer Tool ---

    @tool
    async def submit_answer(self, params: SubmitAnswerParams) -> ToolOutput:
        """
        Submit your final answer for who committed the murder, with what weapon, where, and optionally why.
        This will finish the episode and provide your final score.
        """

        ground_truth = self.task_data.ground_truth

        # Normalize answers for comparison (case-insensitive, strip whitespace)
        def normalize(s: Optional[str]) -> str:
            if s is None:
                return ""
            return s.strip().lower()

        who_correct = normalize(params.who) == normalize(ground_truth.who)
        what_correct = normalize(params.what) == normalize(ground_truth.what)
        where_correct = normalize(params.where) == normalize(ground_truth.where)

        # Check why only if ground truth has a motive
        why_provided = params.why is not None
        why_expected = ground_truth.why is not None

        # Partial reward calculation
        if why_expected:
            # 4 components: who, what, where, why (0.25 each)
            why_correct = normalize(params.why) == normalize(ground_truth.why)
            reward = (
                (0.25 if who_correct else 0.0)
                + (0.25 if what_correct else 0.0)
                + (0.25 if where_correct else 0.0)
                + (0.25 if why_correct else 0.0)
            )

            # Build feedback
            feedback = []
            feedback.append(
                f"WHO? {'✓' if who_correct else '✗'} (Expected: {ground_truth.who})"
            )
            feedback.append(
                f"WHAT? {'✓' if what_correct else '✗'} (Expected: {ground_truth.what})"
            )
            feedback.append(
                f"WHERE? {'✓' if where_correct else '✗'} (Expected: {ground_truth.where})"
            )
            feedback.append(
                f"WHY? {'✓' if why_correct else '✗'} (Expected: {ground_truth.why})"
            )

        else:
            # 3 components: who, what, where (0.33 each, rounded)
            reward = (
                (1 / 3 if who_correct else 0.0)
                + (1 / 3 if what_correct else 0.0)
                + (1 / 3 if where_correct else 0.0)
            )

            # Build feedback
            feedback = []
            feedback.append(
                f"WHO? {'✓' if who_correct else '✗'} (Expected: {ground_truth.who})"
            )
            feedback.append(
                f"WHAT? {'✓' if what_correct else '✗'} (Expected: {ground_truth.what})"
            )
            feedback.append(
                f"WHERE? {'✓' if where_correct else '✗'} (Expected: {ground_truth.where})"
            )

            if why_provided:
                feedback.append("WHY? (not required for this task)")

        # Format final message
        if reward == 1.0:
            message = "🎉 Perfect! You solved the case completely.\n\n"
        elif reward > 0:
            message = f"Partial credit: {reward:.0%} correct.\n\n"
        else:
            message = "❌ Incorrect. You did not solve the case.\n\n"

        message += "\n".join(feedback)

        return ToolOutput(
            blocks=[TextBlock(type="text", text=message)],
            metadata={
                "submitted": {
                    "who": params.who,
                    "what": params.what,
                    "where": params.where,
                    "why": params.why,
                },
                "ground_truth": ground_truth.model_dump(),
                "components": {
                    "who_correct": who_correct,
                    "what_correct": what_correct,
                    "where_correct": where_correct,
                    "why_correct": why_correct if why_expected else None,
                },
                "reward": reward,
            },
            reward=reward,
            finished=True,
        )
