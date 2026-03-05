# WhoDunit

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://www.openreward.ai/GeneralReasoning/WhoDunit)

## Description

WhoDunit is an ORS environment for evaluating deductive reasoning on murder mystery puzzles. It contains 100 cases where agents must gather information about suspects, weapons, locations, and clues to deduce who committed the murder, with what weapon, and where it occurred.

## Capabilities

- Deductive reasoning and logical inference
- Multi-step information gathering
- Evidence synthesis and analysis
- Murder mystery puzzle solving

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## Tasks

There is one split in this environment:

- **train**: 100 tasks (75 elementary + 25 impossible difficulty)

Each case includes suspects with physical descriptions, potential weapons, locations, clues, and optionally motives and suspect statements.

## Reward Structure

This is a multi-turn environment with partial credit scoring. Agents gather information using tools, then submit their answer via `submit_answer`.

**3-Component Tasks**: WHO (33.3%) + WHAT (33.3%) + WHERE (33.3%)

**4-Component Tasks**: WHO (25%) + WHAT (25%) + WHERE (25%) + WHY (25%)

Validation is deterministic case-insensitive exact match. Reward ranges from 0.0 to 1.0 based on correct components.

## Data

Data consists of JSON files (`tasks_elementary.json`, `tasks_impossible.json`, `exhibits.json`) containing murder mystery cases with suspects, weapons, locations, clues, and ground truth answers. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `list_suspects` | View all suspects with physical descriptions and features. |
| `list_weapons` | View all potential murder weapons with weight classifications. |
| `list_locations` | View all locations where the murder could have occurred. |
| `list_clues` | View all clues and evidence found at the crime scene. |
| `list_motives` | View potential motives (if available for the case). |
| `list_statements` | View suspect statements (murderer lies, others tell truth). |
| `view_exhibits` | View exhibits referenced in clues. |
| `submit_answer` | Submit who, what, where (and optionally why). Ends the episode. |

## Time Horizon

Multi-turn. Agents gather information using multiple tool calls before submitting their final answer.

## Environment Difficulty

[Put environment difficulty here]

## Other Environment Requirements

There are no further environment requirements; WhoDunit works out of the box with the OpenReward endpoint without any external API keys.

## Safety

Agents in WhoDunit solve fictional murder mystery puzzles in a standard environment. The environment does not present direct safety risks.

