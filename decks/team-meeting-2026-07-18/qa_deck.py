#!/usr/bin/env python3
"""Score every rendered ADAMA deck slide against source truth and presentation quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from google import genai
from google.genai import types

MODEL = "models/gemini-3.5-flash"
AUDIENCE = (
    "ADAMA campers preparing for Burning Man 2026. They need to leave excited and calm, "
    "understand the journey and operating model, surface questions, and commit to submitting "
    "one preference form within 48 hours. They reward warmth, clarity, visible constraints, "
    "and honest boundaries. They do not need corporate polish or exhaustive operating detail."
)

INTENTS = {
    1: ("Open the meeting and establish the 42-day horizon.", "Gathered and expectant."),
    2: ("Explain that the collective enables individual journeys.", "Personally included."),
    3: ("Set participation, choice, and return-to-home as the camp posture.", "Free and responsible."),
    4: ("Orient everyone to ADAMA inside Entheos without false spatial precision.", "Grounded."),
    5: ("Show that the approved Big C layout fits six RVs and two Shiftpods.", "Reassured."),
    6: ("Make the travel sequence and the two Saturday paths understandable.", "Oriented and calm."),
    7: ("Explain Building, Running, and Striking as the structure that creates freedom.", "Clear."),
    8: ("Recruit Building interest and state the locked minimum.", "Invited to step up."),
    9: ("Introduce Running role families without inventing mechanics.", "Able to imagine contributing."),
    10: ("Make universal Fire Shift participation emotionally meaningful and operationally honest.", "Connected and accountable."),
    11: ("Explain Striking as closing the circle and leaving no trace.", "Responsible."),
    12: ("Define the reliability standard for every responsibility.", "Confident the system can work."),
    13: ("Preview the form so campers can answer honestly.", "Prepared, not surprised."),
    14: ("Show how responses become a published operating plan.", "Trusting the process."),
    15: ("Create visible room for questions before asking for action.", "Safe to raise what matters."),
    16: ("Make the form and exact deadline unmistakable.", "Ready to scan and act."),
}

RUBRIC = """Act as an expert presentation designer and executive copyeditor reviewing ONE slide.

AUDIENCE: {audience}
THIS SLIDE'S JOB: {job}
WE WANT THE AUDIENCE TO FEEL: {feel}

You are given the locked source contract and the rendered slide image.
Score each axis from 1 to 10:
1. content_accuracy: factual correctness against the sources, no unsupported finality, no typos or timeline errors.
2. visual_hierarchy: focal flow, layout, balance, and spacing.
3. typography: presentation-scale readability and typographic consistency.
4. color_contrast: accessible legibility and brand-appropriate contrast.
5. narrative_fit: whether this slide does its intended job for this audience.

Critical flags are limited to factual errors, content outside the source, contrast failures that break readability,
content clipping or overflow, broken images, and broken layouts. Do not call a merely optional polish idea critical.
Give only ranked, high-leverage recommendations. If a recommendation needs facts absent from the source, set needs_input=true.

Return STRICT JSON only:
{{"scores":{{"content_accuracy":N,"visual_hierarchy":N,"typography":N,"color_contrast":N,"narrative_fit":N}},
"mean":N.N,"critical_flags":[],
"high_leverage_recs":[{{"change":"specific change","axis":"axis","est_gain":"+N","needs_input":false}}],
"single_biggest_win":"specific change"}}
"""


def read_key(env_path: Path) -> str:
    target = "GEMINI_API_KEY"
    for line in env_path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == target and value.strip():
            return value.strip().strip('"').strip("'")
    raise RuntimeError(f"{target} not found in {env_path}")


def parse_json(text: str) -> dict:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("```", 2)[1]
        if value.lstrip().startswith("json"):
            value = value.lstrip()[4:].lstrip()
    return json.loads(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path("/opt/data/.env"))
    args = parser.parse_args()

    source_text = "\n\n".join(
        f"=== SOURCE: {path.name} ===\n{path.read_text()}" for path in args.source
    )
    client = genai.Client(api_key=read_key(args.env))
    results: dict[str, dict] = {}

    for number in range(1, 17):
        slide_path = args.slides_dir / f"slide-{number:02d}.png"
        if not slide_path.is_file():
            raise FileNotFoundError(slide_path)
        job, feel = INTENTS[number]
        prompt = RUBRIC.format(audience=AUDIENCE, job=job, feel=feel)
        contents = [
            prompt,
            f"\n=== LOCKED SOURCE CONTRACT ===\n{source_text}\n\n=== RENDERED SLIDE {number} ===",
            types.Part.from_bytes(data=slide_path.read_bytes(), mime_type="image/png"),
        ]
        last_error: Exception | None = None
        for attempt in range(1, 3):
            try:
                response = client.models.generate_content(model=MODEL, contents=contents)
                result = parse_json(response.text or "")
                break
            except Exception as error:
                last_error = error
                if attempt == 2:
                    raise
                time.sleep(2)
        else:
            raise RuntimeError(last_error)

        scores = result.get("scores", {})
        if scores:
            result["mean"] = round(sum(float(value) for value in scores.values()) / len(scores), 2)
        result["passed"] = result.get("mean", 0) >= 7.5 and not result.get("critical_flags")
        results[str(number)] = result
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(
            f"slide {number:02d}: mean={result.get('mean')} "
            f"critical={len(result.get('critical_flags', []))} passed={result['passed']}",
            file=sys.stderr,
            flush=True,
        )

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
