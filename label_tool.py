"""
label_tool.py  Gradio app for labeling Wikimedia candidate images.

Reads candidates.csv (produced by source_wikimedia.py) and writes labels.csv,
one row per labeled image. Resume-safe: already-labeled images are skipped.

Per image, the labeler answers:
  1. Is the candidate motif actually present? (yes / no)
  2. Are any other taxonomy motifs visible? (multi-select)
  3. Scene tag, emotional tone tag (single-select)
  4. Free-text notes
  5. Labeler initials (persisted across submissions)

Usage:
    python label_tool.py
    python label_tool.py --eval-dir data/eval_set
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import gradio as gr

# Adjust this import to wherever taxonomy.py lives in the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src.taxonomy import MOTIFS
except ImportError:
    from taxonomy import MOTIFS


MOTIF_BY_ID = {m.id: m for m in MOTIFS}
ALL_MOTIF_IDS = sorted(MOTIF_BY_ID.keys())

TONES = [
    "joyful", "tender", "festive", "energetic",
    "contemplative", "somber", "neutral",
]
SCENES = [
    "indoor home", "outdoor gathering", "kitchen", "religious site",
    "market", "natural landscape", "urban street",
    "studio or portrait", "performance or stage", "other",
]

LABEL_FIELDS = [
    "image_path", "motif_id_candidate", "candidate_present",
    "other_motifs", "scene", "tone", "notes", "labeler",
]


# ----------------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------------

def load_candidates(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_labels(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["image_path"]: row for row in csv.DictReader(f)}


def save_label(path: Path, label: Dict):
    existing = load_labels(path)
    existing[label["image_path"]] = label
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for row in existing.values():
            w.writerow(row)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------

def build_app(eval_dir: Path):
    candidates_path = eval_dir / "candidates.csv"
    labels_path = eval_dir / "labels.csv"

    candidates = load_candidates(candidates_path)
    if not candidates:
        raise SystemExit(f"No candidates found at {candidates_path}. "
                         "Run source_wikimedia.py first.")

    def next_unlabeled():
        labels = load_labels(labels_path)
        for c in candidates:
            if c["image_path"] not in labels:
                return c
        return None

    def render():
        c = next_unlabeled()
        labeled = len(load_labels(labels_path))
        progress = f"**Progress:** {labeled} / {len(candidates)} labeled"

        if c is None:
            return (
                None,
                "All done. No unlabeled candidates remaining.",
                "",
                progress,
                "",          # hidden motif id
                "",          # hidden image path
                [],          # other_motifs reset
                None,        # scene reset
                None,        # tone reset
                "",          # notes reset
                True,        # candidate_present reset
            )

        motif = MOTIF_BY_ID.get(c["motif_id_candidate"])
        motif_blurb = (
            f"### Candidate: `{c['motif_id_candidate']}`\n\n"
            f"**{motif.name if motif else '?'}** "
            f"({motif.tradition if motif else ''}, {motif.category if motif else ''})\n\n"
            f"_{motif.description if motif else ''}_"
        )
        license_md = (
            f"**License:** {c.get('license', '')}  \n"
            f"**Source:** [{c.get('wikimedia_title', '')}]"
            f"({c.get('source_page_url', '')})"
        )
        img_path = str(eval_dir / c["image_path"])
        return (
            img_path,
            motif_blurb,
            license_md,
            progress,
            c["motif_id_candidate"],
            c["image_path"],
            [],
            None,
            None,
            "",
            True,
        )

    def submit(image_path, motif_id_candidate, candidate_present,
               other_motifs, scene, tone, notes, labeler):
        if not image_path:
            return render()
        label = {
            "image_path": image_path,
            "motif_id_candidate": motif_id_candidate,
            "candidate_present": "yes" if candidate_present else "no",
            "other_motifs": ",".join(other_motifs or []),
            "scene": scene or "",
            "tone": tone or "",
            "notes": notes or "",
            "labeler": labeler or "",
        }
        save_label(labels_path, label)
        return render()

    with gr.Blocks(title="Raices labeling tool") as app:
        gr.Markdown("# Raices evaluation labeling")
        gr.Markdown(f"Eval dir: `{eval_dir}`. Candidates: **{len(candidates)}**.")

        with gr.Row():
            with gr.Column(scale=2):
                img = gr.Image(type="filepath", label="Image", height=480)
                license_md = gr.Markdown()
            with gr.Column(scale=1):
                motif_md = gr.Markdown()
                progress_md = gr.Markdown()

                hidden_motif_id = gr.Textbox(visible=False)
                hidden_image_path = gr.Textbox(visible=False)

                candidate_present = gr.Checkbox(
                    label="Candidate motif IS present in this image",
                    value=True,
                )
                other_motifs = gr.Dropdown(
                    choices=ALL_MOTIF_IDS, multiselect=True,
                    label="Other taxonomy motifs visible",
                )
                scene = gr.Dropdown(choices=SCENES, label="Scene")
                tone = gr.Dropdown(choices=TONES, label="Emotional tone")
                notes = gr.Textbox(label="Notes (optional)", lines=2)
                labeler = gr.Textbox(
                    label="Labeler (your initials, persists across submissions)",
                )

                with gr.Row():
                    save_btn = gr.Button("Save and next", variant="primary")
                    skip_btn = gr.Button("Re-render (skip)")

        outputs = [
            img, motif_md, license_md, progress_md,
            hidden_motif_id, hidden_image_path,
            other_motifs, scene, tone, notes, candidate_present,
        ]

        app.load(render, outputs=outputs)
        save_btn.click(
            submit,
            inputs=[hidden_image_path, hidden_motif_id, candidate_present,
                    other_motifs, scene, tone, notes, labeler],
            outputs=outputs,
        )
        skip_btn.click(render, outputs=outputs)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval_set")
    ap.add_argument("--share", action="store_true",
                    help="Create a public Gradio share link.")
    args = ap.parse_args()
    app = build_app(Path(args.eval_dir))
    app.launch(share=args.share)


if __name__ == "__main__":
    main()