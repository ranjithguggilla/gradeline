"""Gradio front end for scoring a single lot."""
from __future__ import annotations

import gradio as gr

from .data import FEATURES
from .service import load_bundle, score_lot

LABELS = {
    "fixed_acidity": "Fixed acidity (g/L)", "volatile_acidity": "Volatile acidity (g/L)",
    "citric_acid": "Citric acid (g/L)", "residual_sugar": "Residual sugar (g/L)",
    "chlorides": "Chlorides (g/L)", "free_sulfur_dioxide": "Free SO2 (mg/L)",
    "total_sulfur_dioxide": "Total SO2 (mg/L)", "density": "Density (g/cm3)",
    "pH": "pH", "sulphates": "Sulphates (g/L)", "alcohol": "Alcohol (% vol)",
    "is_red": "Red wine (1) or white (0)",
}


def build_app(bundle_path: str = "outputs/gradeline_model.joblib") -> gr.Blocks:
    bundle = load_bundle(bundle_path)
    ref = bundle["reference"]

    def run(*vals):
        out = score_lot(dict(zip(FEATURES, vals, strict=True)), bundle)
        call = "Send to the tasting panel" if out["send_to_panel"] else "Release without review"
        text = [f"{call} (chance of grade 5 or below: {out['p_low_tier']:.0%}, "
                f"cut-off {out['threshold']:.0%})",
                f"Most likely grade: {out['most_likely_grade']}"]
        text += [f"- {r}" for r in out["reasons"]]
        if out["outside_training_range"]:
            text.append("Outside the range seen in training: "
                        + ", ".join(out["outside_training_range"]))
        return out["grade_probabilities"], "\n".join(text)

    with gr.Blocks(title="gradeline") as demo:
        gr.Markdown("## gradeline\nEnter a lot's lab measurements to get a grade "
                    "distribution and a panel triage call.")
        inputs = [gr.Number(label=LABELS[c], value=round(float(ref[c]), 4)) for c in FEATURES]
        btn = gr.Button("Score lot")
        label = gr.Label(label="Grade probabilities")
        notes = gr.Markdown()
        btn.click(run, inputs, [label, notes])
    return demo


if __name__ == "__main__":
    build_app().launch()
