# ActionLens

I built this to answer a real question I kept seeing assumed but never actually 
measured: for recognizing human activity in video, when do you actually need a 
vision-language model instead of a cheap, fast, specialized model?

## The setup

Four classes — walking, sitting, standing, falling. 160 clips, pulled from two 
real datasets: HMDB51 for general actions, and URFD (a real fall-detection 
dataset) mixed with HMDB51's own fall clips, so the "falling" class isn't just 
learning one dataset's room.

32 clips held out for tuning, 128 for the actual scored comparison — same split, 
same clips, for both models, so it's a fair fight.

## What I found

A specialized model (frozen backbone, small trained head — no GPU cluster needed) 
got 60.9% overall. But sitting and standing were nearly a coin flip: 40.6% and 
37.5%. Turns out sitting-down and standing-up look almost identical if you don't 
know which direction time is running, and this model had no way to know.

A vision-language model, given the same clips but told explicitly to compare the 
first frame against the last, got 92.2% overall — and specifically solved sitting 
(93.8%) and standing (87.5%). Where the two models disagreed, the VLM was right 
45 times to the specialized model's 5.

The trade-off: the VLM takes ~4.4 seconds and about a cent per clip; the 
specialized model is instant and free. So it's not "always use the VLM" — it's 
"use the cheap model until you hit something direction-dependent, then bring in 
the smarter one just for that."

Full write-up with the reasoning and honest limitations: [FINDINGS.md](FINDINGS.md)

## Running it

**Data + models** (Python, needs a GPU for reasonable speed):
```bash
pip install -r requirements.txt
python src/data_prep.py
python src/baseline_model.py
python src/vlm_benchmark.py --split eval    # needs ANTHROPIC_API_KEY
python src/evaluate.py
```

**Dashboard** (reads the results above):
```bash
cd dashboard
bun install
bun run dev
```

## Repo layout
src/ data pipeline, baseline model, VLM benchmark, evaluation
dashboard/ results viewer (React + Vite)
results/ metrics, predictions, plots
data/raw/ source datasets (not committed — see scripts/)
scripts/ dataset download scripts
## What I'd do next

Test more than one VLM to see if the sit/stand fix generalizes, and try it on a 
live camera feed instead of pre-recorded clips.