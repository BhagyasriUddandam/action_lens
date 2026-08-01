# ActionLens — what I found

I wanted to answer a real question: for something like fall detection or activity 
monitoring, do you actually need a heavy VLM, or does a plain old specialized model 
do the job? Everyone assumes VLMs are just "better," but nobody puts a number on it.

So I built both and tested them against each other on the same 160 clips 
(walking, sitting, standing, falling — mixed from HMDB51 and a real fall-detection 
dataset, URFD).

## First, the baseline

I almost made a dumb mistake here — I was going to just use a Kinetics-pretrained 
model out of the box, but when I actually checked its label list, none of my four 
classes exist in Kinetics-400. Would've scored basically zero. So instead I froze 
the backbone and only trained a small classifier head on top (32 clips), which is 
the honest way to do this without a GPU cluster.

Got 60.9% overall. Not bad, but here's the interesting part: **sitting and standing 
were basically a coin flip** — 40.6% and 37.5%. I dug into why, and it's actually a 
cool problem: "sitting down" and "standing up" look almost identical if you don't 
know which direction time is flowing. The model has no idea which frame came first, 
so of course it can't tell them apart.

## Then the VLM

Same 160 clips, same fair split. I gave it 8 frames per clip and specifically told 
it "these are in order, compare the first frame to the last" — basically handing it 
the one piece of information the other model didn't have.

92.2% overall. And sitting/standing jumped to 93.8% and 87.5%. That's not a small 
bump — that's the exact failure mode, solved, because the VLM can actually reason 
about time direction instead of just pattern-matching a single frame.

Where they disagreed, the VLM was right 45 times vs. the baseline's 5.

## But it's not free

~4.4 seconds per clip and about a cent each, vs. basically instant and free for the 
baseline. So this isn't "VLMs win, always use them" — it's "use the cheap fast model 
until you hit something direction-dependent, then bring in the VLM."

## What I'd tell someone building this for real

Don't default to the expensive model everywhere. Profile where your specialized 
model actually breaks (for me, it was anything involving direction of motion) and 
only pay the VLM tax there.

## Things I'm not claiming

- Only tested one VLM — different models might behave differently
- Found 2 clips where the VLM's explanation text got cut off oddly — double checked, 
  the actual predictions were fine, just a text field glitch
- Small sample per class, so the exact percentages would move around a bit with 
  more data
- Haven't tried this on live video, only pre-recorded clips
