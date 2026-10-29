# CSST 106 - Milestone 3: Error Analysis and Improvement Planning

## 1. Metrics & Speed

Our local `src/evaluate.py` evaluation pipeline (MediaPipe Tasks API + MLP Classifier) achieved the following performance metrics on the 241-image unseen test split:

- **Inference Speed**: 21.18 ms per image (47.2 FPS)
- **Overall Accuracy**: 73.86%
- **Macro-F1 Score**: 0.778

> **Note:** The local accuracy is an "honest number" because it strictly penalizes frames where MediaPipe drops the detection (predicting `"no_hand"`), whereas earlier isolated training bypassed those frames.

---

## 2. Error Analysis (The 10-Failure Gallery)

We generated an annotated failure gallery (saved in `outputs/failures/`) and identified three primary mechanical flaws:

- **Palm vs. Stop (Look-alike class):** The MLP struggles to differentiate these open-handed gestures because the 3D coordinates lack explicit wrist-angle context.
- **Call vs. Like (Look-alike class):** Wrist rotation hides the extended pinky and thumb from the camera, causing the MLP to perceive a fist (Like).
- **Two-Hand Interference (The "num_hands=1" Flaw):** As seen in our visual analysis, when a subject has a resting hand near their waist and a gesturing hand raised, MediaPipe's `num_hands=1` configuration frequently locks onto the resting hand. The MLP then correctly classifies the resting hand as a fist ("Like") instead of evaluating the raised "Call" gesture.

---

## 3. The Improvement Plan

| Failure cause | Change | Expected effect / how measured |
| :--- | :--- | :--- |
| Palm vs. Stop confusion | Merge "Palm" and "Stop" into a single "Open Hand" class | Eliminates the largest off-diagonal cell; overall Macro-F1 increases. |
| Call vs. Like confusion | Add synthetic rotation/tilt augmentation to the 3D coordinates before passing them to the MLP | MLP becomes robust to wrist tilt; "Call" precision/recall increases. |
| Resting Hand Interference | Update MediaPipe config to num_hands=2 and add a postprocess() rule to select the hand with the highest Y-coordinate | Eliminates false positives from resting hands; accuracy on full-body frames increases. |

