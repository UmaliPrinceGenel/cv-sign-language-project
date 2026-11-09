# Milestone 4 — Test Cases (10 Cases Table)

**Project:** Real-time Sign Language Word Gesture Recognition  
**Milestone:** 4 (80% Progress Check)  
**Git Tag:** `cp4-80`  
**Pipeline Configuration:** MediaPipe Tasks API + Scaler + 2-Layer MLP + Geometric Refinement  

---

## 1. Formal Test Matrix

| ID | Input Source | Test Condition / Variant | Expected Output | Actual Output & Conf. | Status | Failure Cause / Notes |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **TC01** | `data/call_01.jpg` | Standard indoor lighting, centered thumb & pinky | `CALL` | `CALL` (67.2%) | **✔ PASS** | Baseline coordinate check matches phone call pose. |
| **TC02** | `data/dislike_01.jpg` | Downward thumb orientation with closed fingers | `DISLIKE` | `DISLIKE` (99.9%) | **✔ PASS** | Clean negative vertical vector extracted. |
| **TC03** | `data/like_01.jpg` | Upward thumb orientation with closed fingers | `LIKE` | `LIKE` (100.0%) | **✔ PASS** | High-contrast upright thumb classified reliably. |
| **TC04** | `data/ok_02.jpg` | Pinched thumb and index tip, 3 fingers extended | `OK` | `OK` (100.0%) | **✔ PASS** | Geometric refinement verifies thumb-index distance ratio (< 0.36 palm length). |
| **TC05** | `data/peace_01.jpg` | V-sign with index & middle extended, ring/pinky curled | `PEACE` | `PEACE` (100.0%) | **✔ PASS** | Sharp angular separation between fingers 8 and 12. |
| **TC06** | `data/stop_01.jpg` | Open vertical hand with fingers pressed together | `STOP` | `STOP` (99.8%) | **✔ PASS** | Angular spread < 6.0° confirms closed fingers. |
| **TC07** | `data/test/call/c306dd66-e1a0-4a4c-b3fa-8597a81a3c48.jpg` | **[Hard Case 1]** Foreshortened camera perspective where pinky points directly toward lens | `CALL` | `MUTE` (89.0%) | **✘ FAIL** | 2D projection reduces perceived distance between pinky tip and ring knuckle. |
| **TC08** | `data/test/stop/b1fa39af-6d9d-4340-857b-7e880ab7494c.jpg` | **[Hard Case 2]** Boundary finger spread (inter-finger spread angle ~6.2°) | `STOP` | `PALM` (90.0%) | **✘ FAIL** | Subject's fingers slightly spread past the 6.0° threshold, triggering the PALM decision boundary. |
| **TC09** | `data/fist_01.jpg` | **[Hard Case 3]** Out-of-vocabulary negative sample (closed fist without extended thumb) | `REJECT / UNCERTAIN` | `MUTE` (88.1%) | **✘ FAIL** | Closed-world softmax distribution forces assignment to closest trained class (MUTE). |
| **TC10** | Live Webcam Feed | Live video stream on laptop CPU, natural room lighting, dynamic hand movement | Real-time tracking @ $\ge$ 25 FPS | `ACTIVE GESTURE` (28.5 FPS, 14.2 ms) | **✔ PASS** | Background MSMF camera thread maintains smooth 30 FPS without stutter or frame stalls. |

---

## 2. Hard & Edge Case Deep-Dive

### Hard Case 1: Extreme Foreshortening (`TC07`)
- **Observation:** In image `c306dd66-e1a0-4a4c-b3fa-8597a81a3c48.jpg`, the user angled their wrist towards the webcam. Although the pinky was extended in 3D space, its 2D projected distance onto the camera sensor plane collapsed.
- **Remedy for Milestone 5:** Transition from normalized 2D image coordinates to 3D metric world landmarks (`world_landmarks` output from MediaPipe), which measures anatomical distances in metric millimeters regardless of focal angle.

### Hard Case 2: Boundary Finger Spread (`TC08`)
- **Observation:** In image `b1fa39af-6d9d-4340-857b-7e880ab7494c.jpg`, the user intended a `STOP` gesture, but their fingers were relaxed and separated by an average angle of 6.2° (the boundary is 6.0°).
- **Remedy for Milestone 5:** Implement soft multi-feature scoring incorporating thumb posture (thumb tucked in for STOP vs abducted for PALM) to disambiguate the boundary zone.

### Hard Case 3: Out-of-Vocabulary Closed Fist (`TC09`)
- **Observation:** When an untrained gesture (e.g., closed fist, rock sign, or waving hand) is presented, the MLP classifier produces high softmax confidence for the closest class in the 8-class space (fist $\rightarrow$ `MUTE`).
- **Remedy for Milestone 5:** Introduce an explicit OOV / background rejection threshold based on energy scores or an Open-Set Distance metric from class centroids.

---

## 3. How to Execute Test Cases

### Interactive App Execution:
Launch the interactive web application and use the built-in quick test gallery:
```bash
python src/app.py
```
Open `http://127.0.0.1:7860` in any browser to verify `TC01`–`TC10`.

### CLI Batch Reproduction:
Run the reproducible evaluation script to execute all test cases across the entire test set:
```bash
python src/evaluate.py --split test --weights models/best.pt
```
Results are saved to `outputs/eval/metrics.json` and `outputs/eval/confusion_matrix.png`.

