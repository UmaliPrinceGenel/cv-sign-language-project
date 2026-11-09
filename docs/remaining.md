# Milestone 4 — Path to 100% (Remaining Tasks & Roadmap)

**Project:** Real-Time Sign Language Word Gesture Recognition  
**Current Progress:** Milestone 4 (80% Progress Check, Tag `cp4-80`)  
**Target Milestone:** Milestone 5 (100% Final Submission & In-Class Defense)  

---

## 1. To-100% Task Matrix

| Task ID | Task Description | Deliverable / Output | Owner / Lead | Target Date | Status |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **T1** | **Record 5-Minute In-Class Backup Demo Video** | MP4 recording showing `src/app.py` live webcam inference, threshold slider, image uploads, and `TC01`–`TC10` execution. | Kean (Team) | Checkpoint 4 Demo | **Pending User Recording** |
| **T2** | **Apply Git Commit Tag `cp4-80`** | Push git tag `cp4-80` to GitHub repository for teacher evaluation. | Kean | Checkpoint 4 Submission | **Ready to Tag** |
| **T3** | **MediaPipe 3D Metric World Landmarks (`world_landmarks`)** | Extract physical metric distance vectors (mm) rather than 2D screen coordinates to resolve perspective foreshortening in `CALL` and `LIKE`. | Pipeline Engineer | Milestone 5 (Week 1) | Scheduled |
| **T4** | **Open-Set Out-of-Vocabulary (OOV) Rejection** | Introduce distance-to-centroid threshold or energy score to classify untrained hand poses (closed fist, rock sign) as `"UNKNOWN GESTURE"` rather than false positives. | ML Engineer | Milestone 5 (Week 1) | Scheduled |
| **T5** | **Wrist-Centered Full Dataset Model Retraining** | Retrain MLP classifier on translation- and scale-invariant coordinates ($x - x_0, y - y_0, z - z_0$) across all 1,120 images to push accuracy $> 90\%$. | ML Engineer | Milestone 5 (Week 2) | Scheduled |
| **T6** | **Final Milestone 5 Report & Presentation Deck** | Expand `docs/ms4_report.md` into the final Milestone 5 project report following the course template; finalize defense presentation slides. | Documentation Lead | Milestone 5 (Week 2) | Scheduled |
| **T7** | **Final Repository Polish & `final-100` Tagging** | Verify all dependencies in `requirements.txt`, clean temporary artifacts, tag `final-100`. | Team Lead | Final Submission | Scheduled |

---

## 2. In-Class Demo Checklist (Milestone 4 Checkpoint)

Before the instructor calls your team for the live check:
1. [x] **Model Weights Verified:** `models/best.pt` and `models/mediapipe_mlp_model.pkl` present and validated.
2. [x] **Interactive Web App Tested:** `python src/app.py` launches cleanly on `http://127.0.0.1:7860`.
3. [x] **Webcam Stream Tested:** Live video stream responsive with overlay, latency readout (~14 ms), and cumulative counters.
4. [x] **Test Cases Table Ready:** `docs/test_cases.md` covers `TC01` through `TC10` including the 3 hard edge cases.
5. [x] **Reproducible Evaluation Script Verified:**  
   `python src/evaluate.py --split test --weights models/best.pt`  
   Prints exactly **86.25% Accuracy** and **0.897 Macro-F1** and saves `outputs/eval/metrics.json` and `outputs/eval/confusion_matrix.png`.
6. [x] **Documentation & Slides Drafted:** `docs/ms4_report.md` and `docs/slides.md` committed.
7. [ ] **Backup Video Recorded:** Record a 3–5 minute screen capture with audio as backup in case of live hardware failure.
8. [ ] **Git Tag Applied:** Execute `git tag cp4-80` and `git push origin cp4-80`.

