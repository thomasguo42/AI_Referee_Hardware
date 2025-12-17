# AI Saber Fencing Referee – Gemini Prompt

You are helping me design an **AI saber fencing referee** using an existing codebase and labeled data.

---

## 0. Overall mode (VERY IMPORTANT)

You are operating in a **research-grade, careful, slow** mode:

- There is **no time pressure**. Take your time to process and think.
- Do **not** optimize for speed. Optimize for **correctness, rigor, and caution**.
- Use thorough internal reasoning, but in your responses, give **structured, high-level explanations** instead of raw step-by-step inner thoughts.
- **Target accuracy is at least 80%** on deciding who wins each touch. If you don’t think that’s achievable, you must explain why in detail.

You must **not make weird or unsupported assumptions**. When something is unclear, you must explicitly mark it as uncertain and either:
- Propose multiple plausible interpretations, and/or  
- Ask me precise clarification questions.

---

## 1. Repository and data description (must use this carefully)

First, **scan the codebase**, especially the `training_data` folder, to familiarize yourself with it.

The `training_data` folder contains many subfolders.  
Each subfolder represents the information of **a single fencing video** (one bout or one touch context). Inside each subfolder, you will typically find:

1. **Videos**
   - One or more original videos of the fencing action.
   - One or more corrected videos with keypoints overlaid.

2. **Excel file**
   - This is data extracted using YOLO (pose/keypoint extraction) from the fencing video.
   - It contains **four spreadsheets**, representing the **x and y coordinates** for the left and right fencer:
     - `left_x`
     - `left_y`
     - `right_x`
     - `right_y`
   - Each row corresponds to a frame (FPS is around 15; in some data the exact FPS is 15).

3. **TXT file**
   - A text file that encodes event information for that video.
   - The exact format of the TXT file might be **different across different subfolders**,
     but **subfolders taken within the same day will share the same TXT format**.
   - It includes information such as:
     - Blade-to-blade contact (which might correspond to beat, parry, or other actions that you must interpret together with the Excel motion data).
     - When someone hits (touch/valid light information).
     - In some setups, the **actual result** (who the true winner is) may be encoded here or elsewhere in a non-JSON file.  
       You should use these trusted labels to evaluate accuracy.

4. **JSON file**
   - The JSON file *nominally* contains information about:
     - Blade-to-blade contact events (could be beat, parry, or other actions).
     - When each fencer hits.
     - Who the actual winner is.
   - HOWEVER: In this project, the JSON file is **not reliable and should be treated as containing false or untrustworthy information**.

   **Strict rule:**  
   - You must **NOT use the JSON file at all** for any reasoning, labels, training, validation, or evaluation.
   - You may acknowledge that it exists and what it claims to contain, but you must not rely on it.
   - Use ONLY the Excel and TXT files (and any other clearly designated non-JSON label sources) as trustworthy data.

---

## 2. Your first tasks (data understanding, no guessing)

Before proposing any algorithm, you must:

1. **Inspect the actual data structures**:
   - For the Excel file:  
     - Confirm sheet names, column names, indexing, and how frames are represented.
     - Confirm how left/right fencer coordinates are arranged.
   - For the TXT file:  
     - Describe precisely how blade-to-blade contact events, hits, and winners (if present) are encoded.
     - Note all patterns and variations across different subfolders.

2. **Summarize what you truly see**:
   - Provide a concise but precise summary of:
     - Excel structure
     - TXT structure
     - Any known FPS or timing conversion (e.g., 15 FPS, frame index ↔ time).

3. **List all uncertainties and assumptions explicitly**:
   - For example:
     - How you map coordinates to “forward” and “backward” movement for left vs right.
     - How you align TXT timestamps or event markers to Excel frames.
     - How blade-to-blade contact lines should be interpreted (beat vs parry vs something else).
   - If you can’t verify something from the code or data, **do not silently assume it**.
     Instead, mark it clearly as “unverified” and, if necessary, ask me specific questions.

4. **Confirm you are not using JSON**:
   - Explicitly state in your own words that you will **ignore all JSON files** because they contain false information.
   - Base all further design only on Excel + TXT + any trusted non-JSON labels.

---

## 3. Goal: Saber fencing referee algorithm

Your goal is to develop an algorithm that can **referee saber bouts**, i.e., decide **who wins and who loses each touch**, using:

- Motion data (Excel: left/right x and y over time)
- Event data (TXT: blade-to-blade contacts, hits, ground-truth outcomes where present)
- Saber right-of-way rules

You should:

- Use the actual results (from trusted sources such as TXT or other non-JSON labels) to **evaluate accuracy**.
- Aim for **at least 80% accuracy** in deciding the correct winner.

---

## 4. Explore multiple approaches (no single-shot shortcuts)

You must **not** jump directly to one approach.  
Instead, you must think carefully and propose **several distinct strategies** and compare them before committing.

At minimum, consider:

### 4.1 Rule-based system (expert/logic system)

Use handcrafted features from the Excel + TXT:

- Forward and backward movement  
- Arm extension and initiation timing  
- Lunge detection  
- Blade-to-blade contact classification (beat, parry, or other) inferred from the motion around contact events  

Implement saber right-of-way logic:

- Who initiates the attack  
- Whether a parry occurs  
- Whether there is a riposte  
- Handling of simultaneous hits  

### 4.2 Classical ML model

- Build features from time windows around events:
  - Velocities, accelerations, distances, timing of first forward intent, contact windows, etc.
- Train a model (e.g. random forest, gradient boosting, etc.) to predict the winner.
- Discuss interpretability and robustness.

### 4.3 Sequence model

- Treat the keypoint time series as sequences (e.g. RNN / GRU / LSTM / TCN / Transformer).
- Include markers for blade-to-blade contact and hit events from TXT.
- Predict either:
  - Right-of-way state + winner, or
  - Winner directly.

For each approach, you must:

- Explain what features/signals you use.
- Explain how saber right-of-way is encoded.
- Discuss strengths and limitations.
- Assess whether achieving ≥80% accuracy seems realistic.
- Clearly state what additional information or clarifications would improve the approach.

Then, choose **one or two** approaches to pursue first and justify your choice.

---

## 5. Design a rigorous pipeline

After choosing your approach(es), design a **step-by-step pipeline** that includes:

### 5.1 Data loading and normalization

- How you load Excel and TXT.
- How you handle different TXT formats across subfolders.
- How you synchronize frames, timestamps, and events.
- How you handle missing or noisy data.
- How you normalize coordinate directions (e.g., left’s forward vs right’s forward).

### 5.2 Feature extraction

Define saber-relevant features, such as:

- Forward/backward movement  
- Arm extension and attack initiation  
- Lunge detection  
- Blade-to-blade contact classification (beat, parry, unknown)  
- Hit timing windows  
- Simultaneous hit resolution logic  

Explain how each feature relates to saber right-of-way.

### 5.3 Model / logic implementation

- For a rule-based system: describe rules and thresholds.
- For an ML model: describe inputs, model type, and training labels.
- For a sequence model: describe the sequence representation and output labels.

### 5.4 Evaluation protocol

- How you use actual results to compute accuracy (touch-level accuracy at minimum).
- How you split data (e.g., by video or subfolder) to avoid leakage.
- How you will analyze errors, such as:
  - Systematic bias toward left or right fencer.
  - Failures around simultaneous or near-simultaneous actions.
  - Misclassification patterns (e.g., attacks incorrectly judged as counterattacks).

If your proposed method does not realistically reach 80% accuracy, clearly explain why and propose how to improve or extend it.

---

## 6. Strictness about assumptions and questions

Throughout this process:

- **Do not** silently assume anything that isn’t clearly supported by the data or code.
- Every assumption must be:
  - Explicitly written down.
  - Marked as either “verified from data/code” or “unverified, needs confirmation.”
- If you need more information (e.g., example rows of a TXT file, a screenshot of Excel, or exact saber rule nuances as used in this dataset), explicitly ask me.

---

## 7. Output format

Your initial output should include:

1. A clear summary of:
   - What you found in the Excel and TXT files.
   - Confirmation that you have completely ignored JSON.

2. A list of:
   - Explicit assumptions.
   - Open questions and uncertainties.

3. A comparison of **at least three** different approaches (like the ones above).

4. Your chosen approach or combination of approaches and why you chose them.

5. A detailed, step-by-step **pipeline plan** for implementation.

6. An explanation of how you will:
   - Measure accuracy (with a target of ≥80%).
   - Perform error analysis and iterate.

Only after you’ve done all of this should you move on to more detailed design or pseudo-code.

---

Use everything above as strict requirements.  
You may now begin your analysis under these constraints.
