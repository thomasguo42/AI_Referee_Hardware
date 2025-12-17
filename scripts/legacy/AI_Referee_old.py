import numpy as np
import pandas as pd
from data_loader import FencingPhraseLoader

class FencingReferee:
    def __init__(self):
        # Constants for feature extraction
        self.VELOCITY_THRESHOLD = 0.05  # Threshold for "forward movement" (Adjusted from 0.5)
        self.LUNGE_THRESHOLD = 1.2     # Threshold for lunge depth (normalized)
        self.ARM_EXTENSION_THRESHOLD = 0.8 # Threshold for arm extension
        
        # Keypoint Mapping (User provided: 16=front leg, 15=back leg)
        # Assuming standard COCO-like for upper body but user didn't specify.
        # We will use leg distance as a primary proxy for lunge for now.
        self.KP_FRONT_FOOT = 16
        self.KP_BACK_FOOT = 15
        
    def extract_features(self, phrase_data):
        """
        Extracts time-series features from the raw keypoints.
        """
        left_x = phrase_data['left_x']
        right_x = phrase_data['right_x']
        
        # 1. Calculate Center of Mass (approximate with mean of all KPs)
        # This is more robust than a single point for velocity
        left_com_x = left_x.mean(axis=1)
        right_com_x = right_x.mean(axis=1)
        
        # 2. Calculate Velocity (Forward = +X for Left, -X for Right)
        # We use a rolling window to smooth out jitter
        left_vel = left_com_x.diff().rolling(window=3).mean().fillna(0)
        right_vel = right_com_x.diff().rolling(window=3).mean().fillna(0) * -1 # Invert for Right
        
        # 3. Calculate Lunge Depth (Distance between feet)
        # Euclidean distance would be better if we had Y, but X diff is a good proxy for stance width
        left_lunge = (left_x[f'kp_{self.KP_FRONT_FOOT}'] - left_x[f'kp_{self.KP_BACK_FOOT}']).abs()
        right_lunge = (right_x[f'kp_{self.KP_FRONT_FOOT}'] - right_x[f'kp_{self.KP_BACK_FOOT}']).abs()
        
        return {
            'left_vel': left_vel,
            'right_vel': right_vel,
            'left_lunge': left_lunge,
            'right_lunge': right_lunge
        }

    def decide_winner(self, phrase_data):
        """
        Implements the FIE Right-of-Way logic.
        """
        features = self.extract_features(phrase_data)
        events = phrase_data['events']
        
        # Sort events by time
        events.sort(key=lambda x: x['frame'])
        
        # State Machine Variables
        # 0 = Neutral, 1 = Left Attack, 2 = Right Attack, 3 = Simultaneous
        state = "Neutral"
        current_attacker = None
        
        # Logic Loop: Iterate through frames or key events?
        # Better to iterate through key events and check "state" at that moment.
        
        # Find the "Hit" event
        hit_event = next((e for e in events if e['type'] == 'hit'), None)
        if not hit_event:
            return "unknown", "No hit detected"
            
        hit_frame = hit_event['frame']
        
        # Analyze the window BEFORE the hit (e.g., last 2 seconds or from 'start')
        start_frame = max(0, hit_frame - 45) # Look back 3 seconds (45 frames)
        
        # Simple Logic: Who started moving forward first and didn't stop?
        
        left_attack_start = -1
        right_attack_start = -1
        
        # Scan frames leading up to hit
        max_frames = len(features['left_vel'])
        safe_hit_frame = min(hit_frame, max_frames)
        start_frame = max(0, safe_hit_frame - 45)

        for f in range(start_frame, safe_hit_frame):
            if f >= max_frames:
                break
                
            l_vel = features['left_vel'].iloc[f]
            r_vel = features['right_vel'].iloc[f]
            
            # Check for Blade Contact at this frame
            contact = any(e['frame'] == f and e['type'] == 'blade_contact' for e in events)
            
            if contact:
                # If blade contact, priority usually resets or goes to riposte
                # Simplified: If Left was attacking, Right parried -> Right ROW
                if state == "Left Attack":
                    state = "Right Attack" # Riposte
                    current_attacker = "right"
                elif state == "Right Attack":
                    state = "Left Attack" # Riposte
                    current_attacker = "left"
                continue

            # Check for Attack Initiation
            if state == "Neutral":
                if l_vel > self.VELOCITY_THRESHOLD and r_vel < self.VELOCITY_THRESHOLD:
                    state = "Left Attack"
                    current_attacker = "left"
                    left_attack_start = f
                elif r_vel > self.VELOCITY_THRESHOLD and l_vel < self.VELOCITY_THRESHOLD:
                    state = "Right Attack"
                    current_attacker = "right"
                    right_attack_start = f
                elif l_vel > self.VELOCITY_THRESHOLD and r_vel > self.VELOCITY_THRESHOLD:
                    state = "Simultaneous"
            
            # Check for Attack Stops (Mal-parry or stop)
            if state == "Left Attack" and l_vel < 0: # Left stopped/retreated
                state = "Neutral" # Priority lost
            elif state == "Right Attack" and r_vel < 0: # Right stopped/retreated
                state = "Neutral"

        # Final Decision at Hit Frame
        if state == "Left Attack":
            return "left", "Left attack established and maintained"
        elif state == "Right Attack":
            return "right", "Right attack established and maintained"
        elif state == "Simultaneous":
            return "simultaneous", "Both fencers attacked simultaneously"
        else:
            return "simultaneous", "No clear priority established"

    def evaluate(self, all_phrases):
        """
        Runs the referee on all phrases and calculates accuracy.
        """
        correct = 0
        total = 0
        results = []
        
        for phrase in all_phrases:
            ground_truth = phrase['winner']
            if not ground_truth:
                continue # Skip phrases with no clear winner
                
            predicted, reason = self.decide_winner(phrase)
            
            is_correct = (predicted == ground_truth)
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                'folder': phrase['folder'],
                'ground_truth': ground_truth,
                'predicted': predicted,
                'correct': is_correct,
                'reason': reason
            })
            
        accuracy = correct / total if total > 0 else 0
        return accuracy, results

if __name__ == "__main__":
    # Load Data
    loader = FencingPhraseLoader("/workspace/training_data")
    all_phrases = loader.load_all()
    
    # Run Referee
    referee = FencingReferee()
    accuracy, results = referee.evaluate(all_phrases)
    
    print(f"Total Phrases Evaluated: {len(results)}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Show failure analysis
    print("\nFailure Analysis:")
    confusion = {"left": {"right": 0, "simultaneous": 0}, 
                 "right": {"left": 0, "simultaneous": 0},
                 "simultaneous": {"left": 0, "right": 0}}
                 
    for r in results:
        if not r['correct']:
            gt = r['ground_truth']
            pred = r['predicted']
            if gt in confusion and pred in confusion[gt]:
                confusion[gt][pred] += 1
                
    print(f"GT Left -> Pred Right: {confusion['left']['right']}")
    print(f"GT Left -> Pred Sim: {confusion['left']['simultaneous']}")
    print(f"GT Right -> Pred Left: {confusion['right']['left']}")
    print(f"GT Right -> Pred Sim: {confusion['right']['simultaneous']}")
    print(f"GT Sim -> Pred Left: {confusion['simultaneous']['left']}")
    print(f"GT Sim -> Pred Right: {confusion['simultaneous']['right']}")
