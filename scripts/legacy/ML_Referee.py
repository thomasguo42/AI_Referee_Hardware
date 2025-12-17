import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import FencingPhraseLoader

class MLReferee:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.VELOCITY_THRESHOLD = 0.05
        self.KP_FRONT_FOOT = 16
        self.KP_BACK_FOOT = 15

    def extract_phrase_features(self, phrase_data):
        """
        Extracts a single feature vector for the entire phrase.
        """
        left_x = phrase_data['left_x']
        right_x = phrase_data['right_x']
        events = phrase_data['events']
        
        # Find Hit Frame
        hit_event = next((e for e in events if e['type'] == 'hit'), None)
        if not hit_event:
            return None
        hit_frame = hit_event['frame']
        
        # Window: 2 seconds before hit (approx 30 frames)
        max_frames = len(left_x)
        safe_hit_frame = min(hit_frame, max_frames)
        start_frame = max(0, safe_hit_frame - 30)
        
        # Slice Data
        l_x_window = left_x.iloc[start_frame:safe_hit_frame].mean(axis=1) # Center of mass X
        r_x_window = right_x.iloc[start_frame:safe_hit_frame].mean(axis=1)
        
        # Velocity
        l_vel = l_x_window.diff().fillna(0)
        r_vel = r_x_window.diff().fillna(0) * -1 # Invert for Right
        
        # Lunge (at hit frame)
        try:
            l_lunge = abs(left_x.iloc[safe_hit_frame-1][f'kp_{self.KP_FRONT_FOOT}'] - left_x.iloc[safe_hit_frame-1][f'kp_{self.KP_BACK_FOOT}'])
            r_lunge = abs(right_x.iloc[safe_hit_frame-1][f'kp_{self.KP_FRONT_FOOT}'] - right_x.iloc[safe_hit_frame-1][f'kp_{self.KP_BACK_FOOT}'])
        except:
            l_lunge = 0
            r_lunge = 0

        # Feature Vector
        features = {
            'l_max_vel': l_vel.max(),
            'r_max_vel': r_vel.max(),
            'l_avg_vel': l_vel.mean(),
            'r_avg_vel': r_vel.mean(),
            'l_lunge': l_lunge,
            'r_lunge': r_lunge,
            'vel_diff': l_vel.max() - r_vel.max(),
            'lunge_diff': l_lunge - r_lunge,
            'has_blade_contact': 1 if any(e['type'] == 'blade_contact' for e in events) else 0
        }
        
        # Who moved first?
        l_start = (l_vel > self.VELOCITY_THRESHOLD).idxmax() if (l_vel > self.VELOCITY_THRESHOLD).any() else safe_hit_frame
        r_start = (r_vel > self.VELOCITY_THRESHOLD).idxmax() if (r_vel > self.VELOCITY_THRESHOLD).any() else safe_hit_frame
        
        features['start_diff'] = r_start - l_start # Positive means Left started earlier (smaller index)
        
        return features

    def prepare_dataset(self, all_phrases):
        X = []
        y = []
        folders = []
        
        for phrase in all_phrases:
            if not phrase['winner']:
                continue
                
            feats = self.extract_phrase_features(phrase)
            if feats:
                X.append(list(feats.values()))
                y.append(phrase['winner'])
                folders.append(phrase['folder'])
                
        return np.array(X), np.array(y), folders, list(feats.keys())

    def train_and_evaluate(self, all_phrases):
        X, y, folders, feature_names = self.prepare_dataset(all_phrases)
        
        print(f"Dataset Size: {len(X)} samples")
        print(f"Class Distribution: {pd.Series(y).value_counts().to_dict()}")
        
        # Cross Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=cv)
        
        print(f"\nCross-Validation Accuracy: {scores.mean():.2%} (+/- {scores.std() * 2:.2%})")
        
        # Train on full set for feature importance
        self.model.fit(X, y)
        importances = pd.DataFrame({'feature': feature_names, 'importance': self.model.feature_importances_})
        print("\nTop 5 Important Features:")
        print(importances.sort_values('importance', ascending=False).head(5))

if __name__ == "__main__":
    loader = FencingPhraseLoader("/workspace/training_data")
    all_phrases = loader.load_all()
    
    ml_referee = MLReferee()
    ml_referee.train_and_evaluate(all_phrases)
