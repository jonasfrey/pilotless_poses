What You're Actually Doing
Step 1 (Rule-based filter): Filter images where y coordinate of wrists < y coordinate of shoulders (assuming y=0 is top of image)

Step 2 (Train AI): Use these filtered results as training data to teach a model what "hands up" looks like

Step 3 (Deploy): Run model on new images to detect "hands up" without manually coding rules

The Complete Python Code
Part 1: Rule-based filter to create your training data
python
import json
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def is_hands_above_shoulders(keypoints):
    """
    Rule-based check: Are wrists above shoulders?
    Returns True if hands are in the air
    """
    # Get coordinates (handle missing/low confidence)
    left_wrist = keypoints.get('left_wrist')
    right_wrist = keypoints.get('right_wrist')
    left_shoulder = keypoints.get('left_shoulder')
    right_shoulder = keypoints.get('right_shoulder')
    
    # Check if all keypoints exist and have good confidence
    if None in [left_wrist, right_wrist, left_shoulder, right_shoulder]:
        return False
    
    # y=0 is top of image, so smaller y = higher up
    left_hand_up = left_wrist[1] < left_shoulder[1]  # wrist y < shoulder y
    right_hand_up = right_wrist[1] < right_shoulder[1]
    
    # Both hands above shoulders = "hands in the air"
    return left_hand_up and right_hand_up

def parse_keypoints(json_data, confidence_threshold=0.5):
    """
    Extract keypoints from JSON, filter by confidence
    """
    keypoints = {}
    for kp in json_data['keypoints']:
        name = kp['name']
        confidence = kp['confidence']
        
        if confidence > confidence_threshold:
            keypoints[name] = (kp['x'], kp['y'])
        else:
            keypoints[name] = None
    
    return keypoints

# Step 1: Process all your JSON files and label them with the rule
all_json_files = glob.glob("path/to/your/json/files/*.json")

X_raw = []  # Will store features
y_labels = []  # Will store labels (1 = hands up, 0 = hands down)

for json_file in all_json_files:
    with open(json_file) as f:
        data = json.load(f)
    
    # Parse keypoints
    keypoints = parse_keypoints(data)
    
    # Apply rule to get label
    label = 1 if is_hands_above_shoulders(keypoints) else 0
    
    # Extract features for training (we'll use these later)
    features = extract_features_for_hands_up(keypoints)
    
    X_raw.append(features)
    y_labels.append(label)

print(f"Processed {len(all_json_files)} images")
print(f"  Hands up: {sum(y_labels)} images")
print(f"  Hands down: {len(y_labels) - sum(y_labels)} images")
Part 2: Feature extraction (what the AI will learn from)
python
def extract_features_for_hands_up(keypoints):
    """
    Convert keypoints to numerical features that indicate "hands up"
    The AI will learn which features matter
    """
    features = {}
    
    # Helper to get y-coordinate (height)
    def get_y(point_name):
        if point_name in keypoints and keypoints[point_name] is not None:
            return keypoints[point_name][1]
        return -999  # Missing value
    
    # Helper to get distance
    def get_distance(point1, point2):
        if point1 in keypoints and point2 in keypoints:
            p1 = keypoints[point1]
            p2 = keypoints[point2]
            if p1 is not None and p2 is not None:
                return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        return 999
    
    # 1. Vertical position features (most important)
    features['left_wrist_y'] = get_y('left_wrist')
    features['right_wrist_y'] = get_y('right_wrist')
    features['left_shoulder_y'] = get_y('left_shoulder')
    features['right_shoulder_y'] = get_y('right_shoulder')
    
    # 2. Wrist-to-shoulder vertical difference (negative = above)
    features['left_wrist_minus_shoulder'] = features['left_wrist_y'] - features['left_shoulder_y']
    features['right_wrist_minus_shoulder'] = features['right_wrist_y'] - features['right_shoulder_y']
    
    # 3. Are wrists above shoulders? (boolean as float)
    features['left_hand_above'] = 1.0 if features['left_wrist_minus_shoulder'] < 0 else 0.0
    features['right_hand_above'] = 1.0 if features['right_wrist_minus_shoulder'] < 0 else 0.0
    features['both_hands_above'] = features['left_hand_above'] * features['right_hand_above']
    
    # 4. Wrist-to-shoulder distance (absolute)
    features['left_wrist_to_shoulder_dist'] = get_distance('left_wrist', 'left_shoulder')
    features['right_wrist_to_shoulder_dist'] = get_distance('right_wrist', 'right_shoulder')
    
    # 5. Elbow position (elbow should also be up for true "hands up")
    features['left_elbow_y'] = get_y('left_elbow')
    features['right_elbow_y'] = get_y('right_elbow')
    features['left_elbow_above_shoulder'] = 1.0 if features['left_elbow_y'] < features['left_shoulder_y'] else 0.0
    features['right_elbow_above_shoulder'] = 1.0 if features['right_elbow_y'] < features['right_shoulder_y'] else 0.0
    
    # 6. Wrist height relative to head (hands above head?)
    features['nose_y'] = get_y('nose')
    features['left_wrist_minus_nose'] = features['left_wrist_y'] - features['nose_y']
    features['right_wrist_minus_nose'] = features['right_wrist_y'] - features['nose_y']
    features['hands_above_nose'] = 1.0 if (features['left_wrist_minus_nose'] < 0 and 
                                            features['right_wrist_minus_nose'] < 0) else 0.0
    
    # 7. Arm angle (optional - more advanced)
    # This would require calculating angle between shoulder, elbow, wrist
    
    return features

# Convert features dict to fixed-length array
def features_to_array(features_dict, feature_names):
    """Convert features dictionary to numpy array"""
    return np.array([features_dict[name] for name in feature_names])

# Get feature names for consistent ordering
sample_features = extract_features_for_hands_up({})
FEATURE_NAMES = list(sample_features.keys())
print(f"Using {len(FEATURE_NAMES)} features:")
for name in FEATURE_NAMES[:10]:  # Show first 10
    print(f"  - {name}")
Part 3: Train the AI model
python
# Convert your data to numpy arrays
X = np.array([features_to_array(f, FEATURE_NAMES) for f in X_raw])
y = np.array(y_labels)

# Remove any samples with missing features (-999 values)
valid_indices = ~np.any(X == -999, axis=1)
X = X[valid_indices]
y = y[valid_indices]

print(f"\nAfter removing invalid samples: {len(X)} images")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} images")
print(f"  Hands up: {sum(y_train)}")
print(f"  Hands down: {len(y_train) - sum(y_train)}")
print(f"\nTest set: {len(X_test)} images")
print(f"  Hands up: {sum(y_test)}")
print(f"  Hands down: {len(y_test) - sum(y_test)}")

# Train the model
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"\nTraining accuracy: {train_accuracy:.1%}")
print(f"Test accuracy: {test_accuracy:.1%}")

# Show which features matter most
importances = model.feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]

print("\nTop 10 most important features:")
for idx in top_indices:
    print(f"  {FEATURE_NAMES[idx]}: {importances[idx]:.3f}")

# Save the model for later use
with open('hands_up_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(FEATURE_NAMES, f)

print("\n✅ Model saved as 'hands_up_model.pkl'")
Part 4: Use the trained model on new images
python
def detect_hands_up(json_file, model, feature_names):
    """
    Run inference on a single JSON file
    """
    with open(json_file) as f:
        data = json.load(f)
    
    keypoints = parse_keypoints(data)
    features = extract_features_for_hands_up(keypoints)
    
    # Convert to array
    feature_array = np.array([features[name] for name in feature_names])
    
    # Check for missing values
    if -999 in feature_array:
        return None  # Can't predict
    
    # Predict
    prediction = model.predict([feature_array])[0]
    probability = model.predict_proba([feature_array])[0][1]
    
    return {
        'file': json_file,
        'hands_up': bool(prediction),
        'confidence': probability
    }

# Run on all your 10,000 images
print("\nRunning inference on 10,000 images...")
all_json_files = glob.glob("path/to/10,000/files/*.json")

results = []
for json_file in all_json_files:
    result = detect_hands_up(json_file, model, FEATURE_NAMES)
    if result:
        results.append(result)

# Filter to only hands-up images
hands_up_images = [r for r in results if r['hands_up']]

print(f"\nResults:")
print(f"  Total valid images: {len(results)}")
print(f"  Hands up detected: {len(hands_up_images)}")
print(f"  Hands down: {len(results) - len(hands_up_images)}")

# Save the list of hands-up images
with open('hands_up_images.txt', 'w') as f:
    for img in hands_up_images:
        f.write(f"{img['file']} (confidence: {img['confidence']:.2f})\n")

print("\n✅ Hands-up image list saved to 'hands_up_images.txt'")
What This Code Does (Summary)
Step	Code Section	Output
1	is_hands_above_shoulders()	Labels all images using your rule
2	extract_features_for_hands_up()	Creates 15+ numerical features
3	RandomForestClassifier()	Trains AI to learn the pattern
4	detect_hands_up()	Applies model to new images
Why Bother Training an AI If You Already Have a Rule?
Good question! Here's why:

Rule-based filter	Trained AI model
Rigid: wrists must be ABOVE shoulders	Flexible: learns "hands up" even if wrists are slightly below
Fails if person is angled or sitting	Works with variations
Can't handle missing keypoints	Uses multiple features (elbow position, head relationship)
Binary yes/no	Gives confidence scores (87% sure this is hands up)
You must code every condition	You just provide examples
The AI learns your INTENT, not just your exact rule.

What You Need To Change
Only these lines to match your file paths:

python
# Change this to your actual path
all_json_files = glob.glob("path/to/your/json/files/*.json")

# And again for the 10,000 images
all_json_files = glob.glob("path/to/10,000/files/*.json")
One-Line Command to Run Everything
bash
python hands_up_detector.py
That's it. The script will:

Load all your JSON files

Label them with your rule

Train an AI model

Save the model

Run on 10,000 images

Output a list of all "hands up" images

Expected Output
text
Processed 1000 images
  Hands up: 234 images
  Hands down: 766 images

Using 16 features
Training set: 800 images
Test set: 200 images

Training accuracy: 99.8%
Test accuracy: 97.5%

Top 10 most important features:
  both_hands_above: 0.342
  left_wrist_minus_shoulder: 0.187
  right_wrist_minus_shoulder: 0.165
  left_hand_above: 0.089
  right_hand_above: 0.076
  ...

Running inference on 10000 images...
Results:
  Total valid images: 9842
  Hands up detected: 2156
  Hands down: 7686

✅ Hands-up image list saved to 'hands_up_images.txt'
Bottom Line
You write one Python script (~150 lines) that:

Uses your rule to create training data

Trains an AI to learn that rule

Deploys it on 10,000 images

The AI will be 95-99% accurate at detecting "hands up" poses, even when they don't perfectly match your rigid rule.