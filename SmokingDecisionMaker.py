class SmokingDecisionMaker:
    def __init__(self, confirm_threshold):
        self.confirm_threshold = confirm_threshold
        self.smoking_streak = 0
        self.smoking_confirmed = False

    def update(self, detections):
        if detections:
            self.smoking_streak += 1

        else:
            self.smoking_streak = 0
        if self.smoking_streak >= self.confirm_threshold:
            self.smoking_confirmed = True
        else:
            self.smoking_confirmed = False

    def reset(self):
        self.smoking_streak = 0
        self.smoking_confirmed = False

    def is_smoking_confirmed(self):
        return self.smoking_confirmed
