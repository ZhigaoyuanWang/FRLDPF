
class ValueMonitor:
    def __init__(self,threshold):
        self.previous_value = None
        self.increase_count = 0
        self.threshold = threshold

    def check_value(self, value):
        if self.previous_value is not None and value > self.previous_value:
            self.increase_count += 1

        if self.previous_value is not None and value < self.previous_value:
            self.increase_count = 0
        self.previous_value = value

        if self.increase_count >= self.threshold:
            return True
        else:
            return False