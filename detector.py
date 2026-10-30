class CognitionDetector:
    def __init__(self, high=-1.5, low=-3.5):
        self.high, self.low = high, low
        
    def band(self, logprobs):
        if not logprobs: return "MID", 0.0
        vals = [t.logprob for t in logprobs.content]
        score = 0.7 * (sum(vals)/len(vals)) + 0.3 * max(min(vals), -10.0)
        if score >= self.high: return "HIGH", score
        if score <= self.low: return "LOW", score
        return "MID", score
