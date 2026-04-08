import re

class SlakeEvaluator:
    @staticmethod
    def clean_text(text):
        # 소문자 변환, 특수문자 제거, 공백 정리
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def evaluate(self, prediction, ground_truth):
        pred = self.clean_text(prediction)
        gt = self.clean_text(ground_truth)
        
        # Exact Match: 정답 단어가 모델 답변에 포함되어 있는가?
        if gt in pred:
            return 1
        return 0