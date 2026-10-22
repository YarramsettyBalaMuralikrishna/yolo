import numpy as np

class Anchors:
    def generate_anchors(scales, ratios):
        anchors = []
        for scale in scales:
            for ratio in ratios:
                width = scale * np.sqrt(ratio)
                height = scale / np.sqrt(ratio)
                anchors.append((width, height))
        return anchors

# our custom made anchors and scales

# scales = [0.1, 0.2, 0.4]
# ratios = [0.5, 1, 2]
# anchors = Anchors.generate_anchors(scales, ratios)
# print(anchors)