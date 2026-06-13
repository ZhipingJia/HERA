LAYER_ORDER = [
    'features.0', 'features.3',
    'features.7', 'features.10',
    'features.14', 'features.17', 'features.20',
    'features.24', 'features.27', 'features.30',
    'features.34', 'features.37', 'features.40',
    'classifier.0', 'classifier.3', 'classifier.6',
]

LAYER_NAMES_SHORT = [
    'B1_C1', 'B1_C2', 'B2_C1', 'B2_C2',
    'B3_C1', 'B3_C2', 'B3_C3',
    'B4_C1', 'B4_C2', 'B4_C3',
    'B5_C1', 'B5_C2', 'B5_C3',
    'FC1', 'FC2', 'FC3',
]

SHORT_TO_FULL = {short: full for short, full in zip(LAYER_NAMES_SHORT, LAYER_ORDER)}
FULL_TO_SHORT = {full: short for short, full in zip(LAYER_NAMES_SHORT, LAYER_ORDER)}


def build_layer_name_mappings():
    return SHORT_TO_FULL.copy(), FULL_TO_SHORT.copy()
