"""
datasets.py
-----------
Static reference data: dataset descriptions and their event mappings.
Nothing here changes between runs.
"""

DATASET_EVENT_MAPPINGS = {

    'BNCI2014_001': {
        'description': '9 subjects, 4-class MI (left/right hand, feet, tongue)',
        'n_subjects': 9,
        'original_events': ['left_hand', 'right_hand', 'feet', 'tongue'],
        'left_vs_right': {'left_hand': 0, 'right_hand': 1},
        #'all_classes': {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3},
        'movement_vs_rest': None,
    },
    'BNCI2014_004': {
        'description': '9 subjects, 2-class MI (left vs right hand only)',
        'n_subjects': 9,
        'original_events': ['left_hand', 'right_hand'],
        'left_vs_right': {'left_hand': 0, 'right_hand': 1},
        'movement_vs_rest': None,
    },
    'BNCI2014_002': {
        'description': '14 subjects, 2-class MI (right hand vs feet)',
        'n_subjects': 14,
        'original_events': ['right_hand', 'feet'],
        'left_vs_right': None,
        'right_vs_feet': {'right_hand': 0, 'feet': 1},
        'movement_vs_rest': None,
    },
    'Weibo2014': {
        'description': '10 subjects, 3-class MI + rest',
        'n_subjects': 10,
        'original_events': ['left_hand', 'right_hand', 'feet', 'rest'],
        'left_vs_right': {'left_hand': 0, 'right_hand': 1},
        'movement_vs_rest': {'rest': 0, 'left_hand': 1, 'right_hand': 1, 'feet': 1},
    },
    'PhysionetMI': {
        'description': '109 subjects, 4-class MI (left/right hand, feet, both hands)',
        'n_subjects': 109,
        'original_events': ['left_hand', 'right_hand', 'feet', 'hands', 'rest'],
        'left_vs_right': {'left_hand': 0, 'right_hand': 1},
        #'all_classes': {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'hands': 3, 'rest': 4},
        'movement_vs_rest': {'rest': 0, 'left_hand': 1, 'right_hand': 1, 'feet': 1, 'hands': 1},
    },
    'Schirrmeister2017': {
        'description': '14 subjects, 4-class MI with high-gamma',
        'n_subjects': 14,
        'original_events': ['left_hand', 'right_hand', 'feet', 'rest'],
        'left_vs_right': {'left_hand': 0, 'right_hand': 1},
        'movement_vs_rest': {'rest': 0, 'left_hand': 1, 'right_hand': 1, 'feet': 1},
    },
    'Ofner2017': {
        'description': '15 subjects, 6 motor imagery classes (upper limb) + rest',
        'n_subjects': 15,
        'movement_vs_rest': {
            'rest': 0, 'right_elbow_flexion': 1, 'right_elbow_extension': 1,
            'right_supination': 1, 'right_pronation': 1,
            'right_hand_close': 1, 'right_hand_open': 1
        },
    },
    'Lee2019_MI': {
        'description': '54 subjects, 2-class MI',
        'n_subjects': 54,
        'original_events': ['left_hand', 'right_hand'],
        'left_vs_right': {'left_hand': 0, 'right_hand': 1},
    },
    'GrosseWentrup2009': {
        'description': '10 subjects, classic 2-class MI',
        'n_subjects': 10,
        'original_events': ['left_hand', 'right_hand'],
        'left_vs_right': {'left_hand': 0, 'right_hand': 1},
    },
    'AlexMI': {
        'description': '8 subjects, 3-class MI (right hand, feet) + rest',
        'n_subjects': 8,
        'original_events': ['right_hand', 'feet', 'rest'],
        'all_classes': {'right_hand': 2, 'feet': 3, 'rest': 4},
        'movement_vs_rest': {'rest': 0, 'right_hand': 1, 'feet': 1},
    },
    'Beetl2021_A': {
        'description': 'BEETL Competition Dataset A (4-class)',
        'n_subjects': 4,
        'original_events': ['left_hand', 'right_hand', 'feet', 'rest'],
        'left_vs_right': {'left_hand': 1, 'right_hand': 2},
        'all_classes': {'rest': 0, 'left_hand': 1, 'right_hand': 2, 'feet': 3},
        'movement_vs_rest': {'rest': 0, 'left_hand': 1, 'right_hand': 1, 'feet': 1},
    },
}


def print_dataset_summary() -> None:
    """Prints a table of all registered datasets."""
    print("Datasets")
    print(f"\n{'Dataset':<20} {'Subjects':<10} {'left/right':<12} {'move/rest':<12} {'Note'}")
    print("-" * 80)
    for name, info in DATASET_EVENT_MAPPINGS.items():
        subjects   = str(info['n_subjects'])
        left_right = 'yes' if info.get('left_vs_right') else 'no'
        move_rest  = 'yes' if info.get('movement_vs_rest') else 'no'
        note       = info.get('note', '')[:40]
        print(f"{name:<20} {subjects:<10} {left_right:<12} {move_rest:<12} {note}")
    print()
