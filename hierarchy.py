import numpy as np

def create_superclass(x):
    fine_id_coarse_id = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15, 30: 0,
                         31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16,
                         37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16,
                         66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16,
                         75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}
    return fine_id_coarse_id[x]


v_create_superclass = np.vectorize(create_superclass)


def create_ultraclass(x):
    #coarse_id_ultra_id = {0: 0, 1: 4, 2: 1, 3: 2, 4: 1, 5: 2, 6: 2, 7: 3, 8: 4, 9: 6, 10: 6, 11: 4, 12: 0, 13: 3, 14: 0, 15: 4, 16: 0, 17: 6, 18: 5, 19: 5}
    coarse_id_ultra_id = {0: 3, 1: 5, 2: 6, 3: 1, 4: 6, 5: 1, 6: 1, 7: 2, 8: 3, 9: 0, 10: 4, 11: 3, 12: 3, 13: 2, 14: 3, 15: 5, 16: 3, 17: 4, 18: 7, 19: 7}
    
    return coarse_id_ultra_id[x]

v_create_ultraclass = np.vectorize(create_ultraclass)

def get_labels(idx, level='sub'):
    labels = {
        #'ultra': ['mammal', 'plant', 'household', 'invertebrates', 'other animal', 'vehicles', 'nature'],
        'ultra': ['buildings', 'household', 'invertebrates', 'mammal', 'outdoors', 'other animal', 'plant', 'vehicles'],
                
        'super': ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit & vegetables',
                  'household electrical device', 'household furniture', 'insects', 'large carnivores',
                  'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                  'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
                  ],
        'sub': [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
    }
    return labels[level][idx]

v_get_labels = np.vectorize(get_labels)

def get_super(i): return get_labels(create_superclass(i), "super")

def get_ultra(i): return get_labels(create_ultraclass(create_superclass(i)), "ultra")

labs = range(100)
sorted_labels = sorted(labs, key= lambda x: (create_ultraclass(create_superclass(x)), create_superclass(x)))

def print_labels():
    for i in sorted_labels:
        print(f"{get_labels(i, 'sub')}|{get_labels(create_superclass(i), 'super')}|{get_labels(create_ultraclass(create_superclass(i)), 'ultra')}")