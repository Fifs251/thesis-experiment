from plotly.io import write_image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from hierarchy import *
import numpy as np

#re-defined labels with <br> line breaks

def get_labels(idx, level='super'):
    labels = {
        #'ultra': ['mammal', 'plant', 'household', 'invertebrates', 'other animal', 'vehicles', 'nature'],
        'ultra': ['buildings', 'household', 'invertebrates', 'mammal', 'outdoors', 'other animal', 'plant', 'vehicles'],
                
        'super': ['aquatic<br>mammals', 'fish', 'flowers', 'food<br>containers', 'fruit & vegetables',
                  'household<br>electrical<br>device', 'household<br>furniture', 'insects', 'large<br>carnivores',
                  'large man-made<br>outdoor<br>things', 'large natural <br> outdoor scenes', 'large omnivores<br>and herbivores',
                  'medium-sized<br>mammals', 'non-insect<br>invertebrates', 'people', 'reptiles', 'small<br>mammals', 'trees', 'vehicles 1', 'vehicles 2'
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

df = pd.DataFrame({'subs': v_get_labels(range(100), 'sub')})
df['super'] = v_get_labels(v_create_superclass(range(100)), 'super')
df['ultra'] = v_get_labels(v_create_ultraclass(v_create_superclass(range(100))), 'ultra')
df['nr'] = 1

#print(df.tail())

fig = px.sunburst(
    #ids=df.subs,
    #labels=df.super,
    #parents=df.ultra
    df,
    path=['ultra', 'super', 'subs'],
    values='nr',
)

fig.update_layout(
    font_size=30,
    font_family="sans-serif",
    #font_color="blue",
    #title_font_family="sans-serif",
    #title_font_color="red",
    #legend_title_font_color="green"
)

#df = px.data.tips()
#fig = px.sunburst(df, path=['sex', 'day', 'time'], values='total_bill', color='time',
#                  color_discrete_map={'(?)':'black', 'Lunch':'gold', 'Dinner':'darkblue'})

fig.write_image('figs/sunburst.png')
fig.show()